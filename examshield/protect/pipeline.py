import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from examshield import __version__
from examshield.check.compare import CheckResult, Expectation, Ocr, compare
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import Profile
from examshield.protect.seed import derive_seed
from examshield.protect.techniques import decoy, glyph_remap, rasterize
from examshield.protect.techniques.base import TEXT_ALTERED, TEXT_PRESERVED, Context, Outcome
from examshield.verify.metrics import legibility, mean

JOB_SCHEMA = 1
JOB_SUFFIX = ".job.json"
LOG_NAME = "protect.log.jsonl"

TECHNIQUES: dict[str, Callable[[bytes, dict[str, Any], Context], Outcome]] = {
    "rasterize": rasterize.apply,
    "glyph_remap": glyph_remap.apply,
    "decoy": decoy.apply,
}


@dataclass(frozen=True)
class Protected:
    pdf: bytes
    record: dict[str, Any]
    expectation: Expectation
    decoy_answers: tuple[int, ...] | None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protect_bytes(pdf: bytes, profile: Profile, seed_material: bytes | None = None) -> Protected:
    if profile.scenario != "A":
        raise ExamShieldError(
            f"프로파일 '{profile.name}'은 시나리오 {profile.scenario}용이다. "
            "protect는 지금 시나리오 A(디지털 배포) 프로파일만 지원한다."
        )
    unknown = [entry.id for entry in profile.techniques if entry.id not in TECHNIQUES]
    if unknown:
        raise ExamShieldError(
            f"프로파일 '{profile.name}'에 모르는 기법이 있다: {', '.join(unknown)}"
        )
    order = [entry.id for entry in profile.techniques]
    if "decoy" in order and "glyph_remap" in order[order.index("decoy") :]:
        raise ExamShieldError(
            f"프로파일 '{profile.name}': glyph_remap은 decoy보다 먼저 와야 한다. 뒤에 오면 미끼 문구까지 섞인다."
        )
    try:
        source = open_pdf_bytes(pdf)
    except Exception as error:
        raise ExamShieldError("PDF를 열지 못했다.") from error
    if source.needs_pass or source.is_encrypted:
        raise ExamShieldError("암호가 걸린 PDF는 처리하지 않는다.")
    if source.page_count == 0:
        raise ExamShieldError("쪽이 없는 PDF다.")
    originals = tuple(page_text(page) for page in source)
    input_hash = sha256_hex(pdf)
    context = Context(
        seed=derive_seed(input_hash, profile.sha256, seed_material), original_texts=originals
    )

    current = pdf
    text = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoy_texts: tuple[str, ...] = ()
    decoy_answers: tuple[int, ...] | None = None
    applied: list[dict[str, Any]] = []
    for entry in profile.techniques:
        outcome = TECHNIQUES[entry.id](current, entry.params, context)
        current = outcome.pdf
        if outcome.text == TEXT_ALTERED:
            text = TEXT_ALTERED
        if outcome.flattened_dpi is not None:
            flattened_dpi = outcome.flattened_dpi
            decoy_texts, decoy_answers = (), None
        if outcome.decoy_texts is not None:
            decoy_texts, decoy_answers = outcome.decoy_texts, outcome.decoy_answers
        applied.append({"id": entry.id, "params": entry.params, "stats": outcome.stats})

    expectation = Expectation(text=text, flattened_dpi=flattened_dpi, decoys=decoy_texts)
    copies = tuple(page_text(page) for page in open_pdf_bytes(current))
    similarity = mean(
        [legibility(after, before) for before, after in zip(originals, copies, strict=True)]
    )
    record: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "tool_version": __version__,
        "profile": {"name": profile.name, "sha256": profile.sha256},
        "seed": "file" if seed_material is not None else "default",
        "input_sha256": input_hash,
        "output_sha256": sha256_hex(current),
        "pages": len(originals),
        "techniques": applied,
        "text_layer_similarity": round(similarity or 0.0, 4),
        "expect": expectation.as_dict(),
        "decoy_answers": list(decoy_answers) if decoy_answers is not None else None,
    }
    return Protected(
        pdf=current, record=record, expectation=expectation, decoy_answers=decoy_answers
    )


def protect_and_check(
    pdf: bytes, profile: Profile, seed_material: bytes | None = None, ocr: Ocr | None = None
) -> tuple[Protected, CheckResult]:
    protected = protect_bytes(pdf, profile, seed_material)
    result = compare(pdf, protected.pdf, protected.expectation, ocr)
    protected.record["check"] = result.as_dict()
    return protected, result


def job_path(copy_path: Path) -> Path:
    return copy_path.with_name(copy_path.stem + JOB_SUFFIX)


def write_outputs(protected: Protected, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_path = out_dir / f"{stem}.pdf"
    copy_path.write_bytes(protected.pdf)
    job_path(copy_path).write_text(
        json.dumps(protected.record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return copy_path


def append_log(out_dir: Path, protected: Protected, passed: bool, now: datetime) -> None:
    """S4: 해시, 프로파일 이름, 시각만 남긴다."""
    line = {
        "time": now.isoformat(timespec="seconds"),
        "profile": protected.record["profile"]["name"],
        "input_sha256": protected.record["input_sha256"],
        "output_sha256": protected.record["output_sha256"],
        "check_passed": passed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / LOG_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_expectation(job_file: Path, copy: bytes) -> Expectation:
    try:
        record: dict[str, Any] = json.loads(job_file.read_text(encoding="utf-8"))
        expect = Expectation.from_dict(dict(record["expect"]))
        recorded = str(record["output_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExamShieldError(f"작업 기록을 읽지 못했다: {job_file}") from error
    if recorded != sha256_hex(copy):
        raise ExamShieldError(
            f"작업 기록({job_file.name})이 이 복사본의 것이 아니다. 해시가 다르다."
        )
    return expect
