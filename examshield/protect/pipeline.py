import csv
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from examshield import __version__
from examshield.check.compare import PIXEL_TOLERANCE, CheckResult, Expectation, Ocr, compare
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import Profile
from examshield.protect.seed import derive_seed
from examshield.protect.techniques import decoy, dots, glyph_remap, rasterize, watermark
from examshield.protect.techniques.base import TEXT_ALTERED, TEXT_PRESERVED, Context, Outcome
from examshield.verify.metrics import legibility, mean

JOB_SCHEMA = 1
JOB_SUFFIX = ".job.json"
LOG_NAME = "protect.log.jsonl"

TECHNIQUES: dict[str, Callable[[bytes, dict[str, Any], Context], Outcome]] = {
    "rasterize": rasterize.apply,
    "glyph_remap": glyph_remap.apply,
    "decoy": decoy.apply,
    "watermark": watermark.apply,
    "dots": dots.apply,
}
# 수험번호별 키를 받아 복사본마다 다른 무늬를 넣는 기법. 프로파일의 맨 마지막에 하나만 온다.
KEYED = ("watermark", "dots")
# 인쇄하면 사라지는 기법(텍스트 레이어, 2% 밝기 차이)은 시나리오 B 프로파일에 넣지 않는다.
SCENARIO_TECHNIQUES = {
    "A": ("rasterize", "glyph_remap", "decoy", "watermark"),
    "B": ("dots",),
}


# 쪽 번호(1부터) → 그 쪽에 있는 문항들의 정답 번호.
AnswerKey = dict[int, frozenset[int]]
MAX_CHOICE = 9


def read_answer_key(path: Path) -> AnswerKey:
    """정답표 CSV를 읽는다. 한 줄에 '쪽,정답' 하나. 한 쪽에 문항이 여럿이면 줄을 여러 개 쓴다."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ExamShieldError(f"정답표를 읽지 못했다: {path}") from error
    key: dict[int, set[int]] = {}
    for number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        if number == 1 and not cells[0].isdigit():
            continue
        if len(cells) < 2 or not cells[0].isdigit() or not cells[1].isdigit():
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: '쪽,정답' 형식이어야 한다. 예: 3,2"
            )
        page, answer = int(cells[0]), int(cells[1])
        if page < 1 or not 1 <= answer <= MAX_CHOICE:
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: 쪽은 1 이상, 정답은 1~{MAX_CHOICE}이어야 한다."
            )
        key.setdefault(page, set()).add(answer)
    if not key:
        raise ExamShieldError(f"정답표가 비어 있다: {path}")
    return {page: frozenset(answers) for page, answers in key.items()}


def _avoid(
    answer_key: AnswerKey | None, pages: int, profile: Profile
) -> tuple[frozenset[int], ...] | None:
    if answer_key is None:
        return None
    if not any(entry.id == "decoy" for entry in profile.techniques):
        raise ExamShieldError(
            f"프로파일 '{profile.name}'에는 미끼가 없어 정답표를 쓸 곳이 없다. "
            "a-remap-decoy나 a-full과 함께 쓴다."
        )
    beyond = sorted(page for page in answer_key if page > pages)
    if beyond:
        raise ExamShieldError(f"정답표에 문서에 없는 쪽이 있다: {beyond[0]}쪽 (문서는 {pages}쪽)")
    return tuple(answer_key.get(page, frozenset()) for page in range(1, pages + 1))


@dataclass(frozen=True)
class Protected:
    pdf: bytes
    record: dict[str, Any]
    expectation: Expectation
    decoy_answers: tuple[int, ...] | None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protect_bytes(
    pdf: bytes,
    profile: Profile,
    seed_material: bytes | None = None,
    watermark_key: bytes | None = None,
    answer_key: AnswerKey | None = None,
) -> Protected:
    unknown = [entry.id for entry in profile.techniques if entry.id not in TECHNIQUES]
    if unknown:
        raise ExamShieldError(
            f"프로파일 '{profile.name}'에 모르는 기법이 있다: {', '.join(unknown)}"
        )
    order = [entry.id for entry in profile.techniques]
    misplaced = [name for name in order if name not in SCENARIO_TECHNIQUES[profile.scenario]]
    if misplaced:
        raise ExamShieldError(
            f"프로파일 '{profile.name}'(시나리오 {profile.scenario})에 맞지 않는 기법이 있다: "
            f"{', '.join(misplaced)}. 시나리오 {profile.scenario}에서 쓰는 기법은 "
            f"{', '.join(SCENARIO_TECHNIQUES[profile.scenario])}이다."
        )
    if "decoy" in order and "glyph_remap" in order[order.index("decoy") :]:
        raise ExamShieldError(
            f"프로파일 '{profile.name}': glyph_remap은 decoy보다 먼저 와야 한다. 뒤에 오면 미끼 문구까지 섞인다."
        )
    if any(name in KEYED for name in order[:-1]):
        raise ExamShieldError(
            f"프로파일 '{profile.name}': watermark와 dots는 맨 마지막 기법이어야 한다."
        )
    try:
        source = open_pdf_bytes(pdf)
    except Exception as error:
        raise ExamShieldError("PDF를 열지 못했다.") from error
    if not source.is_pdf:
        raise ExamShieldError("PDF가 아니다. 확장자만 .pdf인 다른 형식의 파일일 수 있다.")
    if source.needs_pass or source.is_encrypted:
        raise ExamShieldError("암호가 걸린 PDF는 처리하지 않는다.")
    if source.page_count == 0:
        raise ExamShieldError("쪽이 없는 PDF다.")
    originals = tuple(page_text(page) for page in source)
    input_hash = sha256_hex(pdf)
    context = Context(
        seed=derive_seed(input_hash, profile.sha256, seed_material),
        original_texts=originals,
        watermark_key=watermark_key,
        avoid_answers=_avoid(answer_key, len(originals), profile),
    )

    current = pdf
    text = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoy_texts: tuple[str, ...] = ()
    decoy_answers: tuple[int, ...] | None = None
    applied: list[dict[str, Any]] = []
    overlay_images = 0
    overlay_cell_pt: int | None = None
    dots_spec: dict[str, int] | None = None
    pixel_tolerance = PIXEL_TOLERANCE
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
        overlay_images += outcome.overlay_images
        if outcome.pixel_tolerance is not None:
            pixel_tolerance = outcome.pixel_tolerance
        if outcome.dots is not None:
            dots_spec = outcome.dots
        elif outcome.overlay_images:
            overlay_cell_pt = int(outcome.stats["cell_pt"])
        applied.append({"id": entry.id, "params": entry.params, "stats": outcome.stats})

    expectation = Expectation(
        text=text,
        flattened_dpi=flattened_dpi,
        decoys=decoy_texts,
        overlay_images=overlay_images,
        pixel_tolerance=pixel_tolerance,
        overlay_cell_pt=overlay_cell_pt,
        dots_cell_pt=dots_spec["cell_pt"] if dots_spec else None,
        dots_ink=dots_spec["ink"] if dots_spec else None,
    )
    copies = tuple(page_text(page) for page in open_pdf_bytes(current))
    similarity = mean(
        [legibility(after, before) for before, after in zip(originals, copies, strict=True)]
    )
    record: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "tool_version": __version__,
        "profile": {"name": profile.name, "sha256": profile.sha256},
        "seed": "file" if seed_material is not None else "default",
        # 정답표를 썼는지만 적는다. 정답 자체는 기록에 남기지 않는다.
        "answer_key": answer_key is not None,
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
    pdf: bytes,
    profile: Profile,
    seed_material: bytes | None = None,
    ocr: Ocr | None = None,
    answer_key: AnswerKey | None = None,
) -> tuple[Protected, CheckResult]:
    protected = protect_bytes(pdf, profile, seed_material, None, answer_key)
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
        text = job_file.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        raise ExamShieldError(f"작업 기록을 읽지 못했다: {job_file}") from error
    return expectation_from_text(text, copy, str(job_file))


def expectation_from_text(text: str, copy: bytes, label: str) -> Expectation:
    try:
        record: dict[str, Any] = json.loads(text)
        expect = Expectation.from_dict(dict(record["expect"]))
        recorded = str(record["output_sha256"])
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        raise ExamShieldError(f"작업 기록을 읽지 못했다: {label}") from error
    if recorded != sha256_hex(copy):
        raise ExamShieldError(
            f"작업 기록({Path(label).name})이 이 복사본의 것이 아니다. 해시가 다르다."
        )
    return expect
