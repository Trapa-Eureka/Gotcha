import csv
import hashlib
import io
import json
import unicodedata
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


MAX_CHOICE = 9


@dataclass(frozen=True)
class KeyEntry:
    """정답표의 한 줄. 문항 번호가 없으면 그 쪽에 있는 어느 문항의 정답이라는 뜻이다."""

    page: int
    question: int | None
    answer: int


AnswerKey = tuple[KeyEntry, ...]


def read_answer_key(path: Path) -> AnswerKey:
    """정답표 CSV를 읽는다. 한 줄에 '쪽,정답' 또는 '쪽,문항,정답'. 두 형식을 섞어 쓰지 않는다."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ExamShieldError(f"정답표를 읽지 못했다: {path}") from error
    entries: list[KeyEntry] = []
    for number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        cells = [cell.strip() for cell in row]
        while cells and not cells[-1]:
            cells.pop()
        if not cells:
            continue
        if number == 1 and not cells[0].isdigit():
            continue
        if len(cells) not in (2, 3) or not all(cell.isdigit() for cell in cells):
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: '쪽,정답' 또는 '쪽,문항,정답' 형식이어야 한다. "
                "예: 3,2 또는 3,12,2"
            )
        values = [int(cell) for cell in cells]
        page, answer = values[0], values[-1]
        question = values[1] if len(values) == 3 else None
        if page < 1 or not 1 <= answer <= MAX_CHOICE or (question is not None and question < 1):
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: 쪽과 문항은 1 이상, 정답은 1~{MAX_CHOICE}이어야 한다."
            )
        entries.append(KeyEntry(page, question, answer))
    if not entries:
        raise ExamShieldError(f"정답표가 비어 있다: {path}")
    if len({entry.question is None for entry in entries}) != 1:
        raise ExamShieldError(
            f"{path.name}: 문항 번호가 있는 줄과 없는 줄이 섞여 있다. 한 가지 형식으로 쓴다."
        )
    numbered = [entry.question for entry in entries if entry.question is not None]
    if len(set(numbered)) != len(numbered):
        raise ExamShieldError(f"{path.name}: 같은 문항 번호가 두 번 나온다.")
    return tuple(entries)


def key_for_pages(
    answer_key: AnswerKey | None, pages: int, profile: Profile
) -> tuple[tuple[frozenset[int], ...] | None, tuple[tuple[tuple[int, int], ...], ...] | None]:
    """정답표를 쪽별로 나눈다. (쪽마다 피할 번호, 쪽마다 (문항, 정답) 목록)을 돌려준다."""
    if answer_key is None:
        return None, None
    if not any(entry.id == "decoy" for entry in profile.techniques):
        raise ExamShieldError(
            f"프로파일 '{profile.name}'에는 미끼가 없어 정답표를 쓸 곳이 없다. "
            "a-remap-decoy나 a-full과 함께 쓴다."
        )
    beyond = sorted(entry.page for entry in answer_key if entry.page > pages)
    if beyond:
        raise ExamShieldError(f"정답표에 문서에 없는 쪽이 있다: {beyond[0]}쪽 (문서는 {pages}쪽)")
    avoid = tuple(
        frozenset(entry.answer for entry in answer_key if entry.page == page)
        for page in range(1, pages + 1)
    )
    if not answer_key or answer_key[0].question is None:
        return avoid, None
    questions = tuple(
        tuple(
            (entry.question, entry.answer)
            for entry in answer_key
            if entry.page == page and entry.question is not None
        )
        for page in range(1, pages + 1)
    )
    return avoid, questions


@dataclass(frozen=True)
class Protected:
    pdf: bytes
    record: dict[str, Any]
    expectation: Expectation
    decoy_answers: tuple[int, ...] | None
    decoy_items: tuple[tuple[int, int | None, int], ...] | None = None


def letters_only(text: str) -> str:
    """글자와 숫자만 남긴 글. 공백, 문장부호, 기호를 뺀다."""
    return "".join(char for char in text if unicodedata.category(char)[0] in "LN")


def letters_kept(before: str, after: str, decoy: str = "") -> float:
    """원문의 글자와 숫자 가운데 복사본에서 같은 자리에 그대로 읽히는 비율.

    글자 바꾸기는 글자를 하나씩 다른 글자로 바꾸므로 글자 수가 같다. 그때는 자리끼리 견준다.
    글자 수가 달라졌으면(이미지화, 미끼 문구) 글 전체의 닮은 정도로 본다.
    """
    left, right = letters_only(before), letters_only(after)
    # 숨은 미끼 문구는 원문에 없던 글이다. 빼고 견준다.
    right = right.replace(letters_only(decoy), "", 1) if decoy else right
    if len(left) == len(right):
        return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)
    return legibility(right, left)


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
    avoid, questions = key_for_pages(answer_key, len(originals), profile)
    context = Context(
        seed=derive_seed(input_hash, profile.sha256, seed_material),
        original_texts=originals,
        watermark_key=watermark_key,
        avoid_answers=avoid,
        questions=questions,
    )

    current = pdf
    text = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoy_texts: tuple[str, ...] = ()
    decoy_answers: tuple[int, ...] | None = None
    decoy_items: tuple[tuple[int, int | None, int], ...] | None = None
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
            decoy_texts, decoy_answers, decoy_items = (), None, None
        if outcome.decoy_texts is not None:
            decoy_texts, decoy_answers = outcome.decoy_texts, outcome.decoy_answers
            decoy_items = outcome.decoy_items
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
    # 공백과 문장부호는 어떤 기법도 바꾸지 않아서, 위의 값은 글자를 모두 바꿔도 10~20%가 남는다.
    # 글자와 숫자만 놓고 본 값을 따로 적는다. 글자가 없는 쪽은 세지 않는다.
    hidden = decoy_texts or ("",) * len(copies)
    letter_similarity = mean(
        [
            letters_kept(before, after, decoy)
            for before, after, decoy in zip(originals, copies, hidden, strict=True)
            if letters_only(before)
        ]
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
        "text_layer_letter_similarity": (
            round(letter_similarity, 4) if letter_similarity is not None else None
        ),
        "expect": expectation.as_dict(),
        "decoy_answers": list(decoy_answers) if decoy_answers is not None else None,
        "decoy_items": decoy_items_record(decoy_items),
    }
    return Protected(
        pdf=current,
        record=record,
        expectation=expectation,
        decoy_answers=decoy_answers,
        decoy_items=decoy_items,
    )


def decoy_items_record(
    items: tuple[tuple[int, int | None, int], ...] | None,
) -> list[dict[str, int | None]] | None:
    if items is None:
        return None
    return [
        {"page": page, "question": question, "answer": answer} for page, question, answer in items
    ]


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
