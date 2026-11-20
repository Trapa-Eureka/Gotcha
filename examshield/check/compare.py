from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from examshield.errors import ExamShieldError
from examshield.pdfio import (
    normalize_text,
    open_pdf_bytes,
    page_text,
    render_gray,
    render_rgb,
)
from examshield.verify.capture_sim import encode_png
from examshield.verify.metrics import legibility, mean

CHECK_DPI = 150
MIN_FLATTEN_DPI = 150
PIXEL_TOLERANCE = 2
SIZE_TOLERANCE = 0.01
OCR_TOLERANCE = 0.02
TEXT_PRESERVED = "preserved"
TEXT_ALTERED = "altered"

Ocr = Callable[[bytes], str]


@dataclass(frozen=True)
class Expectation:
    text: str = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoys: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "flattened_dpi": self.flattened_dpi, "decoys": list(self.decoys)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Expectation":
        text = str(data.get("text", TEXT_PRESERVED))
        if text not in (TEXT_PRESERVED, TEXT_ALTERED):
            raise ExamShieldError(f"작업 기록의 expect.text 값이 잘못됐다: {text!r}")
        dpi = data.get("flattened_dpi")
        return cls(
            text=text,
            flattened_dpi=int(dpi) if dpi is not None else None,
            decoys=tuple(str(entry) for entry in data.get("decoys") or ()),
        )


@dataclass
class CheckResult:
    pages_original: int
    pages_copy: int
    text: str = "not_compared"
    render_dpi: int = CHECK_DPI
    visual_max_diff: int | None = None
    visual_changed_ratio: float | None = None
    ocr: dict[str, Any] | None = None
    differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.differences

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pages": {"original": self.pages_original, "copy": self.pages_copy},
            "text": self.text,
            "render_dpi": self.render_dpi,
            "visual_max_diff": self.visual_max_diff,
            "visual_changed_ratio": self.visual_changed_ratio,
            "ocr": self.ocr,
            "differences": list(self.differences),
            "notes": list(self.notes),
        }


def compare(
    original: bytes,
    copy: bytes,
    expect: Expectation | None = None,
    ocr: Ocr | None = None,
) -> CheckResult:
    expect = expect or Expectation()
    try:
        source, target = open_pdf_bytes(original), open_pdf_bytes(copy)
    except Exception as error:
        raise ExamShieldError("PDF를 열지 못했다. 원본과 복사본이 PDF인지 확인한다.") from error
    dpi = expect.flattened_dpi or CHECK_DPI
    result = CheckResult(
        pages_original=source.page_count, pages_copy=target.page_count, render_dpi=dpi
    )
    if source.page_count != target.page_count:
        result.differences.append(
            f"쪽수가 다르다: 원본 {source.page_count}, 복사본 {target.page_count}"
        )
        return result
    if expect.flattened_dpi is not None and expect.flattened_dpi < MIN_FLATTEN_DPI:
        result.differences.append(
            f"이미지화 해상도가 너무 낮다 ({expect.flattened_dpi}dpi, 최소 {MIN_FLATTEN_DPI}dpi). "
            "작은 글자가 뭉개질 수 있다."
        )
    if expect.decoys and len(expect.decoys) != target.page_count:
        raise ExamShieldError("작업 기록의 미끼 수가 쪽수와 다르다.")

    text_same = True
    max_diff = 0
    changed = 0
    total = 0
    for number, (before, after) in enumerate(zip(source, target, strict=True), start=1):
        if (
            abs(before.rect.width - after.rect.width) > SIZE_TOLERANCE
            or abs(before.rect.height - after.rect.height) > SIZE_TOLERANCE
        ):
            result.differences.append(f"{number}쪽: 쪽 크기가 다르다")
            continue
        _compare_images(number, before, after, expect, result)
        decoy = expect.decoys[number - 1] if expect.decoys else ""
        if not _same_text(page_text(before), page_text(after), decoy):
            text_same = False
            if expect.text == TEXT_PRESERVED:
                result.differences.append(f"{number}쪽: 추출한 텍스트가 다르다")
        left, right = render_rgb(before, dpi), render_rgb(after, dpi)
        if left.shape != right.shape:
            result.differences.append(f"{number}쪽: 렌더링 크기가 다르다")
            continue
        delta = np.abs(left.astype(np.int16) - right.astype(np.int16)).max(axis=2)
        page_max = int(delta.max())
        max_diff = max(max_diff, page_max)
        changed += int((delta > PIXEL_TOLERANCE).sum())
        total += int(delta.size)
        if page_max > PIXEL_TOLERANCE:
            result.differences.append(
                f"{number}쪽: 렌더링이 다르다 (최대 밝기 차이 {page_max}/255, 허용 {PIXEL_TOLERANCE})"
            )
    result.visual_max_diff = max_diff
    result.visual_changed_ratio = changed / total if total else 0.0

    if text_same:
        result.text = "identical_without_decoys" if expect.decoys else "identical"
    elif expect.text == TEXT_ALTERED:
        result.text = "altered_as_expected"
        _compare_by_ocr(source, target, dpi, ocr, result)
    else:
        result.text = "different"
        result.notes.append(
            "텍스트 레이어를 교란하는 프로파일로 만든 복사본이면 작업 기록(.job.json)을 --job으로 준다."
        )
    return result


def _same_text(before: str, after: str, decoy: str) -> bool:
    left, right = normalize_text(before), normalize_text(after)
    if decoy:
        right = normalize_text(right.replace(normalize_text(decoy), "", 1))
    return left == right


def _compare_images(
    number: int, before: Any, after: Any, expect: Expectation, result: CheckResult
) -> None:
    had, has = len(before.get_images(full=True)), len(after.get_images(full=True))
    if expect.flattened_dpi is not None:
        if has != 1:
            result.differences.append(f"{number}쪽: 이미지화한 쪽인데 그림이 {has}개다")
    elif had != has:
        result.differences.append(f"{number}쪽: 그림 개수가 다르다 (원본 {had}, 복사본 {has})")


def _compare_by_ocr(
    source: Any, target: Any, dpi: int, ocr: Ocr | None, result: CheckResult
) -> None:
    if ocr is None:
        result.notes.append("OCR 비교는 건너뛰었다(Tesseract 없음). 렌더링 비교로만 판정했다.")
        return
    before_scores: list[float] = []
    after_scores: list[float] = []
    for before, after in zip(source, target, strict=True):
        truth = page_text(before)
        left = encode_png(render_gray(before, dpi))
        right = encode_png(render_gray(after, dpi))
        score = legibility(ocr(left), truth)
        before_scores.append(score)
        after_scores.append(score if left == right else legibility(ocr(right), truth))
    original_score, copy_score = mean(before_scores) or 0.0, mean(after_scores) or 0.0
    result.ocr = {"original": round(original_score, 4), "copy": round(copy_score, 4)}
    if copy_score < original_score - OCR_TOLERANCE:
        result.differences.append(
            f"OCR 판독률이 원본보다 낮다 (원본 {original_score:.1%}, 복사본 {copy_score:.1%})"
        )


def format_result(result: CheckResult) -> str:
    labels = {
        "identical": "같다",
        "identical_without_decoys": "미끼 문구를 빼면 같다",
        "altered_as_expected": "다르다 (프로파일이 의도한 교란)",
        "different": "다르다",
        "not_compared": "비교하지 못했다",
    }
    lines = [
        f"쪽수         원본 {result.pages_original}, 복사본 {result.pages_copy}",
        f"텍스트 레이어 {labels[result.text]}",
    ]
    if result.visual_max_diff is not None and result.visual_changed_ratio is not None:
        lines.append(
            f"렌더링 비교  {result.render_dpi}dpi, 최대 밝기 차이 {result.visual_max_diff}/255, "
            f"달라진 픽셀 {result.visual_changed_ratio:.4%}"
        )
    if result.ocr:
        lines.append(
            f"OCR 판독률   원본 {result.ocr['original']:.1%}, 복사본 {result.ocr['copy']:.1%}"
        )
    lines += [f"차이: {entry}" for entry in result.differences]
    lines += [f"참고: {entry}" for entry in result.notes]
    lines.append("결과         " + ("통과" if result.passed else "실패"))
    return "\n".join(lines)
