from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
MAX_PIXEL_TOLERANCE = 14
WATERMARK_SLACK = 2
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
    overlay_images: int = 0
    pixel_tolerance: int = PIXEL_TOLERANCE
    overlay_cell_pt: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "flattened_dpi": self.flattened_dpi,
            "decoys": list(self.decoys),
            "overlay_images": self.overlay_images,
            "pixel_tolerance": self.pixel_tolerance,
            "overlay_cell_pt": self.overlay_cell_pt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Expectation":
        text = str(data.get("text", TEXT_PRESERVED))
        if text not in (TEXT_PRESERVED, TEXT_ALTERED):
            raise ExamShieldError(f"작업 기록의 expect.text 값이 잘못됐다: {text!r}")
        dpi = data.get("flattened_dpi")
        overlay = int(data.get("overlay_images", 0))
        tolerance = int(data.get("pixel_tolerance", PIXEL_TOLERANCE))
        cell = data.get("overlay_cell_pt")
        relaxed = overlay == 1 or tolerance != PIXEL_TOLERANCE
        if (
            overlay not in (0, 1)
            or not PIXEL_TOLERANCE <= tolerance <= MAX_PIXEL_TOLERANCE
            or (relaxed and (overlay != 1 or cell is None or not 4 <= int(cell) <= 24))
        ):
            raise ExamShieldError("작업 기록의 워터마크 허용값이 범위를 벗어났다.")
        return cls(
            overlay_images=overlay,
            pixel_tolerance=tolerance,
            overlay_cell_pt=int(cell) if cell is not None else None,
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
        if expect.overlay_images and not _looks_like_watermark(left, right, expect, dpi):
            result.differences.append(
                f"{number}쪽: 덧씌운 무늬가 워터마크 격자 모양이 아니다. 다른 내용이 옅게 들어갔을 수 있다"
            )
        page_max = int(delta.max())
        max_diff = max(max_diff, page_max)
        changed += int((delta > expect.pixel_tolerance).sum())
        total += int(delta.size)
        if page_max > expect.pixel_tolerance:
            result.differences.append(
                f"{number}쪽: 렌더링이 다르다 (최대 밝기 차이 {page_max}/255, 허용 {expect.pixel_tolerance})"
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


def _looks_like_watermark(
    before: NDArray[np.uint8], after: NDArray[np.uint8], expect: Expectation, dpi: int
) -> bool:
    """빈 바탕의 칸마다 어두워진 정도가 고르고, 값이 '그대로'와 '무늬 세기' 두 가지뿐인지 본다.

    옅은 글자나 그림을 덧씌우면 칸 안이 고르지 않거나 중간값이 나온다.
    칸 단위로 그린 무늬는 워터마크와 구별하지 못한다.
    """
    cell = expect.overlay_cell_pt
    if cell is None:
        return False
    strength = expect.pixel_tolerance - WATERMARK_SLACK
    light = before.min(axis=2)
    darker = light.astype(np.float64) - after.min(axis=2).astype(np.float64)
    step = cell * dpi / 72.0
    inset = step * 0.25
    height, width = light.shape
    for row in range(int(height / step)):
        top, bottom = round(row * step + inset), round((row + 1) * step - inset)
        for col in range(int(width / step)):
            left, right = round(col * step + inset), round((col + 1) * step - inset)
            if bottom <= top or right <= left or int(np.min(light[top:bottom, left:right])) < 250:
                continue
            patch = darker[top:bottom, left:right]
            level = float(np.mean(patch))
            if float(np.max(patch) - np.min(patch)) > 1.0:
                return False
            if abs(level) > 1.0 and abs(level - strength) > 1.5:
                return False
    return True


def _same_text(before: str, after: str, decoy: str) -> bool:
    left, right = normalize_text(before), normalize_text(after)
    if decoy:
        right = normalize_text(right.replace(normalize_text(decoy), "", 1))
    return left == right


def _compare_images(
    number: int, before: Any, after: Any, expect: Expectation, result: CheckResult
) -> None:
    # 쪽에 실제로 그려진 그림을 센다. 자원 목록으로 세면 쪽끼리 자원을 같이 쓰는 문서에서 틀린다.
    had, has = len(before.get_image_info()), len(after.get_image_info())
    if expect.flattened_dpi is not None:
        if has != 1 + expect.overlay_images:
            result.differences.append(f"{number}쪽: 이미지화한 쪽인데 그림이 {has}개다")
    elif had + expect.overlay_images != has:
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
