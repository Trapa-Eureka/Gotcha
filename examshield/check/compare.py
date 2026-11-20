import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymupdf
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
DOTS_PX_PER_PT = 2
DOTS_CELL_RANGE = (4, 16)
DOTS_INK_RANGE = (64, 255)
# 무늬는 칸의 절반쯤에 점을 찍는다. 크게 벗어나면 무늬가 아니라 점으로 그린 그림일 수 있다.
DOTS_FILL_RANGE = (0.35, 0.65)
SIZE_TOLERANCE = 0.01
OCR_TOLERANCE = 0.02
TEXT_PRESERVED = "preserved"
TEXT_ALTERED = "altered"

Ocr = Callable[[bytes], str]
_NUMBER = rb"[-+]?[0-9]*\.?[0-9]+"
OVERLAY_DRAW = re.compile(
    rb"\s*q\s+(?:" + _NUMBER + rb"\s+){6}cm\s+/(?P<name>[A-Za-z0-9_]+)\s+Do\s+Q\s*"
)


@dataclass(frozen=True)
class Expectation:
    text: str = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoys: tuple[str, ...] = ()
    overlay_images: int = 0
    pixel_tolerance: int = PIXEL_TOLERANCE
    overlay_cell_pt: int | None = None
    dots_cell_pt: int | None = None
    dots_ink: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "flattened_dpi": self.flattened_dpi,
            "decoys": list(self.decoys),
            "overlay_images": self.overlay_images,
            "pixel_tolerance": self.pixel_tolerance,
            "overlay_cell_pt": self.overlay_cell_pt,
            "dots_cell_pt": self.dots_cell_pt,
            "dots_ink": self.dots_ink,
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
        dots_cell, dots_ink = data.get("dots_cell_pt"), data.get("dots_ink")
        if dots_cell is not None or dots_ink is not None:
            if (
                dots_cell is None
                or dots_ink is None
                or overlay != 1
                or tolerance != PIXEL_TOLERANCE
                or cell is not None
                or not DOTS_CELL_RANGE[0] <= int(dots_cell) <= DOTS_CELL_RANGE[1]
                or not DOTS_INK_RANGE[0] <= int(dots_ink) <= DOTS_INK_RANGE[1]
            ):
                raise ExamShieldError("작업 기록의 도트 무늬 값이 범위를 벗어났다.")
            return cls(
                text=text,
                flattened_dpi=int(dpi) if dpi is not None else None,
                decoys=tuple(str(entry) for entry in data.get("decoys") or ()),
                overlay_images=1,
                dots_cell_pt=int(dots_cell),
                dots_ink=int(dots_ink),
            )
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
    # 꺼진 레이어의 내용은 렌더링에 안 나오지만 인쇄 설정에 따라 찍힐 수 있다.
    if len(source.get_ocgs()) != len(target.get_ocgs()):
        result.differences.append("선택 표시 레이어(OCG)의 개수가 다르다")

    text_same = True
    redrawn: list[int] = []
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
        # 화면에는 안 보이고 인쇄에만 나오는 주석이 있다. 렌더링 비교로는 잡지 못해 개수로 본다.
        if len(list(before.annots())) != len(list(after.annots())):
            result.differences.append(f"{number}쪽: 주석의 개수가 다르다")
        if expect.dots_cell_pt is not None:
            problem = _lift_dots(target, after, expect)
            if problem:
                result.differences.append(f"{number}쪽: {problem}")
                continue
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
        if (
            expect.overlay_images
            and expect.dots_cell_pt is None
            and not _looks_like_watermark(left, right, expect, dpi)
        ):
            result.differences.append(
                f"{number}쪽: 덧씌운 무늬가 워터마크 격자 모양이 아니다. 다른 내용이 옅게 들어갔을 수 있다"
            )
        page_max = int(delta.max())
        if page_max:
            redrawn.append(number - 1)
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
        _compare_by_ocr(source, target, redrawn, dpi, ocr, result)
    else:
        result.text = "different"
        result.notes.append(
            "텍스트 레이어를 교란하는 프로파일로 만든 복사본이면 작업 기록(.job.json)을 --job으로 준다."
        )
    return result


def _lift_dots(doc: Any, page: Any, expect: Expectation) -> str | None:
    """도트 무늬 그림이 약속한 모양인지 확인하고 쪽에서 걷어 낸다. 문제가 있으면 그 내용을 돌려준다.

    약속한 모양: 쪽 전체를 덮는 그림 한 장, 점은 칸 가운데에만, 진하기는 기록된 값 하나뿐.
    걷어 낸 뒤의 쪽은 원본과 픽셀 단위로 같아야 한다(호출한 쪽에서 비교한다).
    칸 가운데 점만으로 그린 큰 글자나 그림은 무늬와 구별하지 못한다.
    """
    cell, ink = expect.dots_cell_pt, expect.dots_ink
    if cell is None or ink is None:
        return "도트 무늬 값이 없다"
    # 쪽들이 그림 자원을 같이 쓰는 문서에서는 그림 정보의 번호가 틀릴 수 있다.
    # 그래서 쪽의 마지막 내용 줄기가 "그림 하나를 쪽 전체에 그린다"인지 직접 읽는다.
    contents = page.get_contents()
    drawn = OVERLAY_DRAW.fullmatch(doc.xref_stream(contents[-1])) if contents else None
    if drawn is None:
        return "쪽의 마지막에 그려진 것이 도트 무늬 그림 한 장이 아니다"
    # 이름은 쪽 자신의 자원에서 직접 찾는다. 다른 그림 묶음 안에 든 같은 이름의 그림이나,
    # 글자를 그리는 묶음(Form)을 무늬 그림인 것처럼 내세우는 것을 막는다.
    image_xref = _page_image(doc, page, drawn.group("name").decode("ascii"))
    if image_xref is None:
        return "도트 무늬 자리에 그려진 것이 쪽의 그림이 아니다"
    mask_ref = doc.xref_get_key(image_xref, "SMask")
    mask_xref = int(mask_ref[1].split()[0]) if mask_ref[0] == "xref" else 0
    rect = page.rect
    rows, cols = int(-(-rect.height // cell)), int(-(-rect.width // cell))
    info = page.get_image_info()[-1]
    expected = (rect.x0, rect.y0, rect.x0 + cols * cell, rect.y0 + rows * cell)
    if any(abs(a - b) > SIZE_TOLERANCE for a, b in zip(info["bbox"], expected, strict=True)):
        return "도트 무늬 그림이 쪽 전체를 덮는 자리에 있지 않다"
    try:
        base = pymupdf.Pixmap(doc, image_xref)
        mask = pymupdf.Pixmap(doc, mask_xref) if mask_xref else None
    except Exception:
        return "도트 무늬 그림을 읽지 못했다"
    step = cell * DOTS_PX_PER_PT
    if mask is None or (mask.width, mask.height) != (cols * step, rows * step):
        return "도트 무늬 그림의 크기가 맞지 않는다"
    if mask.n != 1 or base.n != 1 or any(base.samples):
        return "도트 무늬 그림이 검은 점만으로 되어 있지 않다"
    alpha = np.frombuffer(mask.samples, dtype=np.uint8).reshape(mask.height, mask.stride)
    alpha = alpha[:, : mask.width]
    lattice = np.zeros(alpha.shape, dtype=bool)
    lattice[step // 2 :: step, step // 2 :: step] = True
    if np.any(alpha[~lattice]) or np.any((alpha != 0) & (alpha != ink)):
        return "도트 무늬에 칸 가운데가 아닌 점이나 다른 진하기의 점이 있다"
    filled = float(np.count_nonzero(alpha)) / (rows * cols)
    if not DOTS_FILL_RANGE[0] <= filled <= DOTS_FILL_RANGE[1]:
        return "도트 무늬의 점 개수가 무작위 무늬로 보기 어렵다"
    # 내용 줄기는 쪽끼리 같이 쓸 수 있다. 줄기를 비우지 않고 이 쪽의 목록에서만 뺀다.
    kept = " ".join(f"{xref} 0 R" for xref in contents[:-1])
    doc.xref_set_key(page.xref, "Contents", f"[{kept}]")
    return None


def _page_image(doc: Any, page: Any, name: str) -> int | None:
    """쪽의 자원에서 그 이름이 가리키는 것이 그림(Image)이면 그 번호를 돌려준다."""
    owner = page.xref
    for _ in range(32):
        kind, _value = doc.xref_get_key(owner, "Resources")
        if kind != "null":
            break
        parent = doc.xref_get_key(owner, "Parent")
        if parent[0] != "xref":
            return None
        owner = int(parent[1].split()[0])
    found = doc.xref_get_key(owner, f"Resources/XObject/{name}")
    if found[0] != "xref":
        return None
    xref = int(found[1].split()[0])
    if doc.xref_get_key(xref, "Subtype") != ("name", "/Image"):
        return None
    return xref


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
    source: Any, target: Any, pages: list[int], dpi: int, ocr: Ocr | None, result: CheckResult
) -> None:
    """렌더링이 원본과 조금이라도 다른 쪽만 OCR로 읽어 비교한다.

    픽셀이 모두 같은 쪽은 OCR이 읽는 글자도 같으므로 읽지 않는다. OCR이 검사 시간의 대부분이다.
    """
    if not pages:
        result.notes.append("렌더링이 원본과 픽셀 단위로 같아 OCR 비교는 생략했다.")
        return
    if ocr is None:
        result.notes.append("OCR 비교는 건너뛰었다(Tesseract 없음). 렌더링 비교로만 판정했다.")
        return
    before_scores: list[float] = []
    after_scores: list[float] = []
    for index in pages:
        before, after = source[index], target[index]
        truth = page_text(before)
        left = encode_png(render_gray(before, dpi))
        right = encode_png(render_gray(after, dpi))
        score = legibility(ocr(left), truth)
        before_scores.append(score)
        after_scores.append(score if left == right else legibility(ocr(right), truth))
    original_score, copy_score = mean(before_scores) or 0.0, mean(after_scores) or 0.0
    result.ocr = {
        "original": round(original_score, 4),
        "copy": round(copy_score, 4),
        "pages": len(pages),
    }
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
            + (f" (렌더링이 달라진 {result.ocr['pages']}쪽)" if "pages" in result.ocr else "")
        )
    lines += [f"차이: {entry}" for entry in result.differences]
    lines += [f"참고: {entry}" for entry in result.notes]
    lines.append("결과         " + ("통과" if result.passed else "실패"))
    return "\n".join(lines)
