"""인쇄용 미세 도트 무늬(시나리오 B).

쪽 전체에 칸을 나누고, 칸 가운데에 작은 점을 찍거나 비워 둔다. 어느 칸에 찍는지는 키로 정한다.
--copies에서는 수험번호마다 키가 달라서 인쇄물을 찍은 사진에서 수험번호를 찾을 수 있다.
AI 판독을 막는 효과는 가설이다. verify로 잰다.
"""

import hashlib
from typing import Any

import numpy as np
import pymupdf
from numpy.typing import NDArray

from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, own_contents, save_deterministic
from examshield.protect.techniques import watermark
from examshield.protect.techniques.base import Context, Outcome

# 무늬 그림의 해상도. 점 하나는 그림의 한 픽셀이고, 한 변이 1/PX_PER_PT pt(약 0.18mm)다.
PX_PER_PT = 2
DEFAULT_CELL_PT = 6
DEFAULT_INK = 255
# 가독성 상한. 칸이 작을수록 점이 촘촘하고, ink가 클수록 점이 진하다.
CELL_RANGE = (4, 16)
INK_RANGE = (64, 255)
# 점이 덮는 넓이의 상한(쪽 넓이 대비). 칸 4pt에 모든 칸을 채워도 넘지 않는 값이다.
MAX_COVERAGE = 1 / (CELL_RANGE[0] * PX_PER_PT) ** 2
# A0(2384x3370pt)까지. 더 큰 쪽은 무늬 그림이 지나치게 커진다.
MAX_PAGE_PT = 3400


def pattern_key(context: Context) -> bytes:
    """수험번호별 키가 있으면 그것을, 없으면 문서와 프로파일에서 정해지는 시드를 쓴다."""
    return hashlib.sha256(b"examshield-dots\x00" + (context.watermark_key or context.seed)).digest()


def examinee_key(watermark_key: bytes) -> bytes:
    return hashlib.sha256(b"examshield-dots\x00" + watermark_key).digest()


def read_params(params: dict[str, Any]) -> tuple[int, int]:
    cell = int(params.get("cell_pt", DEFAULT_CELL_PT))
    ink = int(params.get("ink", DEFAULT_INK))
    if not CELL_RANGE[0] <= cell <= CELL_RANGE[1]:
        raise ExamShieldError(
            f"dots의 cell_pt는 {CELL_RANGE[0]}~{CELL_RANGE[1]}이어야 한다(가독성 상한): {cell}"
        )
    if not INK_RANGE[0] <= ink <= INK_RANGE[1]:
        raise ExamShieldError(f"dots의 ink는 {INK_RANGE[0]}~{INK_RANGE[1]}이어야 한다: {ink}")
    return cell, ink


def centers(count: int, cell: int) -> NDArray[np.intp]:
    """무늬 그림에서 칸 가운데 픽셀의 좌표."""
    return np.arange(count, dtype=np.intp) * (cell * PX_PER_PT) + (cell * PX_PER_PT) // 2


def alpha_plane(bits: NDArray[np.bool_], cell: int, ink: int) -> NDArray[np.uint8]:
    rows, cols = bits.shape
    plane = np.zeros((rows * cell * PX_PER_PT, cols * cell * PX_PER_PT), dtype=np.uint8)
    marked_rows, marked_cols = np.nonzero(bits)
    plane[centers(rows, cell)[marked_rows], centers(cols, cell)[marked_cols]] = ink
    return plane


def apply(pdf: bytes, params: dict[str, Any], context: Context) -> Outcome:
    cell, ink = read_params(params)
    key = pattern_key(context)
    doc = open_pdf_bytes(pdf)
    marked = 0
    total = 0
    for index, page in enumerate(doc):
        if page.rotation:
            raise ExamShieldError(
                f"{index + 1}쪽이 회전돼 있다(/Rotate {page.rotation}). dots는 회전된 쪽을 다루지 "
                "않는다. 회전 없이 PDF로 다시 내보낸 뒤 쓴다."
            )
        if page.rect.width > MAX_PAGE_PT or page.rect.height > MAX_PAGE_PT:
            raise ExamShieldError(
                f"{index + 1}쪽이 너무 크다. dots는 한 변이 {MAX_PAGE_PT}pt 이하인 쪽만 다룬다."
            )
        rows, cols = watermark.grid_shape(page.rect.width, page.rect.height, cell)
        bits = watermark.pattern(key, index, rows, cols)
        alpha = alpha_plane(bits, cell, ink)
        samples = np.zeros((*alpha.shape, 2), dtype=np.uint8)
        samples[..., 1] = alpha
        pixmap = pymupdf.Pixmap(
            pymupdf.csGRAY, alpha.shape[1], alpha.shape[0], samples.tobytes(), True
        )
        corner = page.rect.tl
        area = pymupdf.Rect(corner.x, corner.y, corner.x + cols * cell, corner.y + rows * cell)
        own_contents(doc, page)
        page.insert_image(area, pixmap=pixmap, overlay=True)
        marked += int(bits.sum())
        total += alpha.size
    return Outcome(
        pdf=save_deterministic(doc),
        stats={
            "cell_pt": cell,
            "ink": ink,
            "dot_pt": round(1 / PX_PER_PT, 3),
            "coverage": round(marked / total, 6) if total else 0.0,
            "pages": doc.page_count,
        },
        overlay_images=1,
        dots={"cell_pt": cell, "ink": ink},
    )
