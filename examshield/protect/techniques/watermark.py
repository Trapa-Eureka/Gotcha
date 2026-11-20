import hashlib
import hmac
import math
from typing import Any

import numpy as np
import pymupdf
from numpy.typing import NDArray

from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.protect.techniques.base import Context, Outcome

DEFAULT_CELL_PT = 8
DEFAULT_STRENGTH = 6
# 가독성 상한. 무늬가 종이 바탕을 어둡게 하는 최대 정도(0~255)와 무늬 칸의 최소 크기.
MAX_STRENGTH = 12
CELL_RANGE = (4, 24)
RENDER_SLACK = 2
CODE_LENGTH = 16


def derive_key(seed: bytes, examinee_id: str) -> bytes:
    return hmac.new(
        seed, b"examshield-watermark\x00" + examinee_id.encode("utf-8"), "sha256"
    ).digest()


def code_of(key: bytes) -> str:
    return key.hex()[:CODE_LENGTH]


def grid_shape(width_pt: float, height_pt: float, cell_pt: int) -> tuple[int, int]:
    return math.ceil(height_pt / cell_pt), math.ceil(width_pt / cell_pt)


def pattern(key: bytes, page_index: int, rows: int, cols: int) -> NDArray[np.bool_]:
    """쪽마다 다른 무늬. 키와 쪽 번호만으로 정해진다."""
    stream = hashlib.shake_256(key + page_index.to_bytes(4, "big")).digest((rows * cols + 7) // 8)
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[: rows * cols]
    return bits.reshape(rows, cols).astype(np.bool_)


def read_params(params: dict[str, Any]) -> tuple[int, int]:
    cell = int(params.get("cell_pt", DEFAULT_CELL_PT))
    strength = int(params.get("strength", DEFAULT_STRENGTH))
    if not CELL_RANGE[0] <= cell <= CELL_RANGE[1]:
        raise ExamShieldError(
            f"watermark의 cell_pt는 {CELL_RANGE[0]}~{CELL_RANGE[1]}이어야 한다: {cell}"
        )
    if not 1 <= strength <= MAX_STRENGTH:
        raise ExamShieldError(
            f"watermark의 strength는 1~{MAX_STRENGTH}이어야 한다(가독성 상한): {strength}"
        )
    return cell, strength


def apply(pdf: bytes, params: dict[str, Any], context: Context) -> Outcome:
    cell, strength = read_params(params)
    if context.watermark_key is None:
        raise ExamShieldError(
            "이 프로파일에는 수험번호별 워터마크가 있다. --copies로 수험번호 목록을, "
            "--seed-file로 시드 파일을 준다."
        )
    doc = open_pdf_bytes(pdf)
    for index, page in enumerate(doc):
        rows, cols = grid_shape(page.rect.width, page.rect.height, cell)
        bits = pattern(context.watermark_key, index, rows, cols)
        # 검은색에 아주 옅은 투명도를 준 그림을 쪽 위에 덮는다. 바탕은 조금 어두워지고 검은 글자는 그대로다.
        samples = np.zeros((rows, cols, 2), dtype=np.uint8)
        samples[..., 1] = np.where(bits, strength, 0)
        pixmap = pymupdf.Pixmap(pymupdf.csGRAY, cols, rows, samples.tobytes(), True)
        corner = page.rect.tl
        area = pymupdf.Rect(corner.x, corner.y, corner.x + cols * cell, corner.y + rows * cell)
        page.insert_image(area, pixmap=pixmap, overlay=True)
    return Outcome(
        pdf=save_deterministic(doc),
        stats={"cell_pt": cell, "strength": strength, "pages": doc.page_count},
        overlay_images=1,
        pixel_tolerance=strength + RENDER_SLACK,
    )
