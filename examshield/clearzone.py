"""저시력 수험생용 복사본. 글자와 그림 둘레를 비우고 빈 바탕에만 무늬를 넣는다.

무늬 칸 가운데 내용(글자, 선, 그림)이 있는 칸과 그 둘레의 칸을 '비울 칸'으로 삼는다. 화면을
확대하거나 대비를 높여 읽는 사람에게 글자 옆의 무늬가 잡음이 되지 않게 하려는 것이다.
어느 칸을 비울지는 무늬를 넣기 전의 쪽만 보고 정한다. 그래서 check와 trace가 원본에서 똑같이 구한다.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from examshield.errors import ExamShieldError
from examshield.pdfio import render_gray

MASK_DPI = 72
# 빈 바탕으로 보는 칸: 칸 안의 밝기가 고르고(가장 밝은 곳과 어두운 곳의 차이가 이 값 이하) 너무 어둡지 않다.
# 흰 종이뿐 아니라 옅은 바탕색이 고르게 깔린 곳도 빈 바탕이다. 글자, 선, 그림, 사진은 고르지 않다.
UNIFORM_RANGE = 5
DARKEST_BACKGROUND = 128
DEFAULT_CLEAR_PT = 18
CLEAR_RANGE = (6, 72)


def read_clear(params: dict[str, Any]) -> int:
    clear = int(params.get("clear_pt", DEFAULT_CLEAR_PT))
    if not CLEAR_RANGE[0] <= clear <= CLEAR_RANGE[1]:
        raise ExamShieldError(f"clear_pt는 {CLEAR_RANGE[0]}~{CLEAR_RANGE[1]}이어야 한다: {clear}")
    return clear


def reach(clear_pt: int, cell_pt: int) -> int:
    """내용이 있는 칸에서 몇 칸까지 비우는지."""
    return math.ceil(clear_pt / cell_pt)


def busy_cells(page: Any, cell_pt: int, rows: int, cols: int, slack: int = 0) -> NDArray[np.bool_]:
    """내용이 조금이라도 있는 칸. 쪽을 1pt에 화소 하나로 그려서 본다.

    slack만큼 느슨하게 본다(밝기 차이를 그만큼 더 봐주고, 그만큼 더 어두운 바탕도 받는다).
    """
    gray = render_gray(page, MASK_DPI)
    height = min(gray.shape[0], rows * cell_pt)
    width = min(gray.shape[1], cols * cell_pt)
    # 쪽 밖으로 나가는 가장자리 칸은 쪽 가장자리의 화소로 채운다. 흰색으로 채우면 색 바탕 쪽의
    # 가장자리 칸이 모두 내용이 있는 칸으로 잡힌다.
    canvas = np.pad(
        gray[:height, :width],
        ((0, rows * cell_pt - height), (0, cols * cell_pt - width)),
        mode="edge",
    )
    blocks = canvas.reshape(rows, cell_pt, cols, cell_pt)
    darkest = blocks.min(axis=(1, 3)).astype(np.int16)
    brightest = blocks.max(axis=(1, 3)).astype(np.int16)
    uneven = brightest - darkest > UNIFORM_RANGE + slack
    return np.asarray(uneven | (darkest < DARKEST_BACKGROUND - slack), dtype=np.bool_)


def open_cells(
    page: Any,
    cell_pt: int,
    clear_pt: int,
    rows: int,
    cols: int,
    slack: int = 0,
) -> NDArray[np.bool_]:
    """무늬를 넣어도 되는 칸. 내용이 있는 칸에서 clear_pt 안쪽의 칸은 뺀다."""
    return ~_spread(busy_cells(page, cell_pt, rows, cols, slack), reach(clear_pt, cell_pt))


def _spread(busy: NDArray[np.bool_], cells: int) -> NDArray[np.bool_]:
    spread = busy.copy()
    for axis in (0, 1):
        source = spread.copy()
        for shift in range(1, cells + 1):
            ahead = [slice(None), slice(None)]
            behind = [slice(None), slice(None)]
            ahead[axis], behind[axis] = slice(shift, None), slice(None, -shift)
            spread[tuple(ahead)] |= source[tuple(behind)]
            spread[tuple(behind)] |= source[tuple(ahead)]
    return spread


def stats(clear_pt: int, opened: int, total: int) -> dict[str, Any]:
    """작업 기록에 적는 값. open_share는 무늬를 넣을 수 있었던 칸의 비율이다."""
    return {
        "clear_pt": clear_pt,
        "open_cells": opened,
        "open_share": round(opened / total, 4) if total else 0.0,
    }


# trace가 수험번호를 가리는 데 쓰는 빈 바탕 칸의 최소 수. trace/detect.py의 MIN_CELLS와 같은 값이다.
TRACE_MIN_CELLS = 300
# 쪽의 절반쯤만 나온 그림으로도 그 수를 채우려면 쪽에 이만큼은 있어야 한다.
THIN_PAGE_CELLS = 2 * TRACE_MIN_CELLS


def blank_cells(doc: Any, cell_pt: int) -> list[int]:
    """쪽마다 내용이 없는 무늬 칸의 수. trace는 이런 칸에서만 무늬를 읽는다."""
    counts = []
    for page in doc:
        rows = math.ceil(page.rect.height / cell_pt)
        cols = math.ceil(page.rect.width / cell_pt)
        counts.append(int((~busy_cells(page, cell_pt, rows, cols)).sum()))
    return counts
