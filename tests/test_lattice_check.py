import numpy as np
import pytest
from numpy.typing import NDArray

from examshield.check.compare import WATERMARK_SLACK, Expectation, _looks_like_watermark


def cell_by_cell(
    before: NDArray[np.uint8], after: NDArray[np.uint8], expect: Expectation, dpi: int
) -> bool:
    """배열 연산으로 바꾸기 전의 검사. 칸마다 하나씩 본다. 새 검사가 같은 답을 내는지 견줄 기준이다."""
    cell = expect.overlay_cell_pt
    assert cell is not None
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


def page(rng: np.random.Generator, height: int, width: int) -> NDArray[np.uint8]:
    """흰 바탕에 글자처럼 어두운 얼룩이 흩어진 쪽."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for _ in range(int(rng.integers(0, 30))):
        top, left = int(rng.integers(0, height - 8)), int(rng.integers(0, width - 30))
        image[top : top + 8, left : left + 30] = int(rng.integers(0, 255))
    return image


def lattice(
    rng: np.random.Generator, image: NDArray[np.uint8], step: float, strength: int
) -> NDArray[np.uint8]:
    after = image.astype(np.int16)
    height, width = image.shape[:2]
    for row in range(int(height / step) + 1):
        for col in range(int(width / step) + 1):
            if rng.random() < 0.5:
                top, left = round(row * step), round(col * step)
                after[top : round((row + 1) * step), left : round((col + 1) * step)] -= strength
    return np.clip(after, 0, 255).astype(np.uint8)


CASES = [
    (seed, dpi, cell, size)
    for seed in range(6)
    for dpi, cell in ((72, 8), (100, 8), (150, 6), (150, 13), (96, 5))
    for size in ((260, 190), (333, 257))
]


@pytest.mark.parametrize(("seed", "dpi", "cell", "size"), CASES)
def test_the_array_check_agrees_with_checking_cell_by_cell(
    seed: int, dpi: int, cell: int, size: tuple[int, int]
) -> None:
    rng = np.random.default_rng(seed * 1000 + dpi + cell)
    strength = int(rng.integers(3, 9))
    expect = Expectation(
        overlay_images=1, pixel_tolerance=strength + WATERMARK_SLACK, overlay_cell_pt=cell
    )
    before = page(rng, *size)
    clean = lattice(rng, before, cell * dpi / 72.0, strength)
    variants = {"clean": clean}
    # 옅은 글자를 덧씌운 것, 세기가 다른 칸, 한 화소만 다른 것, 아무것도 안 바뀐 것.
    stroke = clean.copy()
    top, left = int(rng.integers(0, size[0] - 3)), int(rng.integers(0, size[1] - 40))
    stroke[top : top + 3, left : left + 40] = np.clip(
        stroke[top : top + 3, left : left + 40].astype(np.int16) - 3, 0, 255
    ).astype(np.uint8)
    variants["stroke"] = stroke
    variants["wrong strength"] = lattice(rng, before, cell * dpi / 72.0, strength + 4)
    pixel = clean.copy()
    y, x = int(rng.integers(0, size[0])), int(rng.integers(0, size[1]))
    pixel[y, x] = np.clip(pixel[y, x].astype(np.int16) - 2, 0, 255).astype(np.uint8)
    variants["one pixel"] = pixel
    variants["untouched"] = before.copy()
    variants["one channel"] = clean.copy()
    variants["one channel"][:, :, 1] = np.clip(
        clean[:, :, 1].astype(np.int16) - int(rng.integers(0, 4)), 0, 255
    ).astype(np.uint8)

    for name, after in variants.items():
        expected = cell_by_cell(before, after, expect, dpi)
        assert _looks_like_watermark(before, after, expect, dpi) == expected, name


def test_the_cases_cover_both_answers() -> None:
    rng = np.random.default_rng(7)
    expect = Expectation(overlay_images=1, pixel_tolerance=6 + WATERMARK_SLACK, overlay_cell_pt=8)
    before = page(rng, 260, 190)
    clean = lattice(rng, before, 8 * 100 / 72.0, 6)
    spoiled = clean.copy()
    spoiled[104:108, 20:120] -= 3

    assert _looks_like_watermark(before, clean, expect, 100)
    assert not _looks_like_watermark(before, spoiled, expect, 100)
    assert not _looks_like_watermark(before, clean, Expectation(overlay_images=1), 100)
