import numpy as np
import pytest

from examshield.benchmark import load_items, render_items_pdf
from examshield.errors import ExamShieldError
from examshield.pdfio import GrayImage, open_pdf_bytes, render_gray
from examshield.verify.capture_sim import CaptureConditions, conditions_from, simulate_capture


@pytest.fixture(scope="module")
def page() -> GrayImage:
    doc = open_pdf_bytes(render_items_pdf(load_items().items[:1]))
    return render_gray(doc[0], 150)


def test_same_seed_gives_the_same_image(page: GrayImage) -> None:
    conditions = CaptureConditions(seed=7)

    assert np.array_equal(simulate_capture(page, conditions), simulate_capture(page, conditions))


def test_different_seed_gives_a_different_image(page: GrayImage) -> None:
    first = simulate_capture(page, CaptureConditions(seed=1))
    second = simulate_capture(page, CaptureConditions(seed=2))

    assert not np.array_equal(first, second)


def test_output_has_the_capture_width_and_page_aspect(page: GrayImage) -> None:
    out = simulate_capture(page, CaptureConditions(capture_width=800))

    assert out.dtype == np.uint8
    assert out.shape[1] == 800
    assert out.shape[0] == round(page.shape[0] * 800 / page.shape[1])


def test_capture_darkens_the_page_and_keeps_the_ink(page: GrayImage) -> None:
    out = simulate_capture(page, CaptureConditions())

    assert out.mean() < page.mean()
    assert out.min() < 100


def test_all_distortions_off_only_resizes(page: GrayImage) -> None:
    neutral = CaptureConditions(
        capture_width=page.shape[1],
        skew_degrees=0.0,
        perspective=0.0,
        illumination=0.0,
        blur_sigma=0.0,
        noise_sigma=0.0,
        jpeg_quality=100,
    )
    out = simulate_capture(page, neutral)

    assert np.abs(out.astype(int) - page.astype(int)).mean() < 1.0


def test_unknown_condition_is_rejected() -> None:
    with pytest.raises(ExamShieldError):
        conditions_from({"zoom": 2})
