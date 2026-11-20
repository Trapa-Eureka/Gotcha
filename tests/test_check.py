import json
from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.check.compare import Expectation, compare, format_result
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.profiles import load_profile
from examshield.protect.pipeline import load_expectation, protect_bytes
from tests.conftest import A_PROFILES, make_pdf

BASE = "1. 물의 끓는점은 100도이다. 값은 3,141이고 x = 17이다."


def test_identical_documents_pass(sample_pdf: bytes) -> None:
    result = compare(sample_pdf, sample_pdf)

    assert result.passed
    assert result.text == "identical"
    assert result.visual_max_diff == 0


@pytest.mark.parametrize(
    "changed",
    [
        BASE.replace("3,141", "3.141"),
        BASE.replace("17", "11"),
        BASE.replace("100도", "100℃"),
        BASE + " 추가",
    ],
)
@pytest.mark.parametrize("fontsize", [7, 10])
def test_a_small_content_change_fails(changed: str, fontsize: float) -> None:
    result = compare(make_pdf(BASE, fontsize), make_pdf(changed, fontsize))

    assert not result.passed
    assert result.text == "different"
    assert result.visual_max_diff is not None and result.visual_max_diff > 2


def test_a_missing_page_fails(sample_pdf: bytes) -> None:
    doc = open_pdf_bytes(sample_pdf)
    doc.delete_page(0)
    result = compare(sample_pdf, save_deterministic(doc))

    assert not result.passed
    assert "쪽수" in result.differences[0]


def test_a_different_page_size_fails() -> None:
    def blank(width: float) -> bytes:
        doc = pymupdf.open()
        doc.new_page(width=width, height=842)
        return save_deterministic(doc)

    assert not compare(blank(595), blank(600)).passed


def test_an_added_picture_fails(sample_pdf: bytes) -> None:
    doc = open_pdf_bytes(sample_pdf)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pixmap.clear_with(0)
    doc[0].insert_image(pymupdf.Rect(300, 300, 320, 320), pixmap=pixmap)
    result = compare(sample_pdf, save_deterministic(doc))

    assert any("그림 개수" in entry for entry in result.differences)


def test_a_visible_change_under_an_unchanged_text_layer_fails(sample_pdf: bytes) -> None:
    doc = open_pdf_bytes(sample_pdf)
    doc[1].draw_rect(pymupdf.Rect(80, 95, 200, 110), color=(1, 1, 1), fill=(1, 1, 1))
    result = compare(sample_pdf, save_deterministic(doc))

    assert not result.passed
    assert any("2쪽: 렌더링" in entry for entry in result.differences)


def test_a_colour_change_with_the_same_brightness_fails() -> None:
    def tinted(color: tuple[float, float, float]) -> bytes:
        doc = pymupdf.open()
        doc.new_page().draw_rect(pymupdf.Rect(100, 100, 200, 200), color=color, fill=color)
        return save_deterministic(doc)

    assert not compare(tinted((1, 0, 0)), tinted((0, 0.3, 0))).passed


@pytest.mark.parametrize("profile", A_PROFILES)
def test_every_profile_output_passes_with_its_job_record(profile: str, sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile(profile))
    result = compare(sample_pdf, protected.pdf, protected.expectation)

    assert result.passed, result.differences
    assert result.visual_max_diff == 0


@pytest.mark.parametrize("profile", A_PROFILES)
def test_a_scrambled_copy_fails_without_its_job_record(profile: str, sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile(profile))

    assert not compare(sample_pdf, protected.pdf).passed


def test_a_tampered_decoy_copy_fails_even_with_the_job_record(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-decoy"))
    doc = open_pdf_bytes(protected.pdf)
    doc[0].insert_text((100, 400), "hidden", render_mode=3)
    result = compare(sample_pdf, save_deterministic(doc), protected.expectation)

    assert any("1쪽: 추출한 텍스트" in entry for entry in result.differences)


def test_a_tampered_rasterized_copy_fails(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-rasterize"))
    doc = open_pdf_bytes(protected.pdf)
    doc[0].insert_text((100, 400), "3", fontsize=8)
    result = compare(sample_pdf, save_deterministic(doc), protected.expectation)

    assert not result.passed


def test_ocr_is_skipped_when_every_page_renders_identically(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-remap-decoy"))
    calls: list[int] = []

    def fake_ocr(image: bytes) -> str:
        calls.append(len(image))
        return ""

    result = compare(sample_pdf, protected.pdf, protected.expectation, fake_ocr)

    assert result.passed and result.text == "altered_as_expected"
    assert result.visual_max_diff == 0
    assert calls == [] and result.ocr is None
    assert any("OCR 비교는 생략" in note for note in result.notes)
    assert not any("Tesseract 없음" in note for note in result.notes)


def test_only_the_pages_that_render_differently_are_read_by_ocr(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-glyph-remap"))
    doc = open_pdf_bytes(protected.pdf)
    # 둘째 쪽에만 허용치 안의 옅은 점을 찍는다.
    doc[1].draw_rect(pymupdf.Rect(300, 700, 302, 702), color=None, fill=(0.995, 0.995, 0.995))
    truth = open_pdf_bytes(sample_pdf)[1].get_text()
    replies = iter([truth, ""])

    result = compare(
        sample_pdf, save_deterministic(doc), protected.expectation, lambda _: next(replies)
    )

    assert result.visual_max_diff is not None and 0 < result.visual_max_diff <= 2
    assert result.ocr == {"original": 1.0, "copy": 0.0, "pages": 1}
    assert any("OCR 판독률이 원본보다 낮다" in entry for entry in result.differences)
    assert "렌더링이 달라진 1쪽" in format_result(result)


def test_ocr_is_used_when_the_text_layer_is_scrambled(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-full"), None, b"k" * 16)
    calls: list[int] = []

    def fake_ocr(image: bytes) -> str:
        calls.append(len(image))
        return "읽은 글자"

    result = compare(sample_pdf, protected.pdf, protected.expectation, fake_ocr)

    assert result.text == "altered_as_expected"
    assert result.ocr is not None and result.ocr["original"] == result.ocr["copy"]
    assert result.ocr["pages"] == 3 and len(calls) == 6


def test_a_copy_that_ocr_reads_worse_fails(sample_pdf: bytes) -> None:
    blank = pymupdf.open()
    for page in open_pdf_bytes(sample_pdf):
        blank.new_page(width=page.rect.width, height=page.rect.height)
    truth = [page.get_text() for page in open_pdf_bytes(sample_pdf)]
    replies = iter([truth[0], "", truth[1], "", truth[2], ""])
    result = compare(
        sample_pdf, save_deterministic(blank), Expectation(text="altered"), lambda _: next(replies)
    )

    assert any("OCR" in entry for entry in result.differences)


def test_missing_ocr_is_reported_not_hidden(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-full"), None, b"k" * 16)
    result = compare(sample_pdf, protected.pdf, protected.expectation, None)

    assert result.passed
    assert any("건너뛰었다" in note for note in result.notes)


def test_job_record_of_another_copy_is_refused(sample_pdf: bytes, tmp_path: Path) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-decoy"))
    job = tmp_path / "x.job.json"
    job.write_text(json.dumps(protected.record), encoding="utf-8")

    assert load_expectation(job, protected.pdf).decoys
    with pytest.raises(ExamShieldError, match="해시"):
        load_expectation(job, sample_pdf)


def test_cli_check_exit_codes(sample_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    other = tmp_path / "other.pdf"
    other.write_bytes(make_pdf(BASE))

    assert runner.invoke(app, ["check", str(sample_file), str(sample_file)]).exit_code == 0
    assert runner.invoke(app, ["check", str(sample_file), str(other)]).exit_code == 1
    assert runner.invoke(app, ["check", str(sample_file), str(tmp_path / "no.pdf")]).exit_code == 2


def test_a_job_record_claiming_a_low_resolution_fails(sample_pdf: bytes) -> None:
    result = compare(sample_pdf, sample_pdf, Expectation(text="altered", flattened_dpi=72))

    assert any("해상도" in entry for entry in result.differences)


def test_cli_check_on_a_file_that_is_not_a_pdf_exits_with_code_2(
    sample_file: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_text("not a pdf")

    assert CliRunner().invoke(app, ["check", str(sample_file), str(broken)]).exit_code == 2
