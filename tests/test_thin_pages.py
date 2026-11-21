"""빈 바탕이 적은 쪽은 그림 한 장으로 추적하기 어렵다. protect --copies가 미리 알려 준다."""

from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.clearzone import THIN_PAGE_CELLS, blank_cells
from examshield.cli import app
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.protect.batch import Batch, thin_page_lines
from examshield.protect.copies import read_map
from examshield.trace.detect import trace
from examshield.verify.capture_sim import encode_png
from tests.test_watermark import screenshot

runner = CliRunner()
SEED = b"0123456789abcdef-institution-seed"


def covered_document(sample_pdf: bytes, pages: tuple[int, ...]) -> bytes:
    """내장 문항 PDF에서 고른 쪽의 바탕을 사진처럼 고르지 않은 그림으로 덮는다. 글자는 그대로 보인다."""
    doc = open_pdf_bytes(sample_pdf)
    rng = np.random.default_rng(7)
    for index in pages:
        page = doc[index]
        coarse = rng.integers(150, 256, size=(140, 100), dtype=np.uint8)
        picture = cv2.resize(coarse, (600, 840), interpolation=cv2.INTER_CUBIC)
        page.insert_image(page.rect, stream=encode_png(picture), overlay=False)
    return save_deterministic(doc)


def tinted_document(sample_pdf: bytes, fill: tuple[float, float, float]) -> bytes:
    """모든 쪽의 바탕에 색을 고르게 깐다."""
    doc = open_pdf_bytes(sample_pdf)
    for page in doc:
        page.draw_rect(page.rect, color=None, fill=fill, overlay=False)
    return save_deterministic(doc)


def make(tmp_path: Path, pdf: bytes, profile: str = "a-watermark") -> tuple[int, str, Path]:
    source = tmp_path / "in" / "exam.pdf"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(pdf)
    secrets = tmp_path / "secrets"
    secrets.mkdir(exist_ok=True)
    (secrets / "exam.seed").write_bytes(SEED)
    (tmp_path / "ids.csv").write_text("A001\nA002\nA003\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "protect",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--profile",
            profile,
            "--copies",
            str(tmp_path / "ids.csv"),
            "--seed-file",
            str(secrets / "exam.seed"),
            "--map-file",
            str(secrets / "map.csv"),
        ],  # fmt: skip
    )
    return result.exit_code, result.output, tmp_path


def test_ordinary_pages_have_plenty_of_blank_cells_and_no_notice(
    tmp_path: Path, sample_pdf: bytes
) -> None:
    counts = blank_cells(open_pdf_bytes(sample_pdf), 8)

    code, output, _ = make(tmp_path, sample_pdf)

    assert min(counts) > 5 * THIN_PAGE_CELLS
    assert code == 0 and "빈 바탕이 적은" not in output


@pytest.mark.parametrize("profile", ["a-watermark", "b-dots"])
def test_covered_pages_are_named_and_really_are_not_traced_from_a_capture(
    profile: str, tmp_path: Path, sample_pdf: bytes
) -> None:
    pdf = covered_document(sample_pdf, (1,))

    code, output, base = make(tmp_path, pdf, profile)
    rows = read_map(base / "secrets" / "map.csv")
    copy = (base / "out" / "exam-A002.pdf").read_bytes()
    covered = trace(screenshot(copy, 110, ".png", page=1), ".png", pdf, rows, SEED)
    doc = pymupdf.open(stream=copy, filetype="pdf")
    doc.set_metadata({"title": "leaked"})
    whole = trace(bytes(doc.tobytes()), ".pdf", pdf, rows, SEED)

    assert code == 0, output
    assert "빈 바탕이 적은 쪽이 3쪽 가운데 1쪽 있다(2쪽)" in output
    assert "다른 쪽이 함께 있으면 찾는다" in output
    # 알림이 말한 그대로다. 덮인 쪽 한 장으로는 못 찾고, PDF 전체로는 찾는다.
    assert not covered.found
    assert whole.found and whole.examinee_id == "A002"


def test_a_document_covered_on_every_page_is_told_that_only_the_hash_works(
    tmp_path: Path, sample_pdf: bytes
) -> None:
    code, output, _ = make(tmp_path, covered_document(sample_pdf, (0, 1, 2)))

    assert code == 0, output
    assert "3쪽 가운데 3쪽" in output and "해시로만" in output


def test_a_long_list_of_thin_pages_is_cut_short() -> None:
    batch = Batch([], Path("x"), 0, 0, thin_pages=tuple(range(1, 31)), pages=40, blank_cells=9000)

    lines = thin_page_lines(batch)

    assert "40쪽 가운데 30쪽" in lines[0] and "10쪽 외" in lines[0] and "11" not in lines[0]


@pytest.mark.parametrize("profile", ["a-watermark", "b-dots"])
@pytest.mark.parametrize("fill", [(0.85, 0.9, 0.95), (1.0, 0.92, 0.8), (0.6, 0.6, 0.6)])
def test_an_evenly_tinted_page_counts_as_blank_and_is_traced_from_a_capture(
    profile: str, fill: tuple[float, float, float], tmp_path: Path, sample_pdf: bytes
) -> None:
    """바탕색이 고르게 깔린 쪽은 흰 종이와 같이 무늬를 읽을 수 있다."""
    pdf = tinted_document(sample_pdf, fill)

    code, output, base = make(tmp_path, pdf, profile)
    rows = read_map(base / "secrets" / "map.csv")
    copy = (base / "out" / "exam-A002.pdf").read_bytes()
    png = trace(screenshot(copy, 110, ".png", page=1), ".png", pdf, rows, SEED)
    jpeg = trace(screenshot(copy, 110, ".jpg", page=1), ".jpg", pdf, rows, SEED)
    plain = trace(screenshot(pdf, 110, ".png", page=1), ".png", pdf, rows, SEED)

    assert code == 0 and "빈 바탕이 적은" not in output
    assert png.found and png.examinee_id == "A002"
    assert jpeg.found and jpeg.examinee_id == "A002"
    # 무늬가 없는 원본의 캡처는 누구도 가리키지 않는다.
    assert not plain.found


def test_a_capture_whose_colours_came_out_differently_is_still_traced(
    tmp_path: Path, sample_pdf: bytes
) -> None:
    """화면과 저장 방식에 따라 캡처의 색은 원본과 다르게 나온다. 바탕색마다 다르게 어긋난다."""
    doc = open_pdf_bytes(sample_pdf)
    for page in doc:
        upper = pymupdf.Rect(0, 0, page.rect.width, page.rect.height / 2)
        lower = pymupdf.Rect(0, page.rect.height / 2, page.rect.width, page.rect.height)
        page.draw_rect(upper, color=None, fill=(0.85, 0.9, 0.95), overlay=False)
        page.draw_rect(lower, color=None, fill=(1.0, 0.92, 0.8), overlay=False)
    pdf = save_deterministic(doc)
    _, _, base = make(tmp_path, pdf)
    rows = read_map(base / "secrets" / "map.csv")
    copy = (base / "out" / "exam-A003.pdf").read_bytes()
    pix = open_pdf_bytes(copy)[0].get_pixmap(dpi=110)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    shifted = np.clip((rgb / 255.0) ** np.array([1.06, 0.97, 1.03]) * 255, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))
    assert ok

    result = trace(bytes(encoded.tobytes()), ".png", pdf, rows, SEED)

    assert result.found and result.examinee_id == "A003"


def test_a_dark_background_is_not_read(tmp_path: Path, sample_pdf: bytes) -> None:
    """어두운 바탕에서는 무늬가 거의 보이지 않는다. 읽지 않고, 만들 때 알려 준다."""
    pdf = tinted_document(sample_pdf, (0.2, 0.25, 0.45))

    code, output, base = make(tmp_path, pdf)
    rows = read_map(base / "secrets" / "map.csv")
    copy = (base / "out" / "exam-A002.pdf").read_bytes()
    result = trace(screenshot(copy, 110, ".png", page=1), ".png", pdf, rows, SEED)

    assert code == 0 and "3쪽 가운데 3쪽" in output
    assert not result.found
