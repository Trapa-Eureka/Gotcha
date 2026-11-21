"""빈 바탕이 적은 쪽은 그림 한 장으로 추적하기 어렵다. protect --copies가 미리 알려 준다."""

from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.clearzone import THIN_PAGE_CELLS, blank_cells
from examshield.cli import app
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.protect.batch import Batch, thin_page_lines
from examshield.protect.copies import read_map
from examshield.trace.detect import trace
from tests.test_watermark import screenshot

runner = CliRunner()
SEED = b"0123456789abcdef-institution-seed"


def covered_document(sample_pdf: bytes, pages: tuple[int, ...]) -> bytes:
    """내장 문항 PDF에서 고른 쪽의 바탕을 옅은 색으로 덮는다. 글자는 그대로 보인다."""
    doc = open_pdf_bytes(sample_pdf)
    for index in pages:
        page = doc[index]
        page.draw_rect(page.rect, color=None, fill=(0.85, 0.9, 0.95), overlay=False)
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
