"""실제 문서에서만 나오던 구조를 작게 만들어 놓고 본다. 내려받은 문서는 저장소에 두지 않는다."""

import pymupdf
import pytest

from examshield.check.compare import compare
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.profiles import load_profile
from examshield.protect.pipeline import protect_bytes
from examshield.protect.techniques.glyph_remap import font_xrefs

KEY = b"k" * 32
OVERLAYS = ("a-watermark", "b-dots", "a-full")


def page_with_text(doc: pymupdf.Document, width: float = 595, height: float = 842) -> pymupdf.Page:
    page = doc.new_page(width=width, height=height)
    page.insert_font(fontname="kr", fontbuffer=pymupdf.Font("korea").buffer)
    for line in range(12):
        page.insert_text(
            (60, 90 + line * 40),
            f"{line + 1}. 다음 중 옳은 것은? ① 가 ② 나 ③ 다 ④ 라",
            fontname="kr",
            fontsize=11,
        )
    return page


def checked(pdf: bytes, name: str) -> list[str]:
    made = protect_bytes(pdf, load_profile(name), None, KEY)
    return compare(pdf, made.pdf, made.expectation).differences


def test_layers_with_the_same_name_do_not_fail_the_check() -> None:
    """이름과 설정이 같은 레이어는 저장할 때 하나로 합쳐진다. 내용은 그대로다."""
    doc = pymupdf.open()
    page_with_text(doc)
    for _ in range(4):
        doc.add_ocg("Watermark", on=True)
    doc.add_ocg("Headers/Footers", on=True)
    pdf = bytes(doc.tobytes())
    assert len(open_pdf_bytes(pdf).get_ocgs()) == 5

    for name in ("a-remap-decoy", "a-watermark"):
        assert checked(pdf, name) == []


def test_a_layer_added_to_the_copy_still_fails_the_check() -> None:
    doc = pymupdf.open()
    page_with_text(doc)
    doc.add_ocg("Watermark", on=True)
    pdf = bytes(doc.tobytes())
    made = protect_bytes(pdf, load_profile("a-remap-decoy"))
    forged = open_pdf_bytes(made.pdf)
    forged.add_ocg("숨은 내용", on=False)

    result = compare(pdf, bytes(forged.tobytes()), made.expectation)

    assert any("레이어" in entry for entry in result.differences)


def test_a_rasterized_copy_must_have_no_layers() -> None:
    doc = pymupdf.open()
    page_with_text(doc)
    doc.add_ocg("Watermark", on=True)
    pdf = bytes(doc.tobytes())
    made = protect_bytes(pdf, load_profile("a-rasterize"))
    assert compare(pdf, made.pdf, made.expectation).passed

    forged = open_pdf_bytes(made.pdf)
    forged.add_ocg("숨은 내용", on=False)
    result = compare(pdf, bytes(forged.tobytes()), made.expectation)

    assert any("레이어" in entry for entry in result.differences)


@pytest.mark.parametrize("name", OVERLAYS)
def test_thin_lines_render_the_same_once_the_pattern_is_lifted(name: str) -> None:
    """반투명 그림이 쪽의 자원에 남아 있으면 가는 선의 가장자리가 몇 화소 달라졌다."""
    doc = pymupdf.open()
    page = page_with_text(doc)
    for index in range(40):
        x = 300.13 + index * 6.37
        page.draw_line((x, 500), (x + 0.4, 800), color=(0.35, 0.35, 0.35), width=0.31)
    page.draw_rect(pymupdf.Rect(60.3, 560.2, 250.7, 700.9), color=(0.6, 0.6, 0.6), width=0.25)
    pdf = save_deterministic(doc)

    made = protect_bytes(pdf, load_profile(name), None, KEY)
    result = compare(pdf, made.pdf, made.expectation)

    assert result.differences == []
    assert result.visual_max_diff == 0


def spread_as_three_pages() -> bytes:
    """펼침면 하나를 쪽 셋으로 나눈 문서. 쪽들이 내용 줄기 목록 하나를 같이 쓴다."""
    doc = pymupdf.open()
    wide = doc.new_page(width=1080, height=560)
    wide.insert_font(fontname="kr", fontbuffer=pymupdf.Font("korea").buffer)
    for column in range(3):
        for line in range(8):
            wide.insert_text(
                (30 + column * 360, 60 + line * 50),
                f"{column + 1}면 {line + 1}번 문항 ① 가 ② 나 ③ 다",
                fontname="kr",
                fontsize=11,
            )
    wide.clean_contents()
    stream = wide.get_contents()[0]
    doc.fullcopy_page(0)
    doc.fullcopy_page(0)
    shared = doc.get_new_xref()
    doc.update_object(shared, f"[{stream} 0 R]")
    for index, page in enumerate(doc):
        doc.xref_set_key(page.xref, "Contents", f"{shared} 0 R")
        box = f"[{index * 360} 0 {index * 360 + 360} 560]"
        doc.xref_set_key(page.xref, "MediaBox", box)
        doc.xref_set_key(page.xref, "CropBox", box)
    return bytes(doc.tobytes())


@pytest.mark.parametrize("name", [*OVERLAYS, "a-remap-decoy"])
def test_pages_that_share_one_list_of_content_streams_get_one_pattern_each(name: str) -> None:
    pdf = spread_as_three_pages()
    before = [len(page.get_image_info()) for page in open_pdf_bytes(pdf)]

    made = protect_bytes(pdf, load_profile(name), None, KEY)
    result = compare(pdf, made.pdf, made.expectation)
    after = [len(page.get_image_info()) for page in open_pdf_bytes(made.pdf)]

    assert result.differences == []
    extra = 0 if name == "a-remap-decoy" else 1
    assert after == [count + extra for count in before]


class MissingObjects:
    """객체 목록에 번호는 있는데 읽으면 오류가 나는 객체가 섞인 문서."""

    def __init__(self, doc: pymupdf.Document, missing: set[int]) -> None:
        self.doc, self.missing = doc, missing

    def xref_length(self) -> int:
        return int(self.doc.xref_length()) + 2

    def xref_get_key(self, xref: int, key: str) -> tuple[str, str]:
        if xref in self.missing or xref >= self.doc.xref_length():
            raise RuntimeError(f"cannot find object in xref ({xref} 0 R)")
        found: tuple[str, str] = self.doc.xref_get_key(xref, key)
        return found


def test_objects_listed_but_missing_do_not_stop_the_search_for_fonts() -> None:
    doc = pymupdf.open()
    page_with_text(doc)
    doc = open_pdf_bytes(save_deterministic(doc))
    fonts = font_xrefs(doc)
    others = {xref for xref in range(1, doc.xref_length()) if xref not in fonts}

    assert fonts
    assert font_xrefs(MissingObjects(doc, others)) == fonts


@pytest.mark.parametrize("size", [(595, 842), (842, 1191), (556, 754), (351.3, 567.1), (612, 792)])
def test_a_rasterized_page_of_any_size_renders_like_the_original(size: tuple[float, float]) -> None:
    """쪽 크기가 화소로 딱 떨어지지 않으면 그림이 조금 줄어 글자 가장자리가 달라졌다."""
    doc = pymupdf.open()
    page_with_text(doc, *size)
    pdf = save_deterministic(doc)

    made = protect_bytes(pdf, load_profile("a-rasterize"))
    result = compare(pdf, made.pdf, made.expectation)

    assert result.differences == []
    assert result.visual_max_diff == 0
    copy = open_pdf_bytes(made.pdf)[0]
    assert abs(copy.rect.width - size[0]) < 0.01 and abs(copy.rect.height - size[1]) < 0.01
