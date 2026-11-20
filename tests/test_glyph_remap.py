import pymupdf
from hypothesis import given
from hypothesis import strategies as st

from examshield.check.compare import compare
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import load_profile
from examshield.protect.pipeline import protect_bytes
from examshield.protect.seed import SeedStream
from examshield.protect.techniques.glyph_remap import build_tounicode, parse_tounicode, scramble

CMAP = b"""/CIDInit /ProcSet findresource begin
begincmap
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfchar
<0001> <AC00>
<0002> <0066 0069>
endbfchar
2 beginbfrange
<0010> <0012> <0030>
<0020> <0021> [<B098> <B2E4>]
endbfrange
endcmap"""

codes = st.integers(0, 0xFFFF).map(lambda value: value.to_bytes(2, "big"))
targets = st.one_of(
    st.integers(0xAC00, 0xD7A3), st.integers(0x20, 0x7E), st.integers(0x2460, 0x2473)
).map(lambda value: value.to_bytes(2, "big"))


def test_parse_reads_bfchar_and_both_kinds_of_bfrange() -> None:
    assert parse_tounicode(CMAP) == {
        b"\x00\x01": "가".encode("utf-16-be"),
        b"\x00\x02": "fi".encode("utf-16-be"),
        b"\x00\x10": b"\x00\x30",
        b"\x00\x11": b"\x00\x31",
        b"\x00\x12": b"\x00\x32",
        b"\x00\x20": "나".encode("utf-16-be"),
        b"\x00\x21": "다".encode("utf-16-be"),
    }


@given(st.dictionaries(codes, targets, max_size=300))
def test_build_then_parse_gives_the_mapping_back(mapping: dict[bytes, bytes]) -> None:
    assert parse_tounicode(build_tounicode(mapping, CMAP)) == mapping


@given(st.dictionaries(codes, targets, min_size=1, max_size=300), st.binary(max_size=8))
def test_scramble_moves_every_character_it_touches_and_stays_in_class(
    mapping: dict[bytes, bytes], seed: bytes
) -> None:
    scrambled, changed = scramble(mapping, SeedStream(seed, "t"))
    moved = [code for code in mapping if scrambled[code] != mapping[code]]

    assert len(moved) == changed
    assert sorted(set(scrambled.values())) == sorted(set(mapping.values()))
    for code in moved:
        assert kind(mapping[code]) == kind(scrambled[code]) != "other"


def kind(target: bytes) -> str:
    char = target.decode("utf-16-be")
    if "가" <= char <= "힣":
        return "hangul"
    if "①" <= char <= "⑳":
        return "circled"
    if char.isascii() and char.isalnum():
        return "digit" if char.isdigit() else "upper" if char.isupper() else "lower"
    return "other"


def test_ligatures_and_punctuation_are_left_alone() -> None:
    mapping = {b"\x00\x01": "fi".encode("utf-16-be"), b"\x00\x02": b"\x00\x2e"}

    assert scramble(mapping, SeedStream(b"s", "t")) == (mapping, 0)


def test_extracted_text_changes_but_whitespace_and_punctuation_stay(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-glyph-remap"))
    before = page_text(open_pdf_bytes(sample_pdf)[0])
    after = page_text(open_pdf_bytes(protected.pdf)[0])

    assert len(before) == len(after)
    assert all((a == b) for a, b in zip(before, after, strict=True) if b in " .?,\n")
    assert sum(a != b for a, b in zip(before, after, strict=True)) > len(before) // 2
    assert protected.record["techniques"][0]["stats"]["fonts_scrambled"] == 1
    assert protected.record["text_layer_similarity"] < 0.5


def test_a_font_without_tounicode_is_probed_and_scrambled() -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Plain Helvetica text 123")
    original = doc.tobytes()
    protected = protect_bytes(original, load_profile("a-glyph-remap"))
    after = page_text(open_pdf_bytes(protected.pdf)[0]).strip()

    assert len(after) == len("Plain Helvetica text 123")
    assert not set(after.split()) & {"Plain", "Helvetica", "text", "123"}
    assert compare(original, protected.pdf, protected.expectation).visual_max_diff == 0


def test_a_document_with_no_text_is_reported_as_untouched() -> None:
    doc = pymupdf.open()
    doc.new_page()
    protected = protect_bytes(doc.tobytes(), load_profile("a-glyph-remap"))

    assert protected.record["techniques"][0]["stats"]["codes_changed"] == 0
    assert protected.expectation.text == "preserved"


def test_parser_pads_odd_hex_skips_comments_and_gives_up_on_usecmap() -> None:
    assert parse_tounicode(b"1 beginbfchar\n% <09> <0041>\n<1> <0042>\nendbfchar") == {
        b"\x10": b"\x00\x42"
    }
    assert parse_tounicode(b"/X usecmap 1 beginbfchar <01> <0041> endbfchar") == {}


def test_mixed_code_lengths_get_one_codespace_range_each() -> None:
    mapping = {b"\x01": b"\x00\x41", b"\x00\x02": b"\x00\x42"}
    built = build_tounicode(mapping, b"")

    assert b"2 begincodespacerange\n<00> <FF>\n<0000> <FFFF>\n" in built
    assert parse_tounicode(built) == mapping
