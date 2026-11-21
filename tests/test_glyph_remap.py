from pathlib import Path

import pymupdf
import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from examshield.check.compare import compare
from examshield.cli import app
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


def two_font_pdf(first: str, second: str) -> bytes:
    import pymupdf  # noqa: PLC0415

    from examshield.pdfio import save_deterministic  # noqa: PLC0415

    doc = pymupdf.open()
    page = doc.new_page()
    buffer = pymupdf.Font("korea").buffer
    page.insert_font(fontname="one", fontbuffer=buffer)
    page.insert_text((60, 100), first, fontname="one", fontsize=11)
    other = doc.new_page()
    if second.isascii():
        other.insert_text((60, 100), second, fontname="helv", fontsize=11)
    else:
        other.insert_font(fontname="two", fontbuffer=buffer)
        other.insert_text((60, 100), second, fontname="two", fontsize=11)
    return save_deterministic(doc)


def test_hanja_jamo_and_circled_hangul_are_scrambled_too() -> None:
    """실제 시험지에서 그대로 읽히던 글자들이다."""
    from examshield.pdfio import open_pdf_bytes, page_text  # noqa: PLC0415

    text = "韓國語 文字 大學 ㄱㄴㄷㄹ ㉠㉡㉢㉣ 가나다라"
    pdf = two_font_pdf(text, text)

    made = protect_bytes(pdf, load_profile("a-glyph-remap"))
    after = page_text(open_pdf_bytes(made.pdf)[0])

    kept = [left for left, right in zip(text, after, strict=False) if left == right and left != " "]
    assert kept == []
    # 종류 안에서만 바꾼다. 한자는 한자로, 자모는 자모로.
    assert all("一" <= char <= "鿿" for char in after[:3])
    assert made.record["text_layer_letter_similarity"] == 0.0


def test_similarity_over_letters_ignores_spaces_punctuation_and_the_decoy() -> None:
    from examshield.protect.pipeline import letters_kept, letters_only  # noqa: PLC0415

    assert letters_only("1. 다음 중, 옳은 것은? (3점)") == "1다음중옳은것은3점"
    assert letters_kept("가나 다라.", "가마 다바.") == 0.5
    assert letters_kept("가나 다라.", "미끼 문구 가나 다라.", "미끼 문구") == 1.0
    assert letters_kept("가나다라", "가나") < 1.0


def test_cli_shows_the_share_of_letters_still_readable(sample_file: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["protect", str(sample_file), "--out", str(tmp_path / "o"), "--profile", "a-remap-decoy"],
    )

    assert result.exit_code == 0, result.output
    assert "공백과 문장부호를 빼면" in result.output
    assert "그대로 읽힌다" not in result.output


def test_cli_warns_when_most_letters_survive_the_scrambling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """글자 대응표를 읽어 내지 못한 글꼴의 글자는 그대로 남는다. 그런 글자가 많으면 알린다."""
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "exam.pdf").write_bytes(two_font_pdf("가나다라", "plain words stay " * 6))
    # 대응표가 없는 글꼴(둘째 쪽)을 조사해도 아무것도 얻지 못한 경우를 흉내 낸다.
    monkeypatch.setattr("examshield.protect.techniques.glyph_remap.probe_font", lambda *_: {})

    result = CliRunner().invoke(
        app,
        [
            "protect",
            str(folder / "exam.pdf"),
            "--out",
            str(tmp_path / "o"),
            "--profile",
            "a-glyph-remap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "그대로 읽힌다" in result.output and "효과가 약하다" in result.output
