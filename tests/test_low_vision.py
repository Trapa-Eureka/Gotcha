"""저시력 수험생용 복사본: 글자와 그림 둘레를 비우고 여백에만 무늬를 넣는다."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.check.compare import Expectation, compare
from examshield.clearzone import DEFAULT_CLEAR_PT, busy_cells, open_cells, reach
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes
from examshield.profiles import load_profile
from examshield.protect.copies import Copy, MapRow, map_row, protect_copies, read_map
from examshield.protect.techniques import watermark
from examshield.trace.detect import MIN_CELLS, Evidence, _decide, trace
from tests.test_watermark import screenshot

runner = CliRunner()
SEED = b"0123456789abcdef-institution-seed"
IDS = [f"S{number:04d}" for number in range(12)]
LOW = "S0003"
PLAIN = "S0001"


def make(pdf: bytes, profile: str, low: tuple[str, ...] = (LOW,)) -> dict[str, Copy]:
    _, made = protect_copies(pdf, load_profile(profile), SEED, IDS, low_vision=frozenset(low))
    return {copy.examinee_id: copy for copy in made}


def rows_of(copies: dict[str, Copy]) -> list[MapRow]:
    return [map_row(copy, SEED, f"exam-{name}.pdf") for name, copy in copies.items()]


def changed_cells(original: bytes, copy: bytes, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """첫 쪽에서 무늬가 들어간 칸과, 내용이 있는 칸. 무늬는 덧씌운 그림에서 직접 읽는다."""
    before = open_pdf_bytes(original)[0]
    rows, cols = watermark.grid_shape(before.rect.width, before.rect.height, cell)
    doc = open_pdf_bytes(copy)
    # 쪽들이 그림 목록을 같이 쓴다. 첫 쪽에 넣은 그림이 목록의 첫째다.
    mask = pymupdf.Pixmap(doc, doc[0].get_images()[0][1])
    alpha = np.frombuffer(mask.samples, dtype=np.uint8).reshape(mask.height, mask.stride)
    alpha = alpha[:, : mask.width]
    per_cell = mask.height // rows
    marked = alpha.reshape(rows, per_cell, cols, per_cell).any(axis=(1, 3))
    return marked, busy_cells(before, cell, rows, cols)


@pytest.fixture(scope="module")
def watermarked(sample_pdf: bytes) -> dict[str, Copy]:
    return make(sample_pdf, "a-watermark")


@pytest.fixture(scope="module")
def dotted(sample_pdf: bytes) -> dict[str, Copy]:
    return make(sample_pdf, "b-dots")


@pytest.mark.parametrize(("kind", "cell"), [("watermarked", 8), ("dotted", 6)])
def test_the_pattern_stays_away_from_letters_and_pictures(
    kind: str, cell: int, sample_pdf: bytes, request: pytest.FixtureRequest
) -> None:
    copies: dict[str, Copy] = request.getfixturevalue(kind)
    marked, busy = changed_cells(sample_pdf, copies[LOW].protected.pdf, cell)
    usual, _ = changed_cells(sample_pdf, copies[PLAIN].protected.pdf, cell)
    page = open_pdf_bytes(sample_pdf)[0]
    allowed = open_cells(page, cell, DEFAULT_CLEAR_PT, *marked.shape)
    steps = reach(DEFAULT_CLEAR_PT, cell)
    near = np.zeros_like(busy)
    for row, col in zip(*np.nonzero(busy), strict=True):
        near[max(row - steps, 0) : row + steps + 1, max(col - steps, 0) : col + steps + 1] = True

    assert busy.any() and marked.sum() > 200
    assert not (marked & near).any()
    assert not (marked & ~allowed).any()
    # 보통 복사본은 글자 둘레에도 무늬가 있다.
    assert (usual & near & ~busy).sum() > 100


@pytest.mark.parametrize("kind", ["watermarked", "dotted"])
def test_a_low_vision_copy_passes_the_check_with_its_job_record(
    kind: str, sample_pdf: bytes, request: pytest.FixtureRequest
) -> None:
    copy = request.getfixturevalue(kind)[LOW]
    record = json.loads(json.dumps(copy.protected.record))
    expectation = Expectation.from_dict(record["expect"])

    assert copy.check.passed, copy.check.differences
    assert expectation.clear_pt == DEFAULT_CLEAR_PT
    assert compare(sample_pdf, copy.protected.pdf, expectation).passed
    assert record["techniques"][-1]["stats"]["clear_pt"] == DEFAULT_CLEAR_PT
    assert 0.0 < record["techniques"][-1]["stats"]["open_share"] < 1.0


def test_other_copies_are_the_same_bytes_as_without_the_option(
    sample_pdf: bytes, watermarked: dict[str, Copy]
) -> None:
    usual = make(sample_pdf, "a-watermark", low=())

    assert usual[PLAIN].protected.pdf == watermarked[PLAIN].protected.pdf
    assert usual[PLAIN].protected.record == watermarked[PLAIN].protected.record
    assert "clear_pt" not in watermarked[PLAIN].protected.record["expect"]
    assert usual[LOW].protected.pdf != watermarked[LOW].protected.pdf


def test_an_ordinary_copy_claimed_as_low_vision_fails_the_check(
    sample_pdf: bytes, watermarked: dict[str, Copy]
) -> None:
    copy = watermarked[PLAIN]
    claimed = replace(copy.protected.expectation, clear_pt=DEFAULT_CLEAR_PT)

    result = compare(sample_pdf, copy.protected.pdf, claimed)

    assert not result.passed and "둘레" in result.differences[0]


@pytest.mark.parametrize("value", [0, 5, 73])
def test_a_job_record_with_an_impossible_clear_width_is_refused(value: int) -> None:
    record = {"overlay_images": 1, "pixel_tolerance": 8, "overlay_cell_pt": 8, "clear_pt": value}

    with pytest.raises(ExamShieldError):
        Expectation.from_dict(record)


@pytest.mark.parametrize("kind", ["watermarked", "dotted"])
def test_a_low_vision_copy_is_traced_from_a_resaved_pdf(
    kind: str, sample_pdf: bytes, request: pytest.FixtureRequest
) -> None:
    copies: dict[str, Copy] = request.getfixturevalue(kind)
    rows = rows_of(copies)
    doc = open_pdf_bytes(copies[LOW].protected.pdf)
    doc.set_metadata({"title": "leaked"})

    low = trace(bytes(doc.tobytes()), ".pdf", sample_pdf, rows, SEED)
    doc = open_pdf_bytes(copies[PLAIN].protected.pdf)
    doc.set_metadata({"title": "leaked"})
    plain = trace(bytes(doc.tobytes()), ".pdf", sample_pdf, rows, SEED)

    assert [row.clear_pt for row in rows if row.examinee_id == LOW] == [DEFAULT_CLEAR_PT]
    assert low.found and low.examinee_id == LOW and low.method == "PDF 렌더링"
    assert plain.found and plain.examinee_id == PLAIN


def test_a_low_vision_copy_is_traced_from_a_capture_of_the_whole_page(
    sample_pdf: bytes, watermarked: dict[str, Copy]
) -> None:
    shot = screenshot(watermarked[LOW].protected.pdf, 110, ".png")

    result = trace(shot, ".png", sample_pdf, rows_of(watermarked), SEED)

    assert result.found and result.examinee_id == LOW


def test_too_few_margin_cells_name_nobody_and_say_why() -> None:
    """글자 둘레만 잘라 낸 그림. 무늬가 들어가는 칸이 모자라면 저시력 수험번호는 가리키지 않는다."""
    keys = [watermark.derive_key(SEED, name) for name in ("A", "B")]
    rows, cols = 100, 70
    cells = np.arange(2000, dtype=np.intp)
    opened = np.zeros(2000, dtype=bool)
    opened[: MIN_CELLS - 1] = True
    signs = watermark.pattern(keys[0], 0, rows, cols).ravel()[cells]
    values = np.where(signs & opened, 6.0, 0.0)
    evidence = Evidence(0, rows, cols, cells, values - values.mean(), opened)

    result = _decide([evidence], ["A", "B"], keys, "이미지 정합", (True, False))

    assert not result.found and "저시력" in result.reason


def workspace(tmp_path: Path, sample_file: Path) -> dict[str, Path]:
    secrets = tmp_path / "secrets"
    secrets.mkdir(exist_ok=True)
    (secrets / "exam.seed").write_bytes(SEED)
    (tmp_path / "ids.csv").write_text("수험번호\nA001\nA002\nA003\n", encoding="utf-8")
    (tmp_path / "low.csv").write_text("A002\n", encoding="utf-8")
    return {
        "source": sample_file,
        "ids": tmp_path / "ids.csv",
        "low": tmp_path / "low.csv",
        "seed": secrets / "exam.seed",
        "map": secrets / "exam.map.csv",
        "out": tmp_path / "out",
    }


def copies(space: dict[str, Path], *extra: str) -> tuple[int, str]:
    args = [
        "protect", str(space["source"]), "--out", str(space["out"]), "--profile", "a-full",
        "--copies", str(space["ids"]), "--seed-file", str(space["seed"]),
        "--map-file", str(space["map"]), *extra,
    ]  # fmt: skip
    result = runner.invoke(app, args)
    return result.exit_code, result.output


def test_cli_marks_low_vision_copies_in_the_table_and_checks_them(
    tmp_path: Path, sample_file: Path
) -> None:
    space = workspace(tmp_path, sample_file)

    code, output = copies(space, "--low-vision", str(space["low"]))
    table = {row.examinee_id: row.clear_pt for row in read_map(space["map"])}
    checked = runner.invoke(
        app,
        [
            "check",
            str(space["source"]),
            str(space["out"] / "exam-A002.pdf"),
            "--job",
            str(space["out"] / "exam-A002.job.json"),
        ],  # fmt: skip
    )
    traced = runner.invoke(
        app,
        [
            "trace",
            str(space["out"] / "exam-A002.pdf"),
            "--original",
            str(space["source"]),
            "--map-file",
            str(space["map"]),
            "--seed-file",
            str(space["seed"]),
        ],  # fmt: skip
    )

    assert code == 0, output
    assert "저시력 수험생용 복사본 1개" in output
    assert table == {"A001": 0, "A002": DEFAULT_CLEAR_PT, "A003": 0}
    assert checked.exit_code == 0, checked.output
    assert traced.exit_code == 0 and "A002" in traced.output


def test_cli_refuses_a_low_vision_number_outside_the_list_or_without_copies(
    tmp_path: Path, sample_file: Path
) -> None:
    space = workspace(tmp_path, sample_file)
    space["low"].write_text("A002\nZ999\n", encoding="utf-8")

    code, output = copies(space, "--low-vision", str(space["low"]))
    alone = runner.invoke(
        app,
        [
            "protect",
            str(space["source"]),
            "--out",
            str(space["out"]),
            "--profile",
            "a-glyph-remap",
            "--low-vision",
            str(space["low"]),
        ],  # fmt: skip
    )
    inside = space["out"] / "low.csv"
    space["out"].mkdir(exist_ok=True)
    inside.write_text("A002\n", encoding="utf-8")
    placed = copies(space, "--low-vision", str(inside))

    assert code == 2 and "Z999" in output and not list(space["out"].glob("*.pdf"))
    assert alone.exit_code == 2 and "--copies" in alone.output
    assert placed[0] == 2 and "출력 폴더" in placed[1]


def test_resume_remakes_only_the_copy_whose_kind_changed(tmp_path: Path, sample_file: Path) -> None:
    space = workspace(tmp_path, sample_file)
    copies(space)
    before = {path.name: path.read_bytes() for path in space["out"].glob("*.pdf")}

    code, output = copies(space, "--resume", "--low-vision", str(space["low"]))
    after = {path.name: path.read_bytes() for path in space["out"].glob("*.pdf")}

    assert code == 0, output
    assert "2개는 그대로 두고 1개" in output
    assert [name for name in before if before[name] != after[name]] == ["exam-A002.pdf"]


def test_a_table_from_before_the_option_still_reads(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["examinee_id", "code", "file", "output_sha256", "cell_pt", "kind"])
        writer.writerow(["A1", "0" * 16, "x-A1.pdf", "-", 8, "watermark"])
    wrong = tmp_path / "wrong.csv"
    wrong.write_text(path.read_text().replace("kind", "kind,clear_pt").replace("mark", "mark,500"))

    assert [row.clear_pt for row in read_map(path)] == [0]
    with pytest.raises(ExamShieldError):
        read_map(wrong)
