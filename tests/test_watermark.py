import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from examshield.check.compare import Expectation, compare
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text, save_deterministic
from examshield.profiles import load_profile, parse_profile
from examshield.protect.copies import (
    MapRow,
    map_row,
    protect_copies,
    read_examinee_ids,
    read_map,
)
from examshield.protect.pipeline import protect_bytes
from examshield.protect.techniques import watermark
from examshield.trace.detect import trace

runner = CliRunner()
SEED = b"0123456789abcdef-institution-seed"
IDS = [f"S{number:04d}" for number in range(40)]


def make_copies(pdf: bytes, profile: str = "a-watermark", count: int = 3, seed: bytes = SEED):
    base, made = protect_copies(pdf, load_profile(profile), seed, IDS)
    copies = [next(made) for _ in range(count)]
    return base, copies


def table(seed: bytes = SEED) -> list[MapRow]:
    return [
        MapRow(
            examinee_id=examinee,
            code=watermark.code_of(watermark.derive_key(seed, examinee)),
            file=f"{examinee}.pdf",
            output_sha256="-",
            cell_pt=8,
        )
        for examinee in IDS
    ]


def screenshot(
    pdf: bytes,
    dpi: int,
    suffix: str,
    crop: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    quality: int = 80,
    page: int = 0,
) -> bytes:
    pix = open_pdf_bytes(pdf)[page].get_pixmap(dpi=dpi)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    height, width = image.shape[:2]
    image = image[
        int(height * crop[0]) : int(height * crop[1]), int(width * crop[2]) : int(width * crop[3])
    ]
    ok, encoded = cv2.imencode(
        suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    assert ok
    return bytes(encoded.tobytes())


def test_every_copy_is_different_and_passes_the_check(sample_pdf: bytes) -> None:
    base, copies = make_copies(sample_pdf)

    assert base.passed
    assert all(copy.check.passed for copy in copies), copies[0].check.differences
    assert len({copy.protected.pdf for copy in copies}) == 3
    assert all(copy.check.visual_max_diff <= 8 for copy in copies)


def test_same_seed_and_number_give_the_same_copy_and_another_seed_does_not(
    sample_pdf: bytes,
) -> None:
    first = make_copies(sample_pdf)[1][0].protected.pdf

    assert make_copies(sample_pdf)[1][0].protected.pdf == first
    assert make_copies(sample_pdf, seed=SEED + b"x")[1][0].protected.pdf != first


def test_the_text_layer_is_untouched_by_the_watermark(sample_pdf: bytes) -> None:
    copy = make_copies(sample_pdf, count=1)[1][0]

    assert [page_text(page) for page in open_pdf_bytes(copy.protected.pdf)] == [
        page_text(page) for page in open_pdf_bytes(sample_pdf)
    ]


@pytest.mark.parametrize("profile", ("a-watermark", "a-full"))
def test_the_number_is_recovered_from_a_leaked_pdf(profile: str, sample_pdf: bytes) -> None:
    _, copies = make_copies(sample_pdf, profile)

    for copy in copies:
        # 뒤에 한 바이트를 붙여 해시 대조로는 못 찾게 하고, 무늬로 찾는지 본다.
        result = trace(copy.protected.pdf + b"\n", ".pdf", sample_pdf, table(), SEED)
        assert result.found and result.examinee_id == copy.examinee_id
        assert result.method == "PDF 렌더링"
        assert result.score is not None and result.score > 20


def test_an_untouched_leaked_file_is_matched_by_its_hash(sample_pdf: bytes) -> None:
    _, copies = make_copies(sample_pdf, count=1)
    rows = [map_row(copies[0], SEED, "x.pdf"), *table()[1:]]
    result = trace(copies[0].protected.pdf, ".pdf", sample_pdf, rows, SEED)

    assert result.found and result.method == "파일 해시 일치"
    assert result.examinee_id == copies[0].examinee_id


@pytest.mark.parametrize(
    ("dpi", "suffix", "crop", "quality"),
    [
        (110, ".png", (0.0, 1.0, 0.0, 1.0), 80),
        (96, ".png", (0.05, 0.6, 0.05, 0.95), 80),
        (110, ".jpg", (0.05, 0.6, 0.05, 0.95), 80),
        (96, ".jpg", (0.05, 0.5, 0.1, 0.9), 60),
        (72, ".png", (0.0, 1.0, 0.0, 1.0), 80),
        (220, ".png", (0.0, 0.5, 0.0, 1.0), 80),
    ],
)
def test_the_number_is_recovered_from_a_screenshot(
    dpi: int, suffix: str, crop: tuple[float, float, float, float], quality: int, sample_pdf: bytes
) -> None:
    _, copies = make_copies(sample_pdf, "a-full")
    target = copies[2]
    shot = screenshot(target.protected.pdf, dpi, suffix, crop, quality, page=1)
    result = trace(shot, suffix, sample_pdf, table(), SEED)

    assert result.found, result
    assert result.examinee_id == target.examinee_id
    assert result.pages == (2,)
    assert result.score is not None and result.runner_up is not None
    assert result.score - result.runner_up > 10


def test_a_rasterized_copy_still_carries_the_watermark(sample_pdf: bytes) -> None:
    profile = parse_profile(
        {
            "name": "x",
            "scenario": "A",
            "status": "experimental",
            "techniques": [{"id": "rasterize"}, {"id": "watermark"}],
        }
    )
    base, made = protect_copies(sample_pdf, profile, SEED, IDS)
    copy = next(made)
    shot = screenshot(copy.protected.pdf, 110, ".png")

    assert base.passed and copy.check.passed, copy.check.differences
    assert compare(sample_pdf, copy.protected.pdf, copy.protected.expectation).passed
    assert trace(shot, ".png", sample_pdf, table(), SEED).examinee_id == copy.examinee_id


def test_a_document_without_a_watermark_names_nobody(sample_pdf: bytes) -> None:
    plain = protect_bytes(sample_pdf, load_profile("a-remap-decoy")).pdf

    for leaked, suffix in ((plain, ".pdf"), (screenshot(sample_pdf, 110, ".png"), ".png")):
        result = trace(leaked, suffix, sample_pdf, table(), SEED)
        assert not result.found
        assert result.examinee_id is None


def test_a_copy_made_for_someone_outside_the_table_names_nobody(sample_pdf: bytes) -> None:
    _, made = protect_copies(sample_pdf, load_profile("a-watermark"), SEED, ["OUTSIDER"])
    result = trace(next(made).protected.pdf, ".pdf", sample_pdf, table(), SEED)

    assert not result.found
    assert result.score is not None and result.score < 6


def test_known_limit_a_thresholded_screenshot_loses_the_watermark_without_a_wrong_answer(
    sample_pdf: bytes,
) -> None:
    _, copies = make_copies(sample_pdf, count=1)
    shot = cv2.imdecode(
        np.frombuffer(screenshot(copies[0].protected.pdf, 110, ".png"), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    cleaned = np.where(shot > 200, 255, shot).astype(np.uint8)
    result = trace(
        bytes(cv2.imencode(".png", cleaned)[1].tobytes()), ".png", sample_pdf, table(), SEED
    )

    assert not result.found
    assert result.examinee_id is None


def test_an_unrelated_picture_names_nobody(sample_pdf: bytes) -> None:
    noise = np.random.default_rng(7).integers(0, 255, (600, 800), dtype=np.uint8)
    result = trace(
        bytes(cv2.imencode(".png", noise)[1].tobytes()), ".png", sample_pdf, table(), SEED
    )

    assert not result.found


def test_the_wrong_seed_is_noticed_before_any_guess(sample_pdf: bytes) -> None:
    _, copies = make_copies(sample_pdf, count=1)

    with pytest.raises(ExamShieldError, match="시드"):
        trace(copies[0].protected.pdf + b"\n", ".pdf", sample_pdf, table(), SEED + b"wrong")


def test_a_content_change_hidden_in_a_watermarked_copy_fails_the_check(sample_pdf: bytes) -> None:
    copy = make_copies(sample_pdf, count=1)[1][0]
    doc = open_pdf_bytes(copy.protected.pdf)
    doc[0].insert_text((300, 500), ".", fontsize=7)
    result = compare(sample_pdf, save_deterministic(doc), copy.protected.expectation)

    assert not result.passed


def test_a_job_record_cannot_widen_the_tolerance_beyond_the_cap() -> None:
    with pytest.raises(ExamShieldError):
        Expectation.from_dict({"pixel_tolerance": 60})
    with pytest.raises(ExamShieldError):
        Expectation.from_dict({"overlay_images": 5})
    with pytest.raises(ExamShieldError):
        Expectation.from_dict({"pixel_tolerance": 14})
    with pytest.raises(ExamShieldError):
        Expectation.from_dict({"overlay_images": 1, "pixel_tolerance": 8})


def test_strength_above_the_readability_cap_is_refused(sample_pdf: bytes) -> None:
    profile = parse_profile(
        {
            "name": "x",
            "scenario": "A",
            "status": "experimental",
            "techniques": [{"id": "watermark", "params": {"strength": 40}}],
        }
    )

    with pytest.raises(ExamShieldError, match="가독성"):
        next(protect_copies(sample_pdf, profile, SEED, ["A1"])[1])


def test_watermark_must_be_the_last_technique(sample_pdf: bytes) -> None:
    profile = parse_profile(
        {
            "name": "x",
            "scenario": "A",
            "status": "experimental",
            "techniques": [{"id": "watermark"}, {"id": "decoy"}],
        }
    )

    with pytest.raises(ExamShieldError, match="마지막"):
        protect_bytes(sample_pdf, profile, None, b"k")


def test_examinee_list_parsing(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text("﻿수험번호,이름\n2026-0001,홍길동\n\n2026-0002,김철수\n", encoding="utf-8")

    assert read_examinee_ids(path) == ["2026-0001", "2026-0002"]
    path.write_text("A1\nA1\n")
    with pytest.raises(ExamShieldError, match="겹친다"):
        read_examinee_ids(path)
    path.write_text("../etc/passwd\n")
    with pytest.raises(ExamShieldError, match="수험번호는"):
        read_examinee_ids(path)
    path.write_text("\n")
    with pytest.raises(ExamShieldError, match="비어"):
        read_examinee_ids(path)


@pytest.fixture
def workspace(sample_file: Path, tmp_path: Path) -> dict[str, Path]:
    ids = tmp_path / "ids.csv"
    ids.write_text("수험번호\nA001\nA002\nA003\n", encoding="utf-8")
    seed = tmp_path / "secrets" / "exam.seed"
    seed.parent.mkdir()
    seed.write_bytes(SEED)
    return {
        "source": sample_file,
        "ids": ids,
        "seed": seed,
        "out": tmp_path / "out",
        "map": tmp_path / "secrets" / "exam.watermark-map.csv",
    }


def run_copies(space: dict[str, Path], *extra: str, profile: str = "a-full") -> tuple[int, str]:
    args = [
        "protect", str(space["source"]), "--out", str(space["out"]), "--profile", profile,
        "--copies", str(space["ids"]), "--seed-file", str(space["seed"]),
        "--map-file", str(space["map"]), *extra,
    ]  # fmt: skip
    result = runner.invoke(app, args)
    return result.exit_code, result.output


def test_cli_makes_one_copy_per_number_and_keeps_the_table_apart(
    workspace: dict[str, Path],
) -> None:
    code, output = run_copies(workspace)
    out = workspace["out"]
    rows = read_map(workspace["map"])
    distributed = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in out.iterdir()
        if path.suffix in (".json", ".jsonl")
    )

    assert code == 0, output
    assert sorted(path.name for path in out.glob("*.pdf")) == [
        "exam-A001.pdf",
        "exam-A002.pdf",
        "exam-A003.pdf",
    ]
    assert [row.examinee_id for row in rows] == ["A001", "A002", "A003"]
    assert not list(out.glob("*.csv")) and not list(out.glob("*.seed"))
    assert all(row.code not in distributed for row in rows)
    assert SEED.decode() not in distributed
    record = json.loads((out / "exam-A002.job.json").read_text(encoding="utf-8"))
    assert record["check"]["passed"] and record["watermark_check"]["passed"]
    assert len((out / "protect.log.jsonl").read_text().splitlines()) == 3


def test_cli_check_accepts_each_copy_with_its_job_record(workspace: dict[str, Path]) -> None:
    run_copies(workspace)
    result = runner.invoke(
        app, ["check", str(workspace["source"]), str(workspace["out"] / "exam-A003.pdf")]
    )

    assert result.exit_code == 0, result.output


def test_cli_trace_finds_the_number_from_a_pdf_and_from_a_screenshot(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    run_copies(workspace)
    leaked_pdf = tmp_path / "leak.pdf"
    leaked_pdf.write_bytes((workspace["out"] / "exam-A002.pdf").read_bytes() + b"\n")
    leaked_shot = tmp_path / "leak.jpg"
    leaked_shot.write_bytes(
        screenshot(
            (workspace["out"] / "exam-A003.pdf").read_bytes(), 100, ".jpg", (0.0, 0.7, 0.0, 1.0)
        )
    )
    common = [
        "--original", str(workspace["source"]),
        "--map-file", str(workspace["map"]),
        "--seed-file", str(workspace["seed"]),
    ]  # fmt: skip
    from_pdf = runner.invoke(app, ["trace", str(leaked_pdf), *common])
    from_shot = runner.invoke(app, ["trace", str(leaked_shot), *common])
    from_original = runner.invoke(app, ["trace", str(workspace["source"]), *common])

    assert from_pdf.exit_code == 0 and "수험번호     A002" in from_pdf.output, from_pdf.output
    assert from_shot.exit_code == 0 and "수험번호     A003" in from_shot.output, from_shot.output
    assert from_original.exit_code == 1 and "찾지 못했다" in from_original.output


def test_cli_refuses_to_put_the_table_or_seed_where_copies_are_distributed(
    workspace: dict[str, Path],
) -> None:
    inside = dict(workspace, map=workspace["out"] / "map.csv")
    code, output = run_copies(inside)

    assert code == 2 and "출력 폴더 안에" in output
    assert not workspace["out"].exists() or not list(workspace["out"].glob("*.pdf"))


def test_cli_copies_need_a_seed_file_a_watermark_profile_and_a_long_enough_seed(
    workspace: dict[str, Path],
) -> None:
    base = ["protect", str(workspace["source"]), "--out", str(workspace["out"])]
    ids = ["--copies", str(workspace["ids"]), "--map-file", str(workspace["map"])]
    no_seed = runner.invoke(app, [*base, "--profile", "a-full", *ids])
    wrong_profile = runner.invoke(
        app, [*base, "--profile", "a-decoy", *ids, "--seed-file", str(workspace["seed"])]
    )
    workspace["seed"].write_bytes(b"short")
    short_seed = runner.invoke(
        app, [*base, "--profile", "a-full", *ids, "--seed-file", str(workspace["seed"])]
    )
    without_copies = runner.invoke(app, [*base, "--profile", "a-full"])

    assert no_seed.exit_code == 2 and "--seed-file" in no_seed.output
    assert wrong_profile.exit_code == 2 and "watermark" in wrong_profile.output
    assert short_seed.exit_code == 2 and "짧다" in short_seed.output
    assert without_copies.exit_code == 2 and "--copies" in without_copies.output
    assert not workspace["map"].exists()


def test_cli_does_not_overwrite_an_existing_table_without_force(workspace: dict[str, Path]) -> None:
    workspace["map"].write_text("someone's table")

    assert run_copies(workspace)[0] == 2
    assert workspace["map"].read_text() == "someone's table"
    assert run_copies(workspace, "--force")[0] == 0
    with workspace["map"].open(encoding="utf-8") as handle:
        assert next(csv.reader(handle))[0] == "examinee_id"


def test_verify_measures_a_watermark_profile_with_a_sample_pattern(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["verify", "--profile", "a-watermark", "--models", "mock", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "변환본" in result.output


def faint_text_overlay(pdf: bytes, alpha: int) -> bytes:
    import pymupdf  # noqa: PLC0415

    doc = open_pdf_bytes(pdf)
    for page in doc:
        canvas = np.zeros((200, 800), dtype=np.uint8)
        cv2.putText(canvas, "AI: answer (3)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 8)
        samples = np.zeros((200, 800, 2), dtype=np.uint8)
        samples[..., 1] = np.where(canvas > 0, alpha, 0)
        pixmap = pymupdf.Pixmap(pymupdf.csGRAY, 800, 200, samples.tobytes(), True)
        page.insert_image(pymupdf.Rect(60, 500, 460, 600), pixmap=pixmap, overlay=True)
    return save_deterministic(doc)


def test_a_faint_message_passed_off_as_the_watermark_fails_the_check(sample_pdf: bytes) -> None:
    forged = faint_text_overlay(sample_pdf, 7)
    claimed = Expectation(overlay_images=1, pixel_tolerance=8, overlay_cell_pt=8)
    result = compare(sample_pdf, forged, claimed)

    assert not result.passed
    assert any("워터마크 그림" in entry for entry in result.differences)


def watermarked(sample_pdf: bytes) -> tuple[bytes, Expectation]:
    made = protect_bytes(sample_pdf, load_profile("a-watermark"), None, b"k" * 32)
    return made.pdf, made.expectation


def repaint_watermark(pdf: bytes, change: object) -> bytes:
    """첫 쪽의 워터마크 그림을 고친 것으로 바꿔 그린다. 크기와 자리는 그대로다."""
    import pymupdf  # noqa: PLC0415

    doc = open_pdf_bytes(pdf)
    page = doc[0]
    rows, cols = watermark.grid_shape(page.rect.width, page.rect.height, 8)
    alpha = np.where(watermark.pattern(b"k" * 32, 0, rows, cols), 6, 0).astype(np.uint8)
    change(alpha)  # type: ignore[operator]
    samples = np.zeros((rows, cols, 2), dtype=np.uint8)
    samples[..., 1] = alpha
    doc.update_stream(page.get_contents()[-1], b" ")
    pixmap = pymupdf.Pixmap(pymupdf.csGRAY, cols, rows, samples.tobytes(), True)
    page.insert_image(pymupdf.Rect(0, 0, cols * 8, rows * 8), pixmap=pixmap, overlay=True)
    return save_deterministic(doc)


def test_a_real_watermark_passes_and_is_compared_exactly_once_lifted(sample_pdf: bytes) -> None:
    copy, expectation = watermarked(sample_pdf)

    result = compare(sample_pdf, copy, expectation)

    assert result.passed, result.differences
    # 무늬를 걷어 낸 쪽을 견주므로, 무늬의 세기(6)만큼의 차이는 더는 보이지 않는다.
    assert result.visual_max_diff is not None and result.visual_max_diff <= 2


def test_a_cell_of_another_strength_fails_the_check(sample_pdf: bytes) -> None:
    copy, expectation = watermarked(sample_pdf)

    def one_cell(alpha: np.ndarray) -> None:  # type: ignore[type-arg]
        alpha[10, 10] = 3

    result = compare(sample_pdf, repaint_watermark(copy, one_cell), expectation)

    assert any("세기와 다른 칸" in entry for entry in result.differences)


def test_a_watermark_that_is_not_random_fails_the_check(sample_pdf: bytes) -> None:
    copy, expectation = watermarked(sample_pdf)

    def all_dark(alpha: np.ndarray) -> None:  # type: ignore[type-arg]
        alpha[:] = 6

    result = compare(sample_pdf, repaint_watermark(copy, all_dark), expectation)

    assert any("무작위 무늬" in entry for entry in result.differences)


def faint_line_under_the_watermark(pdf: bytes, top: float) -> bytes:
    """워터마크 아래(먼저 그려지는 자리)에 아주 옅은 가는 줄을 넣는다. 칸 가장자리에 걸치게 한다."""
    import pymupdf  # noqa: PLC0415

    doc = open_pdf_bytes(pdf)
    for page in doc:
        shade = 1 - 4 / 255
        page.draw_rect(
            pymupdf.Rect(40, top, 400, top + 1.5),
            color=None,
            fill=(shade, shade, shade),
            overlay=False,
        )
    return save_deterministic(doc)


@pytest.mark.parametrize("top", [319.5, 323.0, 327.6])
def test_faint_content_hidden_at_the_edge_of_cells_fails_the_check(
    sample_pdf: bytes, top: float
) -> None:
    """예전 검사는 렌더링에서 칸 안쪽만 봐서, 칸 가장자리에 걸친 옅은 획을 놓쳤다."""
    copy, expectation = watermarked(sample_pdf)

    result = compare(sample_pdf, faint_line_under_the_watermark(copy, top), expectation)

    assert not result.passed
    assert any("렌더링이 다르다" in entry for entry in result.differences)


def test_ids_that_differ_only_in_case_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text("a1\nA1\n")

    with pytest.raises(ExamShieldError, match="겹친다"):
        read_examinee_ids(path)


def test_a_table_with_an_impossible_cell_size_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    path.write_text("examinee_id,code,file,output_sha256,cell_pt\nA1,x,f,h,0\n")

    with pytest.raises(ExamShieldError, match="cell_pt"):
        read_map(path)


def test_a_toolbar_over_part_of_the_screenshot_does_not_hide_the_watermark(
    sample_pdf: bytes,
) -> None:
    _, copies = make_copies(sample_pdf)
    shot = cv2.imdecode(
        np.frombuffer(screenshot(copies[1].protected.pdf, 110, ".png"), np.uint8), cv2.IMREAD_COLOR
    )
    shot[:60] = 40
    result = trace(
        bytes(cv2.imencode(".png", shot)[1].tobytes()), ".png", sample_pdf, table(), SEED
    )

    assert result.found and result.examinee_id == copies[1].examinee_id


def test_a_screenshot_of_a_page_that_repeats_is_still_traced() -> None:
    from tests.conftest import make_pdf  # noqa: PLC0415

    one = open_pdf_bytes(make_pdf("같은 내용이 두 쪽에 있다.\n① 가 ② 나 ③ 다 ④ 라", 14))
    one.insert_pdf(open_pdf_bytes(one.tobytes()))
    twin = save_deterministic(one)
    _, copies = make_copies(twin)
    shot = screenshot(copies[0].protected.pdf, 110, ".png", page=1)
    result = trace(shot, ".png", twin, table(), SEED)

    assert result.found and result.examinee_id == copies[0].examinee_id
    assert result.pages == (2,)


def test_two_copies_averaged_together_name_both_as_suspects_not_one(sample_pdf: bytes) -> None:
    _, copies = make_copies(sample_pdf)
    first, second = (
        cv2.imdecode(
            np.frombuffer(screenshot(copy.protected.pdf, 110, ".png"), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        ).astype(np.float64)
        for copy in copies[:2]
    )
    mixed = ((first + second) / 2).astype(np.uint8)
    result = trace(
        bytes(cv2.imencode(".png", mixed)[1].tobytes()), ".png", sample_pdf, table(), SEED
    )

    assert not result.found and result.examinee_id is None
    assert set(result.suspects) == {copies[0].examinee_id, copies[1].examinee_id}
    assert "여럿" in result.reason


def test_cli_refuses_the_table_inside_the_output_folder_under_another_spelling(
    workspace: dict[str, Path],
) -> None:
    workspace["out"].mkdir()
    alias = workspace["out"].parent / "OUT"
    if not alias.exists():
        pytest.skip("대소문자를 구분하는 파일 시스템이다")
    code, output = run_copies(dict(workspace, map=alias / "map.csv"))

    assert code == 2 and "출력 폴더 안에" in output
    assert list(workspace["out"].iterdir()) == []


def test_cli_checks_the_table_is_writable_before_making_any_copy(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    code, _ = run_copies(dict(workspace, map=blocker / "map.csv"))

    assert code == 2
    assert not list(workspace["out"].glob("*.pdf")) if workspace["out"].exists() else True


def test_cli_stops_when_old_copies_would_be_left_out_of_the_new_table(
    workspace: dict[str, Path],
) -> None:
    assert run_copies(workspace)[0] == 0
    workspace["ids"].write_text("A001\nA002\n")
    code, output = run_copies(workspace, "--force")

    assert code == 2 and "exam-A003.pdf" in output
    assert [row.examinee_id for row in read_map(workspace["map"])] == ["A001", "A002", "A003"]


def test_trace_stops_reading_pages_once_one_names_the_examinee(sample_pdf: bytes) -> None:
    from collections.abc import Iterator  # noqa: PLC0415

    from examshield.trace import detect  # noqa: PLC0415

    keys = [watermark.derive_key(SEED, examinee) for examinee in IDS]
    rows, cols = 100, 80
    rng = np.random.default_rng(3)
    read: list[int] = []

    def page_of(owner: bytes | None, index: int) -> detect.Evidence:
        cells = np.arange(rows * cols, dtype=np.intp)
        if owner is None:
            values = rng.normal(size=rows * cols)
        else:
            marks = watermark.pattern(owner, index, rows, cols).ravel()
            values = np.where(marks, 1.0, -1.0) + rng.normal(scale=2.0, size=rows * cols)
        return detect.Evidence(index, rows, cols, cells, values - values.mean())

    def pages(owners: list[bytes | None]) -> Iterator[detect.Evidence]:
        for index, owner in enumerate(owners):
            read.append(index)
            yield page_of(owner, index)

    found = detect._best_page(pages([None, keys[7], None, None]), IDS, keys, "이미지 정합")
    assert found.found and found.examinee_id == IDS[7] and found.pages == (2,)
    assert read == [0, 1]

    read.clear()
    nobody = detect._best_page(pages([None, None, None]), IDS, keys, "이미지 정합")
    assert not nobody.found and read == [0, 1, 2]
    assert not detect._best_page(iter(()), IDS, keys, "이미지 정합").found
