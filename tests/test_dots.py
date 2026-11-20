"""시나리오 B: 인쇄용 점 무늬, 사진에서 수험번호 복원, 촬영 보정."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.benchmark import load_items
from examshield.check.compare import Expectation, compare
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import GrayImage, open_pdf_bytes, render_gray, save_deterministic
from examshield.profiles import load_profile, parse_profile
from examshield.protect.copies import (
    DOTS,
    WATERMARK,
    MapRow,
    map_row,
    protect_copies,
    read_map,
    write_map,
)
from examshield.protect.pipeline import protect_and_check, protect_bytes
from examshield.protect.techniques import dots
from examshield.trace.detect import trace
from examshield.verify.calibrate import measure, reference_pages, simulated, suggest
from examshield.verify.capture_sim import CaptureConditions, encode_png, simulate_capture
from examshield.verify.models.base import IMAGE, ModelInfo, ModelReply, PageInput
from examshield.verify.runner import run_verification
from tests.conftest import make_pdf

SEED = b"0123456789abcdef-print-seed"
IDS = [f"2026-{number:04d}" for number in range(1, 13)]
PROFILE = load_profile("b-dots")
runner = CliRunner()


def photograph(pdf: bytes, page: int, conditions: CaptureConditions) -> GrayImage:
    printed = render_gray(open_pdf_bytes(pdf)[page], conditions.print_dpi)
    return simulate_capture(printed, conditions)


def jpeg(image: GrayImage) -> bytes:
    return bytes(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())


@pytest.fixture(scope="module")
def copies(sample_pdf: bytes) -> list[Any]:
    base, made = protect_copies(sample_pdf, PROFILE, SEED, IDS)
    result = list(made)
    assert base.passed and all(copy.check.passed for copy in result)
    return result


@pytest.fixture(scope="module")
def table(copies: list[Any]) -> list[MapRow]:
    return [map_row(copy, SEED, f"exam-{copy.examinee_id}.pdf") for copy in copies]


def test_dots_are_deterministic_and_pass_the_check(sample_pdf: bytes) -> None:
    first, result = protect_and_check(sample_pdf, PROFILE)
    second = protect_bytes(sample_pdf, PROFILE)
    other = protect_bytes(sample_pdf, PROFILE, b"another seed")
    stats = first.record["techniques"][0]["stats"]

    assert result.passed, result.differences
    assert first.pdf == second.pdf
    assert first.pdf != other.pdf
    assert result.text == "identical"
    assert first.record["text_layer_similarity"] == 1.0
    assert 0 < stats["coverage"] <= dots.MAX_COVERAGE
    assert first.expectation.dots_cell_pt == 6 and first.expectation.dots_ink == 255


def test_the_page_really_carries_visible_dots(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, PROFILE)
    before = render_gray(open_pdf_bytes(sample_pdf)[0], 200)
    after = render_gray(open_pdf_bytes(protected.pdf)[0], 200)

    changed = np.abs(before.astype(int) - after.astype(int)) > 60

    assert 0.001 < changed.mean() < 0.02


@pytest.mark.parametrize("params", [{"cell_pt": 3}, {"cell_pt": 17}, {"ink": 63}, {"ink": 256}])
def test_readability_limits_are_enforced(params: dict[str, int], sample_pdf: bytes) -> None:
    profile = parse_profile(
        {
            "name": "x",
            "scenario": "B",
            "status": "experimental",
            "techniques": [{"id": "dots", "params": params}],
        }
    )

    with pytest.raises(ExamShieldError, match="dots"):
        protect_bytes(sample_pdf, profile)


def tampered(pdf: bytes, change: Any) -> bytes:
    doc = open_pdf_bytes(pdf)
    change(doc)
    return save_deterministic(doc)


def repaint_overlay(doc: Any, paint: Any) -> None:
    """첫 쪽의 무늬 그림을 고친 것으로 바꿔 그린다."""
    page = doc[0]
    cell = 6 * dots.PX_PER_PT
    rows, cols = -(-int(page.rect.height) // 6), -(-int(page.rect.width) // 6)
    samples = np.zeros((rows * cell, cols * cell, 2), dtype=np.uint8)
    paint(samples)
    doc.update_stream(page.get_contents()[-1], b" ")
    pixmap = pymupdf.Pixmap(
        pymupdf.csGRAY, samples.shape[1], samples.shape[0], samples.tobytes(), True
    )
    page.insert_image(pymupdf.Rect(0, 0, cols * 6, rows * 6), pixmap=pixmap, overlay=True)


def off_lattice(samples: Any) -> None:
    samples[6, 6, 1] = 255
    samples[7, 7, 1] = 255


def wrong_ink(samples: Any) -> None:
    samples[6, 6, 1] = 90


def gray_dots(samples: Any) -> None:
    samples[6, 6] = (200, 255)


def text_on_top(doc: Any) -> None:
    doc[0].insert_text((100, 500), "3", fontsize=9)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda doc: repaint_overlay(doc, off_lattice), "칸 가운데가 아닌 점"),
        (lambda doc: repaint_overlay(doc, wrong_ink), "다른 진하기"),
        (lambda doc: repaint_overlay(doc, gray_dots), "검은 점만"),
        (text_on_top, "마지막에 그려진 것"),
        (
            lambda doc: doc[0].draw_rect(pymupdf.Rect(300, 600, 301, 601), color=(0.5,) * 3),
            "마지막에 그려진 것",
        ),
        (lambda doc: doc.update_stream(doc[0].get_contents()[-1], b" "), ""),
    ],
    ids=["off-lattice", "wrong-ink", "gray-dots", "text-on-top", "extra-drawing", "no-overlay"],
)
def test_check_refuses_an_overlay_that_is_not_the_promised_dots(
    change: Any, expected: str, sample_pdf: bytes
) -> None:
    protected = protect_bytes(sample_pdf, PROFILE)

    result = compare(sample_pdf, tampered(protected.pdf, change), protected.expectation)

    assert not result.passed
    assert any(entry.startswith("1쪽") and expected in entry for entry in result.differences), (
        result.differences
    )


def test_check_sees_content_changed_under_the_dots(sample_pdf: bytes) -> None:
    changed = open_pdf_bytes(sample_pdf)
    changed[1].insert_text((100, 500), "정답은 3번", fontname="korea", fontsize=9)
    protected = protect_bytes(save_deterministic(changed), PROFILE)

    result = compare(sample_pdf, protected.pdf, protected.expectation)

    assert any("2쪽: 렌더링이 다르다" in entry for entry in result.differences)


@pytest.mark.parametrize(
    "fields",
    [
        {"dots_cell_pt": 6},
        {"dots_cell_pt": 3, "dots_ink": 255},
        {"dots_cell_pt": 6, "dots_ink": 10},
        {"dots_cell_pt": 6, "dots_ink": 255, "overlay_images": 0},
        {"dots_cell_pt": 6, "dots_ink": 255, "pixel_tolerance": 9},
        {"dots_cell_pt": 6, "dots_ink": 255, "overlay_cell_pt": 8},
    ],
)
def test_job_records_with_impossible_dot_values_are_refused(fields: dict[str, int]) -> None:
    with pytest.raises(ExamShieldError):
        Expectation.from_dict({"overlay_images": 1, **fields})


def test_a_job_record_round_trips(sample_pdf: bytes) -> None:
    expectation = protect_bytes(sample_pdf, PROFILE).expectation

    assert Expectation.from_dict(json.loads(json.dumps(expectation.as_dict()))) == expectation


def test_every_examinee_gets_a_different_pattern(copies: list[Any], table: list[MapRow]) -> None:
    assert len({copy.protected.pdf for copy in copies}) == len(IDS)
    assert {row.kind for row in table} == {DOTS}
    assert copies[0].protected.record["techniques"][-1]["id"] == "dots"
    assert "watermark_check" in copies[0].protected.record


CONDITIONS = {
    "default": CaptureConditions(),
    "tilted": CaptureConditions(skew_degrees=8, perspective=0.08, seed=3),
    "dim": CaptureConditions(illumination=0.5, noise_sigma=6, seed=5),
    "close": CaptureConditions(capture_width=3000, blur_sigma=1.2, seed=7),
    "far": CaptureConditions(capture_width=1200, blur_sigma=1.0, seed=9),
}


@pytest.mark.parametrize("name", sorted(CONDITIONS))
def test_the_examinee_is_found_from_a_photo_of_the_printed_page(
    name: str, sample_pdf: bytes, copies: list[Any], table: list[MapRow]
) -> None:
    photo = jpeg(photograph(copies[4].protected.pdf, 1, CONDITIONS[name]))

    result = trace(photo, ".jpg", sample_pdf, table, SEED)

    assert result.found and result.examinee_id == IDS[4], result
    assert result.method == "사진 정합" and result.pages == (2,)
    assert result.score is not None and result.score >= 10


def test_a_scribbled_photo_still_names_the_examinee(
    sample_pdf: bytes, copies: list[Any], table: list[MapRow]
) -> None:
    photo = photograph(copies[9].protected.pdf, 0, CaptureConditions())
    cv2.line(photo, (200, 900), (1300, 1500), 20, 14)
    cv2.rectangle(photo, (900, 1700), (1400, 2000), 0, -1)

    result = trace(jpeg(photo), ".jpg", sample_pdf, table, SEED)

    assert result.found and result.examinee_id == IDS[9]


def test_nobody_is_named_for_a_photo_without_the_pattern(
    sample_pdf: bytes, copies: list[Any], table: list[MapRow]
) -> None:
    stranger = protect_bytes(sample_pdf, PROFILE, None, b"someone-not-in-the-table").pdf
    unrelated = np.full((1200, 900), 240, dtype=np.uint8)
    cv2.putText(unrelated, "hello", (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 4, 0, 8)

    plain = trace(
        jpeg(photograph(sample_pdf, 1, CaptureConditions())), ".jpg", sample_pdf, table, SEED
    )
    other = trace(
        jpeg(photograph(stranger, 1, CaptureConditions())), ".jpg", sample_pdf, table, SEED
    )
    nothing = trace(jpeg(unrelated), ".jpg", sample_pdf, table, SEED)

    assert not plain.found and not other.found and not nothing.found
    assert plain.examinee_id is None and other.examinee_id is None


def test_a_dotted_pdf_is_traced_by_hash_and_by_rendering(
    sample_pdf: bytes, copies: list[Any], table: list[MapRow]
) -> None:
    exact = trace(copies[2].protected.pdf, ".pdf", sample_pdf, table, SEED)
    resaved = open_pdf_bytes(copies[2].protected.pdf)
    resaved.set_metadata({"title": "유출"})

    rendered = trace(resaved.tobytes(), ".pdf", sample_pdf, table, SEED)
    screenshot = trace(
        encode_png(render_gray(open_pdf_bytes(copies[2].protected.pdf)[2], 130)),
        ".png",
        sample_pdf,
        table,
        SEED,
    )

    assert exact.method == "파일 해시 일치" and exact.examinee_id == IDS[2]
    assert rendered.method == "PDF 렌더링" and rendered.examinee_id == IDS[2]
    assert screenshot.found and screenshot.examinee_id == IDS[2]


def test_the_wrong_seed_is_refused(sample_pdf: bytes, table: list[MapRow]) -> None:
    with pytest.raises(ExamShieldError, match="시드"):
        trace(b"x", ".jpg", sample_pdf, table, b"another-seed-0123456789")


def test_mapping_tables_keep_the_kind_and_old_ones_still_read(
    tmp_path: Path, table: list[MapRow]
) -> None:
    path = tmp_path / "map.csv"
    write_map(path, table)
    old = tmp_path / "old.csv"
    old.write_text(
        "examinee_id,code,file,output_sha256,cell_pt\nA1,abcd,a.pdf,00,8\n", encoding="utf-8"
    )
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "examinee_id,code,file,output_sha256,cell_pt,kind\nA1,abcd,a.pdf,00,8,hologram\n",
        encoding="utf-8",
    )
    wide = tmp_path / "wide.csv"
    wide.write_text(
        "examinee_id,code,file,output_sha256,cell_pt,kind\nA1,abcd,a.pdf,00,20,dots\n",
        encoding="utf-8",
    )

    assert read_map(path) == table
    assert read_map(old)[0].kind == WATERMARK
    for broken in (bad, wide):
        with pytest.raises(ExamShieldError):
            read_map(broken)


def test_cli_makes_dotted_copies_and_traces_a_photo(
    sample_file: Path, sample_pdf: bytes, tmp_path: Path
) -> None:
    seed = tmp_path / "exam.seed"
    seed.write_bytes(SEED)
    ids = tmp_path / "ids.csv"
    ids.write_text("수험번호\nB1\nB2\nB3\n", encoding="utf-8")
    out, mapping = tmp_path / "out", tmp_path / "map.csv"
    made = runner.invoke(
        app,
        [
            "protect", str(sample_file), "--out", str(out), "--profile", "b-dots",
            "--copies", str(ids), "--seed-file", str(seed), "--map-file", str(mapping),
        ],
    )  # fmt: skip
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(jpeg(photograph((out / "exam-B2.pdf").read_bytes(), 0, CaptureConditions())))
    clean = tmp_path / "clean.jpg"
    clean.write_bytes(jpeg(photograph(sample_pdf, 0, CaptureConditions())))
    base = ["--original", str(sample_file), "--map-file", str(mapping), "--seed-file", str(seed)]

    traced = runner.invoke(app, ["trace", str(photo), *base])
    missed = runner.invoke(app, ["trace", str(clean), *base])
    checked = runner.invoke(app, ["check", str(sample_file), str(out / "exam-B2.pdf")])

    assert made.exit_code == 0, made.output
    assert "kind" in mapping.read_text(encoding="utf-8").splitlines()[0]
    assert traced.exit_code == 0 and "수험번호     B2" in traced.stdout, traced.output
    assert missed.exit_code == 1 and "찾지 못했다" in missed.stdout
    assert checked.exit_code == 0, checked.output


def test_copies_need_a_profile_with_a_keyed_technique(sample_file: Path, tmp_path: Path) -> None:
    seed = tmp_path / "exam.seed"
    seed.write_bytes(SEED)
    ids = tmp_path / "ids.csv"
    ids.write_text("B1\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "protect", str(sample_file), "--out", str(tmp_path / "out"), "--profile",
            "none-print", "--copies", str(ids), "--seed-file", str(seed),
            "--map-file", str(tmp_path / "map.csv"),
        ],
    )  # fmt: skip

    assert result.exit_code == 2
    assert "b-dots" in result.stderr


class ImageReader:
    """그림 경로만 받는 가짜 모델. 시나리오 B의 측정 흐름이 도는지 본다."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.info = ModelInfo(
            provider="test", name="image-reader", version="1", kind="mock",
            channels=(IMAGE,), solves=True,
        )  # fmt: skip

    def run(self, page: PageInput, channel: str) -> ModelReply:
        self.sizes.append(len(page.image_png))
        return ModelReply(transcription="", answer=1)


def test_verify_measures_the_dotted_page_through_the_capture_simulation() -> None:
    reader = ImageReader()
    key = b"examshield-verify-sample"

    report = run_verification(
        PROFILE,
        load_items(),
        [reader],
        now=datetime(2026, 11, 20, tzinfo=UTC),
        transform=lambda pdf: protect_bytes(pdf, PROFILE, None, key),
    )

    assert [(row["channel"], row["variant"]) for row in report["results"]] == [
        ("image", "control"),
        ("image", "protected"),
    ]
    assert report["simulation"]["capture_width"] == 1600
    assert report["protection"]["techniques"][0]["id"] == "dots"
    # 대조군과 변환본의 그림이 실제로 다르다.
    assert reader.sizes[:12] != reader.sizes[12:]


def test_calibration_reads_back_the_conditions_of_a_photo(sample_pdf: bytes) -> None:
    pages, texts = reference_pages(sample_pdf)
    soft = CaptureConditions(
        capture_width=2200, skew_degrees=5, illumination=0.5, blur_sigma=2.0, noise_sigma=8, seed=4
    )
    crisp = CaptureConditions(
        capture_width=1400,
        skew_degrees=0.5,
        illumination=0.05,
        blur_sigma=0.4,
        noise_sigma=1,
        seed=8,
    )

    first = measure(photograph(sample_pdf, 2, soft), pages, texts)
    second = measure(photograph(sample_pdf, 2, crisp), pages, texts)

    assert first.page_index == second.page_index == 2
    assert first.page_width_px == pytest.approx(2200, rel=0.03)
    assert second.page_width_px == pytest.approx(1400, rel=0.03)
    assert first.blur_sigma > second.blur_sigma
    assert first.noise_sigma > second.noise_sigma * 2
    assert first.light_falloff > 0.3 > 0.1 > second.light_falloff
    assert abs(first.skew_degrees) > abs(second.skew_degrees)
    assert suggest(first, soft)["capture_width"] == round(first.page_width_px)
    assert suggest(first, soft)["jpeg_quality"] == soft.jpeg_quality


def test_calibration_registers_a_dotted_print(sample_pdf: bytes) -> None:
    """점 무늬는 같은 모양이 되풀이돼서, 그대로 두면 엉뚱한 자리끼리 맞는다."""
    dotted = protect_bytes(sample_pdf, PROFILE).pdf
    pages, texts = reference_pages(dotted)

    real = measure(
        photograph(dotted, 1, CaptureConditions(capture_width=2000, seed=6)), pages, texts
    )
    model = simulated(dotted, 1, CaptureConditions())

    assert real.page_index == 1
    assert real.page_width_px == pytest.approx(2000, rel=0.03)
    assert model.page_width_px == pytest.approx(1600, rel=0.03)


def test_calibration_refuses_a_photo_of_something_else(sample_pdf: bytes) -> None:
    pages, texts = reference_pages(sample_pdf)
    unrelated = np.full((1200, 900), 240, dtype=np.uint8)
    cv2.putText(unrelated, "hello", (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 4, 0, 8)

    with pytest.raises(ExamShieldError, match="맞추지 못했다"):
        measure(unrelated, pages, texts)
    with pytest.raises(ExamShieldError, match="쪽이 없다"):
        measure(unrelated, pages, texts, None, 9)


def test_calibrate_command_prints_both_columns(
    sample_file: Path, sample_pdf: bytes, tmp_path: Path
) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(jpeg(photograph(sample_pdf, 0, CaptureConditions(capture_width=1800))))
    args = ["calibrate", str(photo), "--printed", str(sample_file)]

    shown = runner.invoke(app, args)
    wrong = runner.invoke(app, [*args, "--profile", "a-decoy"])
    missing = runner.invoke(
        app, ["calibrate", str(tmp_path / "x.jpg"), "--printed", str(sample_file)]
    )
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a picture")
    unreadable = runner.invoke(app, ["calibrate", str(broken), "--printed", str(sample_file)])

    assert shown.exit_code == 0, shown.output
    assert "실제 사진" in shown.stdout and "시뮬레이션" in shown.stdout
    width = int(shown.stdout.split("capture_width:")[1].split()[0])
    assert width == pytest.approx(1800, rel=0.03)
    assert wrong.exit_code == 2 and "시나리오 A" in wrong.stderr
    assert missing.exit_code == 2 and unreadable.exit_code == 2


def plain_pages(count: int = 3) -> bytes:
    """쪽마다 자원을 따로 가진 보통의 문서. 내장 문항 세트 PDF는 쪽들이 자원을 같이 쓴다."""
    doc = pymupdf.open()
    for number in range(count):
        page = doc.new_page()
        page.insert_text((72, 100), f"Question {number + 1}. 2+{number}=?", fontsize=14)
        page.insert_text((72, 140), "(1) one (2) two (3) three (4) four", fontsize=12)
    return save_deterministic(doc)


@pytest.mark.parametrize(
    "size", [(595, 842), (612, 792), (842, 1191), (420, 595), (500.3, 707.7)], ids=str
)
def test_ordinary_documents_pass_and_trace(size: tuple[float, float]) -> None:
    doc = pymupdf.open()
    for number in range(2):
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_textbox(
            pymupdf.Rect(50, 60, size[0] - 50, size[1] - 60),
            f"Question {number + 1}. " + "Which of the following is correct? " * 12,
            fontsize=11,
        )
    pdf = save_deterministic(doc)
    base, made = protect_copies(pdf, PROFILE, SEED, IDS[:4])
    found = list(made)
    rows = [map_row(copy, SEED, "x.pdf") for copy in found]

    result = trace(
        jpeg(photograph(found[3].protected.pdf, 1, CaptureConditions())), ".jpg", pdf, rows, SEED
    )

    assert base.passed and all(copy.check.passed for copy in found), found[0].check.differences
    assert result.found and result.examinee_id == IDS[3], result


@pytest.mark.parametrize("build", [plain_pages, lambda: save_deterministic(blank_pages())])
def test_pages_with_their_own_resources_pass_the_check(build: Any) -> None:
    pdf = build()

    protected, result = protect_and_check(pdf, PROFILE)

    assert result.passed, result.differences
    assert open_pdf_bytes(protected.pdf).page_count == 3


def blank_pages() -> Any:
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    return doc


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_pages_are_refused_instead_of_made_untraceable(rotation: int) -> None:
    doc = open_pdf_bytes(plain_pages())
    doc[1].set_rotation(rotation)

    with pytest.raises(ExamShieldError, match="2쪽이 회전"):
        protect_bytes(save_deterministic(doc), PROFILE)


def test_huge_pages_are_refused() -> None:
    doc = pymupdf.open()
    doc.new_page(width=14400, height=14400)

    with pytest.raises(ExamShieldError, match="너무 크다"):
        protect_bytes(save_deterministic(doc), PROFILE)


def page_resources(doc: Any, page: Any) -> int:
    kind, value = doc.xref_get_key(page.xref, "Resources")
    assert kind == "xref", (kind, value)
    return int(value.split()[0])


def overlay_name(doc: Any, page: Any) -> str:
    return str(doc.xref_stream(page.get_contents()[-1]).split(b"/")[1].split()[0].decode())


def form_instead_of_image(doc: Any) -> None:
    """무늬 그림의 이름이 글자를 그리는 묶음(Form)을 가리키게 한다. 묶음 안에서 진짜 무늬도 그린다."""
    page = doc[0]
    name = overlay_name(doc, page)
    resources = page_resources(doc, page)
    image = int(doc.xref_get_key(resources, f"XObject/{name}")[1].split()[0])
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    form = doc.get_new_xref()
    doc.update_object(
        form,
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 600 850] "
        f"/Matrix [{1 / 600:.10f} 0 0 {1 / 846:.10f} 0 {4 / 846:.10f}] "
        f"/Resources << /XObject << /{name} {image} 0 R >> /Font << /Helv {font} 0 R >> >> >>",
    )
    doc.update_stream(
        form,
        b"BT /Helv 28 Tf 0 g 120 300 Td (THE ANSWER IS 3) Tj ET q 600 0 0 846 0 -4 cm /"
        + name.encode()
        + b" Do Q",
        new=True,
    )
    doc.xref_set_key(resources, f"XObject/{name}", f"{form} 0 R")


def another_image_under_the_name(doc: Any) -> None:
    """쪽은 검은 띠 그림을 그리고, 쓰지 않는 묶음 안에 같은 이름으로 진짜 무늬를 넣어 둔다."""
    page = doc[0]
    name = overlay_name(doc, page)
    resources = page_resources(doc, page)
    genuine = int(doc.xref_get_key(resources, f"XObject/{name}")[1].split()[0])
    bar = np.zeros((100, 100, 2), dtype=np.uint8)
    bar[40:45, :, 1] = 255
    page.insert_image(
        pymupdf.Rect(0, 0, 1, 1),
        pixmap=pymupdf.Pixmap(pymupdf.csGRAY, 100, 100, bar.tobytes(), True),
    )
    added = page.get_contents()[-1]
    fake_name = overlay_name(doc, page)
    fake = int(doc.xref_get_key(resources, f"XObject/{fake_name}")[1].split()[0])
    doc.xref_set_key(
        page.xref,
        "Contents",
        "[" + " ".join(f"{xref} 0 R" for xref in page.get_contents() if xref != added) + "]",
    )
    holder = doc.get_new_xref()
    doc.update_object(
        holder,
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] "
        f"/Resources << /XObject << /{name} {genuine} 0 R >> >> >>",
    )
    doc.update_stream(holder, b"q /" + name.encode() + b" Do Q", new=True)
    doc.xref_set_key(resources, "XObject/Holder", f"{holder} 0 R")
    doc.xref_set_key(resources, f"XObject/{name}", f"{fake} 0 R")


def print_only_note(doc: Any) -> None:
    note = doc[0].add_freetext_annot(pymupdf.Rect(100, 500, 300, 540), "ANSWER 3", fontsize=14)
    note.set_flags(pymupdf.PDF_ANNOT_IS_PRINT | pymupdf.PDF_ANNOT_IS_NO_VIEW)
    note.update()


def hidden_layer(doc: Any) -> None:
    layer = doc.add_ocg("answers", on=False)
    doc[0].insert_text((100, 500), "ANSWER 3", fontsize=14, oc=layer)


def too_many_dots(doc: Any) -> None:
    def fill(samples: Any) -> None:
        samples[6::12, 6::12, 1] = 255

    repaint_overlay(doc, fill)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (form_instead_of_image, "쪽의 그림이 아니다"),
        (another_image_under_the_name, ""),
        (print_only_note, "주석"),
        (hidden_layer, ""),
        (too_many_dots, "점 개수"),
    ],
    ids=["form-xobject", "same-name-image", "print-only-annotation", "hidden-layer", "dense"],
)
def test_check_refuses_content_smuggled_past_the_overlay(change: Any, reason: str) -> None:
    pdf = plain_pages()
    protected = protect_bytes(pdf, PROFILE)
    assert compare(pdf, protected.pdf, protected.expectation).passed

    result = compare(pdf, tampered(protected.pdf, change), protected.expectation)

    assert not result.passed
    assert any(reason in entry for entry in result.differences), result.differences


def test_the_plain_check_also_counts_annotations_and_layers() -> None:
    pdf = plain_pages()

    noted = compare(pdf, tampered(pdf, print_only_note))
    layered = compare(pdf, tampered(pdf, hidden_layer))

    assert any("주석" in entry for entry in noted.differences)
    assert any("OCG" in entry for entry in layered.differences)


def encrypted(pdf: bytes) -> bytes:
    return bytes(
        open_pdf_bytes(pdf).tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="o"
        )
    )


HOSTILE_IMAGES = {
    "empty": b"",
    "text": b"not a picture",
    "one-pixel": encode_png(np.zeros((1, 1), dtype=np.uint8)),
    "sliver": encode_png(np.zeros((1, 30000), dtype=np.uint8)),
    "tall-sliver": encode_png(np.zeros((30000, 1), dtype=np.uint8)),
    "white": encode_png(np.full((900, 700), 255, dtype=np.uint8)),
    "black": encode_png(np.zeros((900, 700), dtype=np.uint8)),
}


@pytest.mark.parametrize("name", sorted(HOSTILE_IMAGES))
def test_odd_pictures_end_in_a_clean_answer(
    name: str, sample_file: Path, tmp_path: Path, table: list[MapRow]
) -> None:
    picture = tmp_path / "leak.png"
    picture.write_bytes(HOSTILE_IMAGES[name])
    seed, mapping = tmp_path / "exam.seed", tmp_path / "map.csv"
    seed.write_bytes(SEED)
    write_map(mapping, table)

    traced = runner.invoke(
        app,
        [
            "trace", str(picture), "--original", str(sample_file),
            "--map-file", str(mapping), "--seed-file", str(seed),
        ],
    )  # fmt: skip
    measured = runner.invoke(app, ["calibrate", str(picture), "--printed", str(sample_file)])

    assert traced.exit_code in (1, 2) and traced.exception.__class__ is SystemExit, traced.output
    assert measured.exit_code == 2 and measured.exception.__class__ is SystemExit, measured.output


def test_unreadable_documents_end_in_a_clean_error(
    sample_file: Path, sample_pdf: bytes, tmp_path: Path, table: list[MapRow]
) -> None:
    seed, mapping = tmp_path / "exam.seed", tmp_path / "map.csv"
    seed.write_bytes(SEED)
    write_map(mapping, table)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(jpeg(photograph(sample_pdf, 0, CaptureConditions())))
    locked, garbage, empty = (
        tmp_path / "locked.pdf",
        tmp_path / "garbage.pdf",
        tmp_path / "empty.pdf",
    )
    locked.write_bytes(encrypted(sample_pdf))
    garbage.write_bytes(b"%PDF-nothing")
    empty.write_bytes(b"")
    keys = ["--map-file", str(mapping), "--seed-file", str(seed)]

    results = [
        runner.invoke(app, ["trace", str(photo), "--original", str(bad), *keys])
        for bad in (locked, garbage, empty)
    ]
    results += [runner.invoke(app, ["trace", str(locked), "--original", str(sample_file), *keys])]
    results += [
        runner.invoke(app, ["calibrate", str(photo), "--printed", str(bad)])
        for bad in (locked, garbage, empty)
    ]

    for result in results:
        assert result.exit_code == 2 and result.exception.__class__ is SystemExit, result.output


def test_a_small_photo_of_a_page_dense_with_hangul_is_still_registered() -> None:
    """글자가 빽빽하면 특징점이 수만 개라, 적게 잘라 쓰면 사진과 원본이 맞지 않는다."""
    pdf = make_pdf(
        "가나다라마바사 아자차카타파하 시험 문항의 내용을 그대로 둔다. " * 60, fontsize=9
    )
    base, made = protect_copies(pdf, PROFILE, SEED, IDS[:3])
    found = list(made)
    rows = [map_row(copy, SEED, "x.pdf") for copy in found]
    photo = photograph(found[1].protected.pdf, 0, CaptureConditions(capture_width=1400))
    pages, texts = reference_pages(found[1].protected.pdf)

    result = trace(jpeg(photo), ".jpg", pdf, rows, SEED)
    measured = measure(photo, pages, texts)

    assert base.passed
    assert result.found and result.examinee_id == IDS[1], result
    assert measured.page_width_px == pytest.approx(1400, rel=0.03)
