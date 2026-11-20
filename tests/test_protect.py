import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import load_profile, parse_profile
from examshield.protect.pipeline import protect_bytes
from examshield.protect.techniques.decoy import choice_count
from examshield.verify.metrics import parse_answer
from tests.conftest import A_PROFILES

runner = CliRunner()


def run_protect(source: Path, out: Path, profile: str, *extra: str) -> tuple[int, str]:
    result = runner.invoke(
        app, ["protect", str(source), "--out", str(out), "--profile", profile, *extra]
    )
    return result.exit_code, result.output


def test_rasterize_removes_the_text_layer_and_keeps_the_page_size(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-rasterize"))
    before, after = open_pdf_bytes(sample_pdf), open_pdf_bytes(protected.pdf)

    assert [page_text(page).strip() for page in after] == ["", "", ""]
    assert [tuple(page.rect) for page in after] == [tuple(page.rect) for page in before]


def test_decoy_is_extracted_first_and_points_at_the_recorded_answer(sample_pdf: bytes) -> None:
    protected = protect_bytes(sample_pdf, load_profile("a-decoy"))
    assert protected.decoy_answers is not None

    for page, answer in zip(open_pdf_bytes(protected.pdf), protected.decoy_answers, strict=True):
        first_line = page_text(page).splitlines()[0]
        assert parse_answer(first_line, 4) == answer
        assert 1 <= answer <= 4


def test_glyph_remap_after_decoy_is_refused(sample_pdf: bytes) -> None:
    profile = parse_profile(
        {
            "name": "x",
            "scenario": "A",
            "status": "experimental",
            "techniques": [{"id": "decoy"}, {"id": "glyph_remap"}],
        }
    )

    with pytest.raises(ExamShieldError, match="먼저"):
        protect_bytes(sample_pdf, profile)


def test_choice_count_comes_from_the_page_or_falls_back() -> None:
    assert choice_count("① 가 ② 나 ③ 다 ④ 라", 5) == 4
    assert choice_count("서술형 문항", 5) == 5


def test_scenario_b_and_unknown_techniques_are_refused(sample_pdf: bytes) -> None:
    unknown = parse_profile(
        {"name": "x", "scenario": "A", "status": "experimental", "techniques": [{"id": "dots"}]}
    )

    with pytest.raises(ExamShieldError, match="시나리오 A"):
        protect_bytes(sample_pdf, load_profile("none-print"))
    with pytest.raises(ExamShieldError, match="dots"):
        protect_bytes(sample_pdf, unknown)
    with pytest.raises(ExamShieldError):
        protect_bytes(b"not a pdf", load_profile("none"))


@pytest.mark.parametrize("profile", A_PROFILES)
def test_cli_writes_the_copy_the_job_record_and_a_log(
    profile: str, sample_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    code, output = run_protect(sample_file, out, profile)
    record = json.loads((out / "exam.job.json").read_text(encoding="utf-8"))

    assert code == 0, output
    assert sorted(path.name for path in out.iterdir()) == [
        "exam.job.json",
        "exam.pdf",
        "protect.log.jsonl",
    ]
    assert record["output_sha256"] == hashlib.sha256((out / "exam.pdf").read_bytes()).hexdigest()
    assert record["check"]["passed"] is True
    assert record["profile"]["name"] == profile


def test_the_original_is_not_modified_and_can_be_read_only(
    sample_file: Path, tmp_path: Path
) -> None:
    before = sample_file.read_bytes()
    sample_file.chmod(stat.S_IRUSR)
    sample_file.parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        code, output = run_protect(sample_file, tmp_path / "out", "a-remap-decoy")
    finally:
        sample_file.parent.chmod(stat.S_IRWXU)

    assert code == 0, output
    assert sample_file.read_bytes() == before
    assert [path.name for path in sample_file.parent.iterdir()] == ["exam.pdf"]


def test_writing_into_the_folder_of_the_original_is_refused(sample_file: Path) -> None:
    code, output = run_protect(sample_file, sample_file.parent, "a-decoy")

    assert code == 2
    assert [path.name for path in sample_file.parent.iterdir()] == ["exam.pdf"]


def test_existing_output_is_kept_unless_forced(sample_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "exam.pdf").write_bytes(b"someone else's file")

    assert run_protect(sample_file, out, "a-decoy")[0] == 2
    assert (out / "exam.pdf").read_bytes() == b"someone else's file"
    assert run_protect(sample_file, out, "a-decoy", "--force")[0] == 0


def test_log_and_job_record_hold_no_question_text(sample_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_protect(sample_file, out, "a-remap-decoy")
    written = (out / "protect.log.jsonl").read_text(encoding="utf-8")
    line = json.loads(written.splitlines()[0])
    job = (out / "exam.job.json").read_text(encoding="utf-8")

    assert set(line) == {"time", "profile", "input_sha256", "output_sha256", "check_passed"}
    for word in ("끓는", "수도", "삼각형"):
        assert word not in written
        assert word not in job


def test_no_temporary_files_are_left_behind(sample_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_protect(sample_file, out, "a-rasterize")

    assert not list(out.glob(".examshield-tmp*"))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["in", "out"]


def test_a_folder_is_processed_file_by_file(
    sample_file: Path, sample_pdf: bytes, tmp_path: Path
) -> None:
    (sample_file.parent / "second.pdf").write_bytes(sample_pdf)
    (sample_file.parent / "notes.txt").write_text("skip me")
    out = tmp_path / "out"
    code, output = run_protect(sample_file.parent, out, "a-decoy")

    assert code == 0, output
    assert {path.name for path in out.glob("*.pdf")} == {"exam.pdf", "second.pdf"}
    assert len((out / "protect.log.jsonl").read_text().splitlines()) == 2


def test_seed_file_changes_the_copy(sample_file: Path, tmp_path: Path) -> None:
    seed = tmp_path / "a.seed"
    seed.write_bytes(b"one")
    run_protect(sample_file, tmp_path / "o1", "a-decoy")
    run_protect(sample_file, tmp_path / "o2", "a-decoy", "--seed-file", str(seed))
    record = json.loads((tmp_path / "o2" / "exam.job.json").read_text(encoding="utf-8"))

    assert (tmp_path / "o1" / "exam.pdf").read_bytes() != (
        tmp_path / "o2" / "exam.pdf"
    ).read_bytes()
    assert record["seed"] == "file"
    assert "one" not in json.dumps(record)
    assert run_protect(sample_file, tmp_path / "o3", "a-decoy", "--seed-file", "nope")[0] == 2


def test_unsupported_and_missing_inputs_exit_with_code_2(tmp_path: Path) -> None:
    text = tmp_path / "exam.hwp"
    text.write_text("x")

    assert run_protect(text, tmp_path / "out", "a-decoy")[0] == 2
    assert run_protect(tmp_path / "missing.pdf", tmp_path / "out", "a-decoy")[0] == 2
    assert run_protect(text, tmp_path / "out", "no-such-profile")[0] == 2


def test_word_input_goes_through_a_local_converter_inside_the_work_folder(
    sample_pdf: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "exam.docx").write_bytes(b"fake word file")
    canned = tmp_path / "canned.pdf"
    canned.write_bytes(sample_pdf)
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = tools / "soffice"
    fake.write_text(
        "#!/bin/bash\n"
        'while [ $# -gt 0 ]; do [ "$1" = "--outdir" ] && out="$2"; shift; done\n'
        f'pwd > "{tmp_path}/cwd.txt"; echo "$out" > "{tmp_path}/outdir.txt"\n'
        f'cp "{canned}" "$out/input.pdf"\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    out = tmp_path / "out"
    code, output = run_protect(source / "exam.docx", out, "a-decoy")

    assert code == 0, output
    assert (tmp_path / "outdir.txt").read_text().strip().startswith(str(out / ".examshield-tmp-"))
    assert sorted(path.name for path in out.iterdir()) == [
        "exam.job.json",
        "exam.pdf",
        "protect.log.jsonl",
    ]
    assert [path.name for path in source.iterdir()] == ["exam.docx"]


def test_word_input_without_libreoffice_explains_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "exam.docx"
    source.write_bytes(b"fake")
    monkeypatch.setenv("PATH", str(tmp_path))
    code, output = run_protect(source, tmp_path / "out", "a-decoy")

    assert code == 2
    assert "LibreOffice" in output


def test_the_same_folder_under_another_spelling_is_still_refused(sample_file: Path) -> None:
    before = sample_file.read_bytes()
    alias = sample_file.parent.parent / "IN"
    if not alias.exists():
        pytest.skip("대소문자를 구분하는 파일 시스템이다")
    code, _ = run_protect(sample_file, alias, "a-decoy", "--force")

    assert code == 2
    assert sample_file.read_bytes() == before


def test_out_pointing_at_a_file_exits_with_code_2(sample_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")

    assert run_protect(sample_file, target, "a-decoy")[0] == 2
    assert target.read_text() == "x"


def test_a_folder_the_user_already_had_is_not_deleted(sample_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    keep = out / ".examshield-tmp" / "mine.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep")
    run_protect(sample_file, out, "a-decoy")
    runner.invoke(app, ["check", str(sample_file), str(out / "exam.pdf")])

    assert keep.read_text() == "keep"


def test_rasterize_refuses_a_resolution_that_would_blur_small_text(sample_pdf: bytes) -> None:
    low = parse_profile(
        {
            "name": "x",
            "scenario": "A",
            "status": "experimental",
            "techniques": [{"id": "rasterize", "params": {"dpi": 72}}],
        }
    )

    with pytest.raises(ExamShieldError, match="dpi"):
        protect_bytes(sample_pdf, low)
