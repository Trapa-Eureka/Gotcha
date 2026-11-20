import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import Profile, load_profile, parse_profile
from examshield.protect.pipeline import protect_bytes
from examshield.protect.techniques.decoy import choice_count
from examshield.verify.capture_sim import encode_png
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


def test_misplaced_and_unknown_techniques_are_refused(sample_pdf: bytes) -> None:
    def profile(scenario: str, *ids: str) -> Profile:
        return parse_profile(
            {
                "name": "x",
                "scenario": scenario,
                "status": "experimental",
                "techniques": [{"id": name} for name in ids],
            }
        )

    with pytest.raises(ExamShieldError, match="시나리오 A"):
        protect_bytes(sample_pdf, profile("A", "dots"))
    with pytest.raises(ExamShieldError, match="시나리오 B"):
        protect_bytes(sample_pdf, profile("B", "decoy"))
    with pytest.raises(ExamShieldError, match="맨 마지막"):
        protect_bytes(sample_pdf, profile("B", "dots", "dots"))
    with pytest.raises(ExamShieldError, match="모르는 기법"):
        protect_bytes(sample_pdf, profile("A", "hologram"))
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
    text = tmp_path / "exam.txt"
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


@pytest.mark.parametrize("suffix", [".hwp", ".hwpx", ".HWP"])
def test_hangul_input_goes_through_the_converter_with_a_review_notice(
    suffix: str, sample_pdf: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / f"exam{suffix}").write_bytes(b"fake hangul file")
    canned = tmp_path / "canned.pdf"
    canned.write_bytes(sample_pdf)
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = tools / "soffice"
    fake.write_text(
        "#!/bin/bash\n"
        'while [ $# -gt 0 ]; do [ "$1" = "--outdir" ] && out="$2"; last="$1"; shift; done\n'
        f'echo "$last" > "{tmp_path}/source.txt"\n'
        f'cp "{canned}" "$out/input.pdf"\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    out = tmp_path / "out"

    code, output = run_protect(source / f"exam{suffix}", out, "a-decoy")
    checked = CliRunner().invoke(
        app, ["check", str(source / f"exam{suffix}"), str(out / "exam.pdf")]
    )

    assert code == 0, output
    assert "한글 문서는 LibreOffice가 PDF로 바꾼 결과를 원본으로 삼는다" in output
    assert (tmp_path / "source.txt").read_text().strip().endswith(f"input{suffix.lower()}")
    assert sorted(path.name for path in out.iterdir()) == [
        "exam.job.json",
        "exam.pdf",
        "protect.log.jsonl",
    ]
    assert [path.name for path in source.iterdir()] == [f"exam{suffix}"]
    assert checked.exit_code == 0, checked.output


def test_hangul_input_that_libreoffice_cannot_open_points_to_pdf_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "exam.hwp"
    source.write_bytes(b"fake")
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = tools / "soffice"
    fake.write_text("#!/bin/bash\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))

    failed = run_protect(source, tmp_path / "out", "a-decoy")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    missing = run_protect(source, tmp_path / "out2", "a-decoy")

    assert failed[0] == 2 and "한글에서 PDF로 내보낸 파일" in failed[1]
    assert missing[0] == 2 and "LibreOffice가 필요하다" in missing[1]


def test_names_that_differ_only_in_case_are_refused_in_a_folder(
    sample_pdf: bytes, tmp_path: Path
) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "exam.pdf").write_bytes(sample_pdf)
    (source / "EXAM.docx").write_bytes(b"fake")
    if len(list(source.iterdir())) != 2:
        pytest.skip("대소문자를 구분하지 않는 파일 시스템이라 두 파일을 함께 둘 수 없다")

    code, output = run_protect(source, tmp_path / "out", "none")

    assert code == 2 and "이름이 같은 파일" in output
    assert not (tmp_path / "out").exists()


def test_a_file_that_is_not_really_a_pdf_ends_in_a_clean_error(
    sample_pdf: bytes, tmp_path: Path
) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(encode_png(np.zeros((40, 40), dtype=np.uint8)))
    locked = tmp_path / "locked.pdf"
    locked.write_bytes(
        open_pdf_bytes(sample_pdf).tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="p")
    )
    cut = tmp_path / "cut.pdf"
    cut.write_bytes(sample_pdf[: len(sample_pdf) // 2])
    runner = CliRunner()

    results = [
        runner.invoke(
            app, ["protect", str(path), "--out", str(tmp_path / "o"), "--profile", "a-decoy"]
        )
        for path in (fake, locked, cut)
    ]
    results += [runner.invoke(app, ["check", str(path), str(path)]) for path in (fake, locked)]

    for result in results:
        assert result.exit_code == 2 and result.exception.__class__ is SystemExit, result.output
        assert result.stderr.startswith("오류:")


def test_license_command_names_the_license_and_the_source() -> None:
    result = CliRunner().invoke(app, ["license"])

    assert result.exit_code == 0
    assert (
        "AGPL-3.0-or-later" in result.stdout and "github.com/Trapa-Eureka/Gotcha" in result.stdout
    )
    assert "PyMuPDF" in result.stdout


def answer_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "answers.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_decoys_avoid_the_answers_in_the_key(sample_pdf: bytes) -> None:
    profile = load_profile("a-remap-decoy")
    plain = protect_bytes(sample_pdf, profile)
    assert plain.decoy_answers is not None
    # 정답표 없이 고른 미끼 번호를 그대로 정답으로 삼는다. 정답표를 주면 전부 달라져야 한다.
    key = {page: frozenset({answer}) for page, answer in enumerate(plain.decoy_answers, start=1)}

    keyed = protect_bytes(sample_pdf, profile, None, None, key)
    again = protect_bytes(sample_pdf, profile, None, None, key)

    assert keyed.decoy_answers is not None
    assert all(a != b for a, b in zip(keyed.decoy_answers, plain.decoy_answers, strict=True))
    assert all(1 <= answer <= 4 for answer in keyed.decoy_answers)
    assert keyed.pdf == again.pdf
    assert keyed.record["answer_key"] is True and plain.record["answer_key"] is False


def test_without_a_key_the_copy_is_what_it_always_was(sample_pdf: bytes) -> None:
    profile = load_profile("a-decoy")

    assert (
        protect_bytes(sample_pdf, profile).pdf
        == protect_bytes(sample_pdf, profile, None, None, None).pdf
    )
    assert protect_bytes(sample_pdf, profile, None, None, {}).decoy_answers == (
        protect_bytes(sample_pdf, profile).decoy_answers
    )


def test_a_page_where_every_choice_is_an_answer_is_counted(sample_pdf: bytes) -> None:
    key = {1: frozenset({1, 2, 3, 4}), 2: frozenset({1, 2, 3})}

    protected = protect_bytes(sample_pdf, load_profile("a-decoy"), None, None, key)
    stats = protected.record["techniques"][0]["stats"]

    assert stats["pages_where_every_choice_is_an_answer"] == 1
    assert protected.decoy_answers is not None and protected.decoy_answers[1] == 4


def test_the_answers_never_reach_the_job_record_or_the_log(
    sample_file: Path, tmp_path: Path
) -> None:
    key = answer_file(tmp_path, "쪽,정답\n1,3\n2,1\n3,2\n")
    out = tmp_path / "out"

    code, output = run_protect(sample_file, out, "a-remap-decoy", "--answer-key", str(key))
    record = json.loads((out / "exam.job.json").read_text(encoding="utf-8"))

    assert code == 0, output
    assert record["answer_key"] is True
    assert record["decoy_answers"][0] != 3 and record["decoy_answers"][1] != 1
    assert record["decoy_answers"][2] != 2
    written = "".join(path.read_text(encoding="utf-8") for path in out.glob("*.json*"))
    assert "answers.csv" not in written and '"3,2"' not in written


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "비어 있다"),
        ("쪽,정답\n", "비어 있다"),
        ("1,x\n", "형식"),
        ("1\n", "형식"),
        ("0,1\n", "1 이상"),
        ("1,0\n", "1~9"),
        ("1,10\n", "1~9"),
        ("9,1\n", "문서에 없는 쪽"),
    ],
)
def test_bad_answer_keys_are_refused(
    text: str, message: str, sample_file: Path, tmp_path: Path
) -> None:
    key = answer_file(tmp_path, text)

    code, output = run_protect(sample_file, tmp_path / "out", "a-decoy", "--answer-key", str(key))

    assert code == 2 and message in output
    assert not (tmp_path / "out" / "exam.pdf").exists()


def test_answer_key_needs_a_decoy_profile_and_must_stay_out_of_the_output(
    sample_file: Path, tmp_path: Path
) -> None:
    key = answer_file(tmp_path, "1,3\n")
    out = tmp_path / "out"
    out.mkdir()
    inside = out / "answers.csv"
    inside.write_text("1,3\n", encoding="utf-8")

    no_decoy = run_protect(sample_file, tmp_path / "o1", "a-glyph-remap", "--answer-key", str(key))
    leaked = run_protect(sample_file, out, "a-decoy", "--answer-key", str(inside))
    missing = run_protect(
        sample_file, tmp_path / "o2", "a-decoy", "--answer-key", str(tmp_path / "x")
    )
    (sample_file.parent / "second.pdf").write_bytes(sample_file.read_bytes())
    folder = run_protect(sample_file.parent, tmp_path / "o3", "a-decoy", "--answer-key", str(key))

    assert no_decoy[0] == 2 and "미끼가 없어" in no_decoy[1]
    assert leaked[0] == 2 and "출력 폴더 안" in leaked[1]
    assert missing[0] == 2 and "읽지 못했다" in missing[1]
    assert folder[0] == 2 and "파일 하나" in folder[1]
