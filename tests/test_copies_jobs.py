import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.profiles import load_profile
from examshield.protect.copies import choose_jobs, protect_copies

SEED = b"0123456789abcdef-institution-seed"
IDS = [f"S{number:04d}" for number in range(9)]


def test_copies_made_in_several_processes_are_the_same_bytes_in_the_same_order(
    sample_pdf: bytes,
) -> None:
    profile = load_profile("a-full")

    _, one_by_one = protect_copies(sample_pdf, profile, SEED, IDS, jobs=1)
    _, together = protect_copies(sample_pdf, profile, SEED, IDS, jobs=3)
    first = list(one_by_one)
    second = list(together)

    assert [copy.examinee_id for copy in second] == IDS
    assert [copy.protected.pdf for copy in second] == [copy.protected.pdf for copy in first]
    assert [copy.protected.record for copy in second] == [copy.protected.record for copy in first]
    assert all(copy.check.passed for copy in second)


def test_stopping_early_does_not_wait_for_the_rest(sample_pdf: bytes) -> None:
    ids = [f"S{number:04d}" for number in range(60)]
    _, made = protect_copies(sample_pdf, load_profile("a-watermark"), SEED, ids, jobs=2)

    started = time.perf_counter()
    first = next(made)
    made.close()

    assert first.examinee_id == "S0000"
    # 남은 쉰여 개를 다 만들었다면 훨씬 오래 걸린다.
    assert time.perf_counter() - started < 20


def test_job_count_is_chosen_from_the_number_of_copies() -> None:
    assert choose_jobs(None, 3) == 1
    assert choose_jobs(None, 1000) >= 1
    assert choose_jobs(4, 2) == 2
    assert choose_jobs(1, 500) == 1
    with pytest.raises(ExamShieldError, match="--jobs"):
        choose_jobs(0, 10)


def test_cli_takes_jobs_only_with_copies(sample_file: Path, tmp_path: Path) -> None:
    seed = tmp_path / "exam.seed"
    seed.write_bytes(SEED)
    ids = tmp_path / "ids.csv"
    ids.write_text("A1\nA2\nA3\n")
    base = ["protect", str(sample_file), "--out", str(tmp_path / "out")]
    copies = [
        *base, "--profile", "a-watermark", "--copies", str(ids), "--seed-file", str(seed),
        "--map-file", str(tmp_path / "map.csv"),
    ]  # fmt: skip

    alone = CliRunner().invoke(app, [*base, "--profile", "a-remap-decoy", "--jobs", "2"])
    zero = CliRunner().invoke(app, [*copies, "--jobs", "0"])
    written_before = list((tmp_path / "out").glob("*.pdf"))
    made = CliRunner().invoke(app, [*copies, "--jobs", "2"])

    assert alone.exit_code == 2 and "--copies" in alone.output
    assert zero.exit_code == 2 and written_before == []
    assert made.exit_code == 0, made.output
    assert len(list((tmp_path / "out").glob("exam-A*.pdf"))) == 3
