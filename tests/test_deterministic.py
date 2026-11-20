import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from examshield.profiles import load_profile
from examshield.protect.pipeline import protect_and_check, protect_bytes
from examshield.protect.seed import SeedStream, derive_seed
from examshield.protect.techniques.decoy import subset_font
from tests.conftest import A_PROFILES


@pytest.mark.parametrize("profile", ("none", *A_PROFILES))
def test_same_input_profile_and_seed_give_the_same_bytes(profile: str, sample_pdf: bytes) -> None:
    loaded = load_profile(profile)
    first, _ = protect_and_check(sample_pdf, loaded, b"seed-1")
    second, _ = protect_and_check(sample_pdf, loaded, b"seed-1")

    assert first.pdf == second.pdf
    assert first.record == second.record


@pytest.mark.parametrize("profile", ("a-glyph-remap", "a-decoy", "a-remap-decoy"))
def test_a_different_seed_gives_a_different_copy(profile: str, sample_pdf: bytes) -> None:
    loaded = load_profile(profile)

    assert (
        protect_bytes(sample_pdf, loaded, b"seed-1").pdf
        != protect_bytes(sample_pdf, loaded, b"seed-2").pdf
    )


def test_the_embedded_decoy_font_does_not_carry_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    first = subset_font("정답은 3번")
    monkeypatch.setattr(time, "time", lambda: 4_000_000_000.0)

    assert subset_font("정답은 3번") == first


def test_no_seed_file_still_gives_a_fixed_result(sample_pdf: bytes) -> None:
    loaded = load_profile("a-remap-decoy")

    assert protect_bytes(sample_pdf, loaded).pdf == protect_bytes(sample_pdf, loaded).pdf


@pytest.mark.parametrize("hash_seed", ("0", "12345"))
def test_output_is_the_same_in_a_fresh_process(
    hash_seed: str, sample_file: Path, sample_pdf: bytes, tmp_path: Path
) -> None:
    seed_file = tmp_path / "exam.seed"
    seed_file.write_bytes(b"seed-1")
    out = tmp_path / f"out-{hash_seed}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examshield",
            "protect",
            str(sample_file),
            "--out",
            str(out),
            "--profile",
            "a-remap-decoy",
            "--seed-file",
            str(seed_file),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    expected = protect_bytes(sample_pdf, load_profile("a-remap-decoy"), b"seed-1").pdf

    assert (
        hashlib.sha256((out / "exam.pdf").read_bytes()).hexdigest()
        == hashlib.sha256(expected).hexdigest()
    )


def test_seed_stream_values_are_pinned() -> None:
    stream = SeedStream(derive_seed("a", "b", b"c"), "label")

    assert [stream.below(10) for _ in range(8)] == PINNED_VALUES
    assert SeedStream(b"k", "x").shuffled(list(range(6))) == PINNED_SHUFFLE


PINNED_VALUES = [5, 6, 2, 3, 9, 2, 4, 7]
PINNED_SHUFFLE = [1, 2, 5, 4, 0, 3]
