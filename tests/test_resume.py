import csv
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from examshield.benchmark import load_items
from examshield.cli import app
from examshield.errors import ModelUnavailableError
from examshield.progress import Progress
from examshield.verify.checkpoint import Checkpoint, Tracked, checkpoint_path
from examshield.verify.models.base import TEXT, ModelReply, PageInput
from examshield.verify.models.mock import MockModel

NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
SEED = b"0123456789abcdef-institution-seed"


@pytest.fixture
def workspace(tmp_path: Path, sample_file: Path) -> dict[str, Path]:
    seed = tmp_path / "exam.seed"
    seed.write_bytes(SEED)
    ids = tmp_path / "ids.csv"
    ids.write_text("".join(f"A{number}\n" for number in range(1, 7)))
    return {
        "source": sample_file,
        "seed": seed,
        "ids": ids,
        "map": tmp_path / "map.csv",
        "out": tmp_path / "out",
    }


def copies(workspace: dict[str, Path], *extra: str) -> tuple[int, str]:
    result = CliRunner().invoke(
        app,
        [
            "protect", str(workspace["source"]), "--out", str(workspace["out"]),
            "--profile", "a-full", "--copies", str(workspace["ids"]),
            "--seed-file", str(workspace["seed"]), "--map-file", str(workspace["map"]),
            "--jobs", "1", *extra,
        ],
    )  # fmt: skip
    return result.exit_code, result.output


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_resume_keeps_finished_copies_and_makes_the_rest(workspace: dict[str, Path]) -> None:
    assert copies(workspace)[0] == 0
    whole = {path.name: path.read_bytes() for path in workspace["out"].glob("exam-*.pdf")}
    table = rows_of(workspace["map"])

    # 끊긴 실행을 흉내 낸다: 뒤의 복사본 둘이 없고, 매핑표에는 앞의 넷만 적혀 있다.
    for name in ("exam-A5", "exam-A6"):
        (workspace["out"] / f"{name}.pdf").unlink()
        (workspace["out"] / f"{name}.job.json").unlink()
    with workspace["map"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table[:4])
    stamps = {name: (workspace["out"] / name).stat().st_mtime_ns for name in whole if "A1" in name}

    plain = copies(workspace)
    code, output = copies(workspace, "--resume")

    assert plain[0] == 2 and "--resume" in plain[1]
    assert code == 0, output
    assert "이미 만든 복사본 4개" in output and "2개를 만든다" in output
    assert {path.name: path.read_bytes() for path in workspace["out"].glob("exam-*.pdf")} == whole
    assert rows_of(workspace["map"]) == table
    assert all(
        (workspace["out"] / name).stat().st_mtime_ns == stamp for name, stamp in stamps.items()
    )


def test_resume_remakes_copies_it_cannot_vouch_for(workspace: dict[str, Path]) -> None:
    assert copies(workspace)[0] == 0
    whole = {path.name: path.read_bytes() for path in workspace["out"].glob("exam-*.pdf")}
    (workspace["out"] / "exam-A2.pdf").write_bytes(b"%PDF-1.7 tampered")
    record = workspace["out"] / "exam-A3.job.json"
    changed = json.loads(record.read_text(encoding="utf-8"))
    changed["input_sha256"] = "0" * 64
    record.write_text(json.dumps(changed), encoding="utf-8")

    code, output = copies(workspace, "--resume")

    assert code == 0, output
    assert "이미 만든 복사본 4개" in output
    assert {path.name: path.read_bytes() for path in workspace["out"].glob("exam-*.pdf")} == whole


def test_resume_with_another_seed_remakes_everything(workspace: dict[str, Path]) -> None:
    assert copies(workspace)[0] == 0
    before = (workspace["out"] / "exam-A1.pdf").read_bytes()
    workspace["seed"].write_bytes(b"another-seed-0123456789abcdef")

    code, output = copies(workspace, "--resume")

    assert code == 0, output
    assert "이미 만든 복사본 0개" in output
    assert (workspace["out"] / "exam-A1.pdf").read_bytes() != before


def test_the_table_grows_as_copies_are_written(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """네 번째 복사본을 쓰다가 죽으면, 앞의 셋은 매핑표에 적혀 있고 이어 할 수 있다."""
    import examshield.protect.batch as cli  # noqa: PLC0415

    real = cli.write_outputs
    written: list[str] = []

    def dies_on_the_fourth(protected: object, out: Path, stem: str) -> Path:
        if len(written) == 3:
            raise KeyboardInterrupt
        written.append(stem)
        return real(protected, out, stem)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "write_outputs", dies_on_the_fourth)
    interrupted = CliRunner().invoke(
        app,
        [
            "protect", str(workspace["source"]), "--out", str(workspace["out"]),
            "--profile", "a-watermark", "--copies", str(workspace["ids"]),
            "--seed-file", str(workspace["seed"]), "--map-file", str(workspace["map"]),
            "--jobs", "1",
        ],
    )  # fmt: skip
    monkeypatch.setattr(cli, "write_outputs", real)

    assert interrupted.exit_code != 0
    assert [row["examinee_id"] for row in rows_of(workspace["map"])] == ["A1", "A2", "A3"]

    resumed = CliRunner().invoke(
        app,
        [
            "protect", str(workspace["source"]), "--out", str(workspace["out"]),
            "--profile", "a-watermark", "--copies", str(workspace["ids"]),
            "--seed-file", str(workspace["seed"]), "--map-file", str(workspace["map"]),
            "--jobs", "1", "--resume",
        ],
    )  # fmt: skip
    assert resumed.exit_code == 0, resumed.output
    assert "이미 만든 복사본 3개" in resumed.output
    assert [row["examinee_id"] for row in rows_of(workspace["map"])] == [
        f"A{n}" for n in range(1, 7)
    ]


def test_resume_notices_a_changed_answer_key(workspace: dict[str, Path], tmp_path: Path) -> None:
    """정답표가 바뀌면 미끼가 달라진다. 예전 정답표로 만든 복사본을 그대로 두면 섞인다."""
    key = tmp_path / "key.csv"
    key.write_text("1,2\n2,3\n3,1\n", encoding="utf-8")
    assert copies(workspace, "--answer-key", str(key))[0] == 0
    first = {path.name: path.read_bytes() for path in workspace["out"].glob("exam-*.pdf")}

    same = copies(workspace, "--answer-key", str(key), "--resume")
    # 빈 줄과 제목 줄만 다른 같은 정답표는 같은 것으로 본다.
    key.write_text("쪽,정답\n1,2\n\n2,3\n3,1\n", encoding="utf-8")
    reformatted = copies(workspace, "--answer-key", str(key), "--resume")
    key.write_text("1,2\n2,4\n3,1\n", encoding="utf-8")
    changed = copies(workspace, "--answer-key", str(key), "--resume")
    after = {path.name: path.read_bytes() for path in workspace["out"].glob("exam-*.pdf")}
    without = copies(workspace, "--resume")
    fresh = workspace["out"].with_name("fresh")
    assert (
        copies(
            {**workspace, "out": fresh, "map": tmp_path / "fresh.csv"}, "--answer-key", str(key)
        )[0]
        == 0
    )

    assert "6개는 그대로 두고 0개" in same[1] and "6개는 그대로 두고 0개" in reformatted[1]
    assert changed[0] == 0 and "0개는 그대로 두고 6개" in changed[1]
    assert "0개는 그대로 두고 6개" in without[1]
    # 다시 만든 것은 새 정답표로 처음부터 만든 것과 같다.
    assert after != first
    assert after == {path.name: path.read_bytes() for path in fresh.glob("exam-*.pdf")}


def test_the_answer_key_mark_in_the_job_record_does_not_give_the_answers_away(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    from examshield.protect.copies import answer_key_tag  # noqa: PLC0415
    from examshield.protect.pipeline import read_answer_key  # noqa: PLC0415

    key = tmp_path / "key.csv"
    key.write_text("1,2\n2,3\n3,1\n", encoding="utf-8")
    copies(workspace, "--answer-key", str(key))
    record = json.loads((workspace["out"] / "exam-A1.job.json").read_text(encoding="utf-8"))
    answers = read_answer_key(key)
    plain = hashlib.sha256(key.read_bytes()).hexdigest()

    # 시드를 열쇠로 쓴 값이다. 시드가 다르면 값이 다르고, 정답표만의 해시와도 다르다.
    assert record["answer_key_tag"] == answer_key_tag(SEED, answers)
    assert record["answer_key_tag"] != answer_key_tag(b"another-seed-0123456789", answers)
    assert record["answer_key_tag"] not in plain and plain not in json.dumps(record)
    assert answer_key_tag(SEED, None) is None
    plain_run = workspace["out"].with_name("plain")
    copies({**workspace, "out": plain_run, "map": tmp_path / "plain.csv"})
    assert "answer_key_tag" not in (plain_run / "exam-A1.job.json").read_text(encoding="utf-8")


def test_resume_is_refused_without_copies(sample_file: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "protect",
            str(sample_file),
            "--out",
            str(tmp_path / "o"),
            "--profile",
            "none",
            "--resume",
        ],
    )

    assert result.exit_code == 2 and "--copies" in result.output


def test_progress_writes_a_line_every_tenth_when_not_on_a_terminal() -> None:
    stream = io.StringIO()
    ticks = iter(range(1000))
    progress = Progress(40, "복사본", stream=stream, clock=lambda: float(next(ticks)))

    for _ in range(40):
        progress.step()
    progress.close()
    lines = stream.getvalue().splitlines()

    assert len(lines) == 10
    assert lines[0].startswith("복사본 4/40 (10%), 남은 시간 약 ")
    assert lines[-1] == "복사본 40/40 (100%)"


def test_progress_says_nothing_for_a_single_step_and_counts_what_was_already_done() -> None:
    quiet = io.StringIO()
    Progress(1, "복사본", stream=quiet).step()
    resumed = io.StringIO()
    progress = Progress(10, "복사본", stream=resumed, clock=lambda: 0.0, already=8)
    progress.step()
    progress.step()

    assert quiet.getvalue() == ""
    lines = resumed.getvalue().splitlines()
    assert lines[0].startswith("복사본 9/10 (90%)") and lines[1] == "복사본 10/10 (100%)"


class Flaky:
    """정해진 호출에서 실패하는 모델. 호출한 횟수를 센다."""

    def __init__(self, fail_on: set[int]) -> None:
        self.inner = MockModel(load_items().items)
        self.info = self.inner.info
        self.fail_on = fail_on
        self.calls = 0

    def run(self, page: PageInput, channel: str) -> ModelReply:
        self.calls += 1
        if self.calls in self.fail_on:
            raise ModelUnavailableError("호출 한도에 걸렸다.")
        return self.inner.run(page, channel)


def pages() -> list[PageInput]:
    return [
        PageInput(text=item.text, image_png=b"", choice_count=len(item.choices))
        for item in load_items().items[:5]
    ]


def test_answers_already_received_are_not_asked_for_again(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "a-decoy")
    first = Flaky(fail_on={3})
    store = Checkpoint(path, NOW, resume=False)
    replies = []
    for page in pages():
        try:
            replies.append(Tracked(first, store).run(page, TEXT))
        except ModelUnavailableError:
            replies.append(None)

    second = Flaky(fail_on=set())
    resumed = Checkpoint(path, NOW + timedelta(minutes=5), resume=True)
    received = len(resumed)
    again = [Tracked(second, resumed).run(page, TEXT) for page in pages()]

    assert first.calls == 5 and replies[2] is None
    assert received == 4
    # 실패했던 호출 하나만 다시 했다.
    assert second.calls == 1 and resumed.reused == 4
    assert [reply.answer for reply in again] == [item.answer for item in load_items().items[:5]]
    # 다시 쓴 응답은 처음 받을 때 걸린 시간을 함께 전한다.
    assert again[0].seconds is not None
    assert replace(again[0], seconds=None) == replies[0]


def test_old_answers_and_a_fresh_start_are_not_reused(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "a-decoy")
    model = Flaky(fail_on=set())
    store = Checkpoint(path, NOW, resume=False)
    for page in pages():
        Tracked(model, store).run(page, TEXT)
    path.write_text(path.read_text(encoding="utf-8") + '{"key": "broken', encoding="utf-8")

    assert len(Checkpoint(path, NOW + timedelta(hours=23), resume=True)) == 5
    assert len(Checkpoint(path, NOW + timedelta(hours=25), resume=True)) == 0
    assert len(Checkpoint(path, NOW - timedelta(hours=1), resume=True)) == 0
    assert len(Checkpoint(path, NOW, resume=False)) == 0 and not path.exists()


def test_cli_verify_leaves_no_checkpoint_after_a_clean_run_and_refuses_resume_with_a_custom_file(
    tmp_path: Path, sample_file: Path
) -> None:
    base = ["verify", "--profile", "a-remap-decoy", "--models", "mock", "--out", str(tmp_path)]

    clean = CliRunner().invoke(app, base)
    resumed = CliRunner().invoke(app, [*base, "--resume"])
    custom = CliRunner().invoke(app, [*base, "--allow-custom", str(sample_file), "--resume"])
    plain_custom = CliRunner().invoke(app, [*base, "--allow-custom", str(sample_file)])

    assert clean.exit_code == 0, clean.output
    assert resumed.exit_code == 0 and "받아 둔 응답 0건" in resumed.output
    assert custom.exit_code == 2 and "--resume" in custom.output
    assert plain_custom.exit_code == 0, plain_custom.output
    assert not list(tmp_path.glob("*.checkpoint.jsonl"))


def test_cli_verify_resume_calls_only_what_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import examshield.cli as cli  # noqa: PLC0415

    made: list[Flaky] = []

    def flaky_mock(items: object) -> Flaky:
        made.append(Flaky(fail_on={2, 5} if not made else set()))
        return made[-1]

    monkeypatch.setattr(cli, "MockModel", flaky_mock)
    base = ["verify", "--profile", "none", "--models", "mock", "--out", str(tmp_path)]

    first = CliRunner().invoke(app, base)
    kept = checkpoint_path(tmp_path, "none")
    assert first.exit_code == 0, first.output
    assert "오류가 난 호출이 2건" in first.output and kept.exists()

    second = CliRunner().invoke(app, [*base, "--resume"])
    reports = sorted(tmp_path.glob("verify-none-*.json"))
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    row = next(entry for entry in report["results"] if entry["model"] == "mock-reader")

    assert second.exit_code == 0, second.output
    assert made[0].calls == 12 and made[1].calls == 2
    assert row["errors"] == 0 and row["accuracy"] == 1.0
    assert any("받아 둔 응답" in note for note in report["notes"])
    assert not kept.exists()
