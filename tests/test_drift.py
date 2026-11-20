import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from examshield import __version__
from examshield.benchmark import load_items
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.profiles import Profile, load_profile, parse_profile
from examshield.verify.drift import (
    FAILED,
    NOT_APPLICABLE,
    PASSED,
    UNVERIFIED,
    Current,
    assess,
    find_latest,
    format_assessment,
    model_verdicts,
    record_refusal,
    write_record,
)
from examshield.verify.models.base import TEXT, ModelInfo
from examshield.verify.models.baseline import TextLayerModel
from examshield.verify.models.mock import MockModel
from examshield.verify.runner import run_verification

NOW = datetime(2026, 11, 20, 9, 0, 0, tzinfo=UTC)
ITEMS = load_items()
CURRENT = Current(__version__, ITEMS.version, ITEMS.sha256)
runner = CliRunner()


def row(model: str, channel: str, variant: str, accuracy: float | None) -> dict[str, Any]:
    return {
        "model": model,
        "channel": channel,
        "variant": variant,
        "items": 12,
        "errors": 0,
        "refusals": 0,
        "legibility": 0.5,
        "accuracy": accuracy,
        "decoy_hit_rate": None,
        "decoy_items": None,
    }


def make_report(
    profile: Profile,
    *,
    days_ago: int = 0,
    kind: str = "cloud",
    control: float = 1.0,
    protected: float = 0.25,
    **changes: Any,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": 1,
        "created_at": (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "profile": {
            "name": profile.name,
            "sha256": profile.sha256,
            "scenario": profile.scenario,
            "status": profile.status,
        },
        "benchmark": {"version": ITEMS.version, "sha256": ITEMS.sha256, "item_count": 12},
        "simulation": None,
        "protection": None,
        "models": [
            {"provider": "x", "name": "reader", "version": "1", "kind": kind, "note": ""},
            {"provider": "x", "name": "text-layer", "version": "1", "kind": "baseline", "note": ""},
        ],
        "results": [
            row("reader", "text", "control", control),
            row("reader", "text", "protected", protected),
            row("text-layer", "text", "control", None),
        ],
        "skipped": [],
        "notes": [],
    }
    report.update(changes)
    return report


def save(folder: Path, name: str, report: Any) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


PROFILE = load_profile("a-remap-decoy")


def test_a_profile_nobody_verified_is_reported_as_unverified() -> None:
    result = assess(PROFILE, None, NOW, CURRENT)

    assert result.state == UNVERIFIED
    assert any("검증한 기록이 없다" in warning for warning in result.warnings)
    assert any("verify --profile a-remap-decoy" in warning for warning in result.warnings)


def test_a_fresh_passing_report_raises_no_drift_warning() -> None:
    result = assess(PROFILE, make_report(PROFILE, days_ago=3), NOW, CURRENT)

    assert result.state == PASSED
    assert result.age_days == 3
    # experimental 상태 경고 하나만 남는다.
    assert len(result.warnings) == 1 and "experimental" in result.warnings[0]


@pytest.mark.parametrize(("days", "stale"), [(30, False), (31, True), (400, True)])
def test_reports_older_than_thirty_days_are_flagged(days: int, stale: bool) -> None:
    result = assess(PROFILE, make_report(PROFILE, days_ago=days), NOW, CURRENT)

    assert any("30일이 넘었" in warning for warning in result.warnings) is stale
    assert ("오래됨" in result.label) is stale


def test_a_report_dated_in_the_future_is_not_trusted(tmp_path: Path) -> None:
    save(tmp_path, "future.json", make_report(PROFILE, days_ago=-400))
    save(tmp_path, "skew.json", make_report(PROFILE, days_ago=0, protected=0.9))

    report, source, _ = find_latest(PROFILE, [tmp_path], NOW)

    assert source == tmp_path / "skew.json"
    assert assess(PROFILE, report, NOW, CURRENT).state == FAILED
    with pytest.raises(ValueError, match="미래"):
        assess(PROFILE, make_report(PROFILE, days_ago=-400), NOW, CURRENT)


@pytest.mark.parametrize(
    ("protected", "state"), [(0.0, PASSED), (0.5, PASSED), (0.51, FAILED), (1.0, FAILED)]
)
def test_a_model_is_held_only_when_its_accuracy_halves(protected: float, state: str) -> None:
    result = assess(PROFILE, make_report(PROFILE, protected=protected), NOW, CURRENT)

    assert result.state == state
    if state == FAILED:
        assert any(
            "기준에 못 미쳤다" in warning and "reader" in warning for warning in result.warnings
        )
        assert any("deprecated로 표시" in warning for warning in result.warnings)


def test_the_best_channel_for_the_attacker_decides() -> None:
    report = make_report(PROFILE, protected=0.0)
    report["results"] += [
        row("reader", "image", "control", 0.9),
        row("reader", "image", "protected", 0.9),
    ]

    verdict = model_verdicts(report)[0]

    assert (verdict.control, verdict.protected, verdict.held) == (1.0, 0.9, False)
    assert assess(PROFILE, report, NOW, CURRENT).state == FAILED


def test_one_broken_model_fails_the_profile_even_if_another_holds() -> None:
    report = make_report(PROFILE)
    report["models"].append(
        {"provider": "y", "name": "strong", "version": "2", "kind": "cloud", "note": ""}
    )
    report["results"] += [
        row("strong", "image", "control", 1.0),
        row("strong", "image", "protected", 1.0),
    ]

    result = assess(PROFILE, report, NOW, CURRENT)

    assert result.state == FAILED
    assert [verdict.held for verdict in result.models] == [True, False]


def test_a_model_that_cannot_solve_the_control_gives_no_verdict() -> None:
    result = assess(PROFILE, make_report(PROFILE, control=0.0, protected=0.0), NOW, CURRENT)

    assert result.state == UNVERIFIED
    assert any("대조군 문항을 풀지 못했다" in warning for warning in result.warnings)


def test_tool_and_benchmark_changes_are_flagged() -> None:
    report = make_report(PROFILE, tool_version="0.0.1")
    report["benchmark"] = {"version": "v0", "sha256": "0" * 64, "item_count": 3}

    warnings = assess(PROFILE, report, NOW, CURRENT).warnings

    assert any("examshield 0.0.1" in warning for warning in warnings)
    assert any("문항 세트(v0)" in warning for warning in warnings)


def test_local_only_verification_warns_that_cloud_models_were_not_measured() -> None:
    warnings = assess(PROFILE, make_report(PROFILE, kind="local"), NOW, CURRENT).warnings

    assert any("클라우드 모델이 없다" in warning for warning in warnings)


def test_deprecated_profiles_warn_without_any_report() -> None:
    deprecated = parse_profile(
        {"name": "old", "scenario": "A", "status": "deprecated", "techniques": [{"id": "decoy"}]}
    )

    assert any(
        "deprecated" in warning for warning in assess(deprecated, None, NOW, CURRENT).warnings
    )


@pytest.mark.parametrize("name", ["none", "none-print", "a-watermark"])
def test_profiles_without_reading_resistance_are_not_judged(name: str) -> None:
    profile = load_profile(name)

    result = assess(profile, make_report(profile, protected=1.0), NOW, CURRENT)

    assert result.state == NOT_APPLICABLE
    assert not any("검증" in warning for warning in result.warnings)


def test_the_newest_matching_report_is_chosen(tmp_path: Path) -> None:
    other = load_profile("a-decoy")
    save(tmp_path / "a", "old.json", make_report(PROFILE, days_ago=40, protected=1.0))
    newest = save(tmp_path / "b", "new.json", make_report(PROFILE, days_ago=2))
    save(tmp_path / "a", "other.json", make_report(other, days_ago=0, protected=1.0))

    report, source, changed = find_latest(
        PROFILE, [tmp_path / "a", tmp_path / "b", tmp_path / "x"], NOW
    )

    assert source == newest
    assert report is not None and not changed
    assert assess(PROFILE, report, NOW, CURRENT, source=source).state == PASSED


def test_reports_that_do_not_count_are_skipped(tmp_path: Path) -> None:
    custom = make_report(PROFILE)
    custom["benchmark"]["version"] = "custom"
    mock = make_report(PROFILE, kind="mock")
    failed_calls = make_report(PROFILE)
    for entry in failed_calls["results"]:
        entry["errors"] = entry["items"]
    truncated = make_report(PROFILE)
    del truncated["results"]
    save(tmp_path, "custom.json", custom)
    save(tmp_path, "mock.json", mock)
    save(tmp_path, "failed.json", failed_calls)
    save(tmp_path, "truncated.json", truncated)
    save(tmp_path, "list.json", [1, 2])
    save(tmp_path, "null.json", None)
    save(tmp_path, "date.json", make_report(PROFILE, created_at="어제"))
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe\x00")

    assert find_latest(PROFILE, [tmp_path], NOW) == (None, None, False)


def test_a_report_for_an_older_version_of_the_profile_does_not_count(tmp_path: Path) -> None:
    stale = make_report(PROFILE)
    stale["profile"]["sha256"] = "f" * 64
    save(tmp_path, "stale.json", stale)

    report, _, changed = find_latest(PROFILE, [tmp_path], NOW)
    result = assess(PROFILE, report, NOW, CURRENT, changed=changed)

    assert report is None and changed
    assert any("마지막 검증 뒤에 바뀌었다" in warning for warning in result.warnings)


def test_a_mock_run_cannot_become_a_verification_record() -> None:
    report = run_verification(
        load_profile("none"), ITEMS, [TextLayerModel(), MockModel(ITEMS.items)], now=NOW
    )

    assert "실제 AI 모델" in (record_refusal(report) or "")
    assert "미래" in (record_refusal(make_report(PROFILE, days_ago=-9), NOW) or "")
    assert record_refusal(make_report(PROFILE)) is None


def test_a_record_is_found_again_from_the_record_folder(tmp_path: Path) -> None:
    path = write_record(make_report(PROFILE, days_ago=1), tmp_path / "verified")

    report, source, _ = find_latest(PROFILE, [tmp_path / "verified"], NOW)

    assert path.name == "a-remap-decoy.json"
    assert source == path and report is not None
    assert "기준 통과" in format_assessment(assess(PROFILE, report, NOW, CURRENT, source=source))


def test_protect_warns_but_still_writes_the_copy(sample_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["protect", str(sample_file), "--out", str(tmp_path / "out"), "--profile", "a-decoy"]
    )

    assert result.exit_code == 0
    assert (tmp_path / "out" / f"{sample_file.stem}.pdf").is_file()
    assert "경고: 프로파일 'a-decoy'을 AI 모델로 검증한 기록이 없다" in result.stderr
    assert "경고" not in result.stdout


def test_protect_with_the_control_profile_prints_no_warning(
    sample_file: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["protect", str(sample_file), "--out", str(tmp_path / "out"), "--profile", "none"]
    )

    assert result.exit_code == 0
    assert "경고" not in result.stderr


def cli_report(profile: Profile, days_ago: int, **changes: Any) -> dict[str, Any]:
    report = make_report(profile, **changes)
    created = datetime.now().astimezone() - timedelta(days=days_ago, minutes=5)
    report["created_at"] = created.isoformat(timespec="seconds")
    return report


def test_protect_reads_reports_and_records_to_decide_what_to_warn(
    sample_file: Path, tmp_path: Path, isolated_verification: dict[str, Path]
) -> None:
    profile = load_profile("a-decoy")
    args = ["protect", str(sample_file), "--profile", "a-decoy", "--out"]

    save(isolated_verification["reports"], "fresh.json", cli_report(profile, 1))
    fresh = runner.invoke(app, [*args, str(tmp_path / "one")])
    save(isolated_verification["reports"], "fresh.json", cli_report(profile, 45))
    old = runner.invoke(app, [*args, str(tmp_path / "two")])
    save(isolated_verification["records"], "a-decoy.json", cli_report(profile, 2, protected=0.9))
    broken = runner.invoke(app, [*args, str(tmp_path / "three")])

    assert fresh.exit_code == old.exit_code == broken.exit_code == 0
    assert "검증" not in fresh.stderr
    assert "45일 전" in old.stderr
    assert "기준에 못 미쳤다" in broken.stderr and "45일 전" not in broken.stderr
    copies = {
        (tmp_path / name / f"{sample_file.stem}.pdf").read_bytes()
        for name in ("one", "two", "three")
    }
    assert len(copies) == 1


def test_protect_warns_even_when_it_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["protect", str(tmp_path / "없음.pdf"), "--out", str(tmp_path), "--profile", "a-decoy"]
    )

    assert result.exit_code == 2
    assert "오류: 입력을 찾지 못했다" in result.stderr
    assert "경고: 프로파일 'a-decoy'을 AI 모델로 검증한 기록이 없다" in result.stderr


def test_profile_list_shows_every_profile_with_its_verification_state(
    isolated_verification: dict[str, Path],
) -> None:
    save(isolated_verification["reports"], "r.json", cli_report(PROFILE, 3))

    result = runner.invoke(app, ["profile", "list"])

    assert result.exit_code == 0
    lines = {line.split()[0]: line for line in result.stdout.splitlines()}
    assert set(lines) == {
        "이름", "a-decoy", "a-full", "a-glyph-remap", "a-rasterize", "a-remap-decoy",
        "a-watermark", "b-dots", "none", "none-print",
    }  # fmt: skip
    assert "기준 통과" in lines["a-remap-decoy"] and "3일 전" in lines["a-remap-decoy"]
    assert "검증 안 됨" in lines["a-decoy"]
    assert "대상 아님" in lines["none"]


def test_profile_list_can_read_another_report_folder(tmp_path: Path) -> None:
    save(tmp_path / "elsewhere", "r.json", cli_report(PROFILE, 50, protected=1.0))

    result = runner.invoke(app, ["profile", "list", "--reports", str(tmp_path / "elsewhere")])

    line = next(line for line in result.stdout.splitlines() if line.startswith("a-remap-decoy"))
    assert "기준 미달" in line and "오래됨" in line


def test_profile_show_prints_techniques_hash_and_warnings() -> None:
    result = runner.invoke(app, ["profile", "show", "a-full"])

    assert result.exit_code == 0
    assert load_profile("a-full").sha256 in result.stdout
    assert "watermark (cell_pt=8, strength=6)" in result.stdout
    assert "검증 상태    검증 안 됨" in result.stdout
    assert "경고: 프로파일 'a-full'은 experimental이다" in result.stdout


def test_profile_show_refuses_an_unknown_name() -> None:
    result = runner.invoke(app, ["profile", "show", "nope"])

    assert result.exit_code == 2
    assert "찾지 못했다" in result.stderr


def test_report_command_adds_the_verdict(tmp_path: Path) -> None:
    passing = save(tmp_path, "pass.json", cli_report(PROFILE, 40))
    changed = cli_report(PROFILE, 1)
    changed["profile"]["sha256"] = "0" * 64
    other = save(tmp_path, "changed.json", changed)

    first = runner.invoke(app, ["report", str(passing)])
    second = runner.invoke(app, ["report", str(other)])

    assert first.exit_code == second.exit_code == 0
    assert "기준 통과" in first.stdout and "40일 전" in first.stdout
    assert "reader [cloud] 정답률 대조군 100.0% → 변환본 25.0%  막음" in first.stdout
    assert "마지막 검증 뒤에 바뀌었다" in second.stdout


def test_record_is_refused_for_mock_and_custom_runs(
    sample_file: Path, tmp_path: Path, isolated_verification: dict[str, Path]
) -> None:
    base = ["verify", "--profile", "a-decoy", "--out", str(tmp_path / "r"), "--record"]

    mock = runner.invoke(app, [*base, "--models", "mock"])
    custom = runner.invoke(app, [*base, "--allow-custom", str(sample_file)])

    assert mock.exit_code == custom.exit_code == 2
    assert "--record" in mock.stderr
    assert not isolated_verification["records"].exists()
    assert not (tmp_path / "r").exists()


class FakeLocalAI(MockModel):
    """모의 모델과 똑같이 답하지만 로컬 AI 모델로 신고한다. --record 경로를 끝까지 태우려는 것이다."""

    def __init__(self) -> None:
        super().__init__(ITEMS.items)
        self.info = ModelInfo(
            provider="ollama", name="fake-local", version="1", kind="local",
            channels=(TEXT,), solves=True,
        )  # fmt: skip


def test_record_stores_the_report_and_protect_then_uses_it(
    sample_file: Path,
    tmp_path: Path,
    isolated_verification: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("examshield.cli._local_models", lambda _: ([FakeLocalAI()], []))

    verified = runner.invoke(
        app, ["verify", "--profile", "a-decoy", "--out", str(tmp_path / "r"), "--record"]
    )
    record = isolated_verification["records"] / "a-decoy.json"
    protected = runner.invoke(
        app, ["protect", str(sample_file), "--out", str(tmp_path / "out"), "--profile", "a-decoy"]
    )

    assert verified.exit_code == 0, verified.output
    assert f"검증 기록: {record}" in verified.stdout
    assert "검증 상태    기준 통과" in verified.stdout
    assert json.loads(record.read_text(encoding="utf-8"))["profile"]["name"] == "a-decoy"
    assert "검증한 기록이 없다" not in protected.stderr
    assert "클라우드 모델이 없다" in protected.stderr


def test_record_is_not_written_when_no_ai_model_answered(
    tmp_path: Path, isolated_verification: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("examshield.cli._local_models", lambda _: ([TextLayerModel()], []))

    result = runner.invoke(
        app, ["verify", "--profile", "a-decoy", "--out", str(tmp_path / "r"), "--record"]
    )

    assert result.exit_code == 1
    assert "검증 기록으로 남기지 않았다" in result.stderr
    assert not isolated_verification["records"].exists()


def with_rows(*rows: dict[str, Any], **changes: Any) -> dict[str, Any]:
    report = make_report(PROFILE, **changes)
    report["results"] = list(rows)
    return report


def failing(entry: dict[str, Any], errors: int) -> dict[str, Any]:
    return {**entry, "errors": errors}


@pytest.mark.parametrize(
    "rows",
    [
        # 변환본 호출이 전부 오류
        [
            row("reader", "text", "control", 1.0),
            failing(row("reader", "text", "protected", None), 12),
        ],
        # 변환본 12건 중 11건 오류, 남은 한 건이 오답
        [
            row("reader", "text", "control", 1.0),
            failing(row("reader", "text", "protected", 0.0), 11),
        ],
        # 대조군을 한 건만 잼
        [
            failing(row("reader", "text", "control", 1.0), 11),
            row("reader", "text", "protected", 0.5),
        ],
        # 변환본 줄이 없음
        [row("reader", "text", "control", 1.0)],
        # 이미지 경로의 변환본만 빠짐
        [
            row("reader", "text", "control", 1.0),
            row("reader", "text", "protected", 0.0),
            row("reader", "image", "control", 1.0),
        ],
    ],
)
def test_a_partly_measured_model_is_never_reported_as_held(rows: list[dict[str, Any]]) -> None:
    result = assess(PROFILE, with_rows(*rows), NOW, CURRENT)

    assert result.state == UNVERIFIED
    assert result.models[0].held is None
    assert any("판정하지 못한 AI 모델" in warning for warning in result.warnings)
    assert "판정 못 함" in format_assessment(result)


def test_a_partly_measured_model_keeps_the_profile_from_passing() -> None:
    report = make_report(PROFILE)
    report["models"].append(
        {"provider": "y", "name": "strong", "version": "2", "kind": "cloud", "note": ""}
    )
    report["results"] += [
        row("strong", "image", "control", 1.0),
        failing(row("strong", "image", "protected", None), 12),
    ]

    result = assess(PROFILE, report, NOW, CURRENT)

    assert result.state == UNVERIFIED
    assert any("strong" in warning for warning in result.warnings)


def test_a_model_that_still_solves_fails_even_when_some_calls_errored() -> None:
    report = with_rows(
        row("reader", "text", "control", 1.0), failing(row("reader", "text", "protected", 0.9), 3)
    )

    assert assess(PROFILE, report, NOW, CURRENT).state == FAILED


def test_skipped_models_are_named_in_a_warning() -> None:
    report = make_report(PROFILE, skipped=[{"model": "openai", "reason": "401"}])

    result = assess(PROFILE, report, NOW, CURRENT)

    assert result.state == PASSED
    assert any("건너뛴 모델이 있다: openai" in warning for warning in result.warnings)


def edited(path: str, value: Any) -> dict[str, Any]:
    report = make_report(PROFILE, protected=0.9)
    target: Any = report
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[int(key)] if key.isdigit() else target[key]
    target[int(keys[-1]) if keys[-1].isdigit() else keys[-1]] = value
    return report


HOSTILE = {
    "nan": edited("results.1.accuracy", float("nan")),
    "negative": edited("results.1.accuracy", -5),
    "infinite": edited("results.0.accuracy", float("inf")),
    "huge-accuracy": edited("results.1.accuracy", 10**400),
    "bool-accuracy": edited("results.1.accuracy", False),
    "infinite-errors": edited("results.1.errors", float("inf")),
    "errors-over-items": edited("results.1.errors", 13),
    "unknown-model-row": edited("results.1.model", "ghost"),
    "odd-variant": edited("results.1.variant", "변환"),
    "duplicate-names": edited(
        "models.1", {"provider": "x", "name": "reader", "version": "1", "kind": "baseline"}
    ),
    "name-not-text": edited("profile.name", ["a-remap-decoy"]),
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_reports_with_impossible_values_are_ignored(name: str, tmp_path: Path) -> None:
    save(tmp_path, "bad.json", HOSTILE[name])

    assert find_latest(PROFILE, [tmp_path], NOW) == (None, None, False)
    with pytest.raises(ValueError):
        assess(PROFILE, HOSTILE[name], NOW, CURRENT)
    assert record_refusal(HOSTILE[name]) is not None


def test_no_file_in_the_report_folder_can_stop_protect(
    sample_file: Path, tmp_path: Path, isolated_verification: dict[str, Path]
) -> None:
    folder = isolated_verification["reports"]
    for name, report in HOSTILE.items():
        save(folder, f"{name}.json", report)
    (folder / "deep.json").write_text("[" * 100_000, encoding="utf-8")
    (folder / "big.json").write_text(" " * 5_000_001, encoding="utf-8")
    (folder / "folder.json").mkdir()
    os.mkfifo(folder / "pipe.json")

    result = runner.invoke(
        app, ["protect", str(sample_file), "--out", str(tmp_path / "out"), "--profile", "a-decoy"]
    )
    listed = runner.invoke(app, ["profile", "list"])

    assert result.exit_code == 0, result.output
    assert "검증한 기록이 없다" in result.stderr
    assert listed.exit_code == 0, listed.output


def test_protect_survives_a_failure_inside_the_drift_check(
    sample_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object) -> None:
        raise RuntimeError("drift")

    monkeypatch.setattr("examshield.cli.find_latest", boom)

    result = runner.invoke(
        app, ["protect", str(sample_file), "--out", str(tmp_path / "out"), "--profile", "a-decoy"]
    )

    assert result.exit_code == 0
    assert (tmp_path / "out" / f"{sample_file.stem}.pdf").is_file()


def test_control_characters_in_a_report_do_not_reach_the_terminal() -> None:
    report = make_report(PROFILE, tool_version="9\x1b]0;x\x07\n경고: 가짜")
    report["models"][0]["name"] = report["results"][0]["model"] = "re\ud800\x1bader"
    report["results"][1]["model"] = "re\ud800\x1bader"

    text = format_assessment(assess(PROFILE, report, NOW, CURRENT))

    assert "\x1b" not in text and "\x07" not in text and "\ud800" not in text
    assert len(text.splitlines()) == len([line for line in text.splitlines() if line.strip()])
    assert not any(line.startswith("경고: 가짜") for line in text.splitlines())


@pytest.mark.parametrize("name", ["../../pyproject", "/etc/hosts", "a" * 5000, "a-decoy\x00"])
def test_report_command_does_not_follow_a_profile_name_as_a_path(name: str, tmp_path: Path) -> None:
    report = make_report(PROFILE)
    report["profile"]["name"] = name
    path = save(tmp_path, "r.json", report)

    result = runner.invoke(app, ["report", str(path)])

    assert result.exit_code == 0, result.output
    assert "판정하지 못한다" in result.stdout


def test_report_command_still_prints_a_report_it_cannot_judge(tmp_path: Path) -> None:
    path = save(tmp_path, "r.json", make_report(PROFILE, created_at="어제"))

    result = runner.invoke(app, ["report", str(path)])

    assert result.exit_code == 0
    assert "판정하지 못한다" in result.stdout


def test_a_record_is_only_written_under_a_builtin_profile_name(tmp_path: Path) -> None:
    report = make_report(PROFILE)
    report["profile"]["name"] = "../escaped"

    with pytest.raises(ExamShieldError):
        write_record(report, tmp_path / "verified")
    assert not (tmp_path / "escaped.json").exists()
