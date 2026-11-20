import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from examshield.benchmark import load_items
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.profiles import load_profile, parse_profile
from examshield.protect.pipeline import protect_bytes
from examshield.verify.drift import record_refusal
from examshield.verify.models.base import ModelClient
from examshield.verify.models.baseline import TesseractModel, TextLayerModel
from examshield.verify.models.mock import MockModel
from examshield.verify.report import format_report, load_report, write_report
from examshield.verify.runner import Skipped, run_verification

NOW = datetime(2026, 11, 20, 9, 0, 0, tzinfo=UTC)
REQUIRED_REPORT_KEYS = {
    "created_at",
    "tool_version",
    "profile",
    "benchmark",
    "simulation",
    "models",
    "results",
}


def control_report() -> dict[str, Any]:
    item_set = load_items()
    models: list[ModelClient] = [TextLayerModel(), MockModel(item_set.items)]
    return run_verification(load_profile("none"), item_set, models, now=NOW)


def test_control_group_scores_come_out_for_the_untransformed_profile() -> None:
    report = control_report()
    rows = {row["model"]: row for row in report["results"]}

    assert rows["text-layer"]["variant"] == "control"
    assert rows["text-layer"]["legibility"] == 1.0
    assert rows["text-layer"]["accuracy"] is None
    assert rows["mock-reader"]["accuracy"] == 1.0
    assert rows["mock-reader"]["items"] == 12


def test_report_carries_every_required_field() -> None:
    report = control_report()

    assert report.keys() >= REQUIRED_REPORT_KEYS
    assert report["created_at"] == "2026-11-20T09:00:00+00:00"
    assert report["profile"]["name"] == "none"
    assert len(report["profile"]["sha256"]) == 64
    assert report["benchmark"]["version"] == "v1"
    assert {model["name"] for model in report["models"]} == {"text-layer", "mock-reader"}
    assert all(model["version"] for model in report["models"])


def test_report_does_not_contain_question_text() -> None:
    dumped = json.dumps(control_report(), ensure_ascii=False)

    for item in load_items().items:
        assert item.question not in dumped


def test_print_scenario_records_the_simulation_conditions_and_skips_the_text_channel() -> None:
    item_set = load_items()
    models: list[ModelClient] = [TextLayerModel(), MockModel(item_set.items)]
    report = run_verification(load_profile("none-print"), item_set, models, now=NOW)

    assert report["simulation"]["print_dpi"] == 300
    assert report["simulation"]["jpeg_quality"] == 80
    assert report["results"] == []


def test_profile_with_techniques_is_refused_until_protect_exists() -> None:
    profile = parse_profile(
        {"name": "x", "scenario": "A", "status": "experimental", "techniques": [{"id": "decoy"}]}
    )

    with pytest.raises(ExamShieldError):
        run_verification(profile, load_items(), [TextLayerModel()], now=NOW)


def test_report_round_trips_through_a_file_and_formats_as_a_table(tmp_path: Path) -> None:
    report = control_report()
    report["skipped"] = [{"model": "ollama", "reason": "연결하지 못했다"}]
    path = write_report(report, tmp_path)
    text = format_report(load_report(path))

    assert path.name == "verify-none-20261120T090000.json"
    assert "글자 판독률" in text
    assert "100.0%" in text
    assert "건너뜀: ollama" in text


def test_report_file_with_missing_fields_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"created_at": "x"}')

    with pytest.raises(ExamShieldError):
        load_report(path)


def test_skipped_models_are_listed_in_the_report() -> None:
    report = run_verification(
        load_profile("none"),
        load_items(),
        [TextLayerModel()],
        now=NOW,
        skipped=[Skipped("ollama", "없다")],
    )

    assert report["skipped"] == [{"model": "ollama", "reason": "없다"}]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract가 없다")
def test_tesseract_baseline_reads_the_simulated_capture() -> None:
    report = run_verification(load_profile("none-print"), load_items(), [TesseractModel()], now=NOW)
    row = report["results"][0]

    assert row["channel"] == "image"
    assert row["errors"] == 0
    assert 0.0 < row["legibility"] <= 1.0


def test_cli_verify_writes_a_report_and_report_command_reads_it(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["verify", "--profile", "none", "--models", "mock", "--out", str(tmp_path)]
    )
    files = list(tmp_path.glob("verify-none-*.json"))

    assert result.exit_code == 0, result.output
    assert len(files) == 1
    shown = runner.invoke(app, ["report", str(files[0])])
    assert shown.exit_code == 0
    assert "mock-reader" in shown.output


def test_cli_unknown_profile_exits_with_code_2(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["verify", "--profile", "nope", "--models", "mock", "--out", str(tmp_path)]
    )

    assert result.exit_code == 2


def protected_report(profile_name: str) -> dict[str, Any]:
    item_set = load_items()
    profile = load_profile(profile_name)
    models: list[ModelClient] = [TextLayerModel(), MockModel(item_set.items)]
    return run_verification(
        profile, item_set, models, now=NOW, transform=lambda pdf: protect_bytes(pdf, profile)
    )


def rows_for(report: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    return {row["variant"]: row for row in report["results"] if row["model"] == model}


def test_scrambled_profile_is_measured_next_to_the_control_group() -> None:
    report = protected_report("a-glyph-remap")
    text_layer = rows_for(report, "text-layer")
    reader = rows_for(report, "mock-reader")

    assert text_layer["control"]["legibility"] == 1.0
    assert text_layer["protected"]["legibility"] < 0.5
    assert reader["control"]["accuracy"] == 1.0
    assert reader["protected"]["accuracy"] == 0.0
    assert reader["protected"]["decoy_hit_rate"] is None
    assert report["protection"]["techniques"][0]["id"] == "glyph_remap"
    assert report["notes"] == []


def test_decoy_hit_rate_counts_only_items_where_the_decoy_is_wrong() -> None:
    report = protected_report("a-decoy")
    reader = rows_for(report, "mock-reader")

    assert reader["control"]["decoy_hit_rate"] is None
    assert reader["protected"]["decoy_hit_rate"] == 1.0
    assert 0 < reader["protected"]["decoy_items"] <= 12
    assert reader["protected"]["accuracy"] == (12 - reader["protected"]["decoy_items"]) / 12
    assert rows_for(report, "text-layer")["protected"]["decoy_hit_rate"] is None


def test_cli_verify_measures_a_profile_with_techniques(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["verify", "--profile", "a-remap-decoy", "--models", "mock", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "변환본" in result.output


def test_verify_can_be_limited_to_chosen_channels(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "verify", "--profile", "a-decoy", "--models", "mock",
            "--channels", "text", "--out", str(tmp_path),
        ],
    )  # fmt: skip
    report = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))

    assert result.exit_code == 0, result.output
    assert {row["channel"] for row in report["results"]} == {"text"}
    assert report["channels_limited"] is True
    assert "검증 기록으로는 쓰지 않는다" in result.stdout
    assert record_refusal(report) is not None


def test_bad_channel_choices_are_refused(tmp_path: Path) -> None:
    def run(*extra: str) -> Any:
        return CliRunner().invoke(
            app,
            ["verify", "--profile", "a-decoy", "--models", "mock", "--out", str(tmp_path), *extra],
        )

    unknown = run("--channels", "image,fax")
    recorded = run("--channels", "text", "--record")
    unused = CliRunner().invoke(
        app,
        ["verify", "--profile", "none-print", "--models", "mock", "--channels", "text",
         "--out", str(tmp_path)],
    )  # fmt: skip

    assert unknown.exit_code == 2 and "모르는 경로" in unknown.stderr
    assert recorded.exit_code == 2 and "--channels" in recorded.stderr
    assert unused.exit_code == 2 and "재지 않는다" in unused.stderr
    assert not list(tmp_path.glob("*.json"))


def test_a_full_run_is_not_marked_as_channel_limited() -> None:
    report = control_report()

    assert report["channels_limited"] is False
