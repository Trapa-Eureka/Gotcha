import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from examshield.benchmark import load_items, render_items_pdf
from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import load_profile, parse_profile
from examshield.protect.pipeline import KeyEntry, protect_bytes
from examshield.verify.drift import record_refusal
from examshield.verify.metrics import parse_answers
from examshield.verify.models.base import ModelClient, PageInput, prompt_of, read_reply
from examshield.verify.models.baseline import TesseractModel, TextLayerModel
from examshield.verify.models.mock import MockModel
from examshield.verify.report import format_report, load_report, write_report
from examshield.verify.runner import (
    Skipped,
    builtin_subject,
    custom_subject,
    run_verification,
)

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


def test_avoid_answers_measures_the_profile_as_if_a_key_was_given(tmp_path: Path) -> None:
    def run(*extra: str) -> dict[str, Any]:
        folder = tmp_path / ("keyed" if extra else "plain")
        result = CliRunner().invoke(
            app,
            [
                "verify",
                "--profile",
                "a-remap-decoy",
                "--models",
                "mock",
                "--out",
                str(folder),
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return dict(json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8")))

    plain, keyed = run(), run("--avoid-answers")
    score = {
        name: next(
            row
            for row in report["results"]
            if row["variant"] == "protected" and row["model"] == "mock-reader"
        )
        for name, report in (("plain", plain), ("keyed", keyed))
    }

    # 모의 모델은 미끼를 그대로 따른다. 미끼가 정답을 피하면 한 문항도 못 맞힌다.
    assert score["plain"]["accuracy"] > 0 and score["plain"]["decoy_items"] < 12
    assert score["keyed"]["accuracy"] == 0.0 and score["keyed"]["decoy_items"] == 12
    assert keyed["answer_key"] is True and plain["answer_key"] is False
    assert record_refusal(keyed) is not None


def test_avoid_answers_cannot_be_recorded_or_used_with_a_custom_file(
    sample_file: Path, tmp_path: Path
) -> None:
    base = ["verify", "--profile", "a-decoy", "--models", "mock", "--out", str(tmp_path / "r")]

    custom = CliRunner().invoke(app, [*base, "--avoid-answers", "--allow-custom", str(sample_file)])

    assert custom.exit_code == 2 and "내장 문항" in custom.stderr


def test_benchmark_v2_puts_four_items_on_a_page() -> None:
    items = load_items("v2")
    pdf = render_items_pdf(items.items, items.per_page)
    doc = open_pdf_bytes(pdf)

    assert (len(items.items), items.per_page, doc.page_count) == (36, 4, 9)
    assert [item.number for item in items.pages[1]] == [5, 6, 7, 8]
    assert all(1 <= item.answer <= 4 and len(item.choices) == 4 for item in items.items)
    assert len({item.question for item in items.items}) == 36
    assert "5. " in page_text(doc[1]) and "8. " in page_text(doc[1])
    assert load_items().per_page == 1


def test_numbered_answers_are_read_per_question() -> None:
    reply = "본문\n정답 5: 3\n정답 6번: ②\n정답 7 : 9\n정답 99: 1\n정답 5: 4"

    assert parse_answers(reply, (5, 6, 7, 8), 4) == {5: 4, 6: 2}
    page = PageInput(text="", image_png=b"", choice_count=4, questions=(5, 6))
    assert read_reply(reply, page).answers == {5: 4, 6: 2}
    assert read_reply(reply, page).transcription == "본문"
    assert "문항 5, 6이(가) 차례로 있다" in prompt_of(page)
    assert "문항 한 개" in prompt_of(PageInput(text="", image_png=b"", choice_count=4))


def run_v2(profile_name: str, *, numbered: bool, keyed: bool = True) -> dict[str, Any]:
    items = load_items("v2")
    subject = builtin_subject(items)
    profile = load_profile(profile_name)
    key = tuple(
        KeyEntry(page, number if numbered else None, answer)
        for page, truth in enumerate(subject.truths, start=1)
        for number, answer in truth.questions
    )
    report = run_verification(
        profile,
        subject,
        [MockModel(items.items)],
        now=NOW,
        transform=lambda pdf: protect_bytes(pdf, profile, None, None, key if keyed else None),
    )
    return {row["variant"]: row for row in report["results"]}


def test_multi_question_pages_are_scored_per_question() -> None:
    rows = run_v2("a-remap-decoy", numbered=True)

    assert rows["control"]["items"] == 9 and rows["control"]["questions"] == 36
    assert rows["control"]["accuracy"] == 1.0
    # 모의 모델은 문항을 가리키는 미끼를 그대로 따른다. 미끼가 정답을 피하므로 하나도 못 맞힌다.
    assert rows["protected"]["accuracy"] == 0.0
    assert rows["protected"]["decoy_hit_rate"] == 1.0 and rows["protected"]["decoy_items"] == 36


def test_a_page_level_decoy_is_counted_for_every_question_on_the_page() -> None:
    rows = run_v2("a-remap-decoy", numbered=False)

    # 쪽마다 미끼가 하나다. 모의 모델은 그 번호를 그 쪽의 모든 문항에 답한다. 한 쪽의 네 문항이
    # 네 번호를 모두 정답으로 쓰는 쪽에서는 미끼가 정답을 피할 수 없어서, 그 쪽에서 한 문항씩 맞힌다.
    assert 0.0 < rows["protected"]["accuracy"] < 0.25
    assert rows["protected"]["decoy_hit_rate"] == 1.0
    assert rows["protected"]["decoy_items"] == round(36 * (1 - rows["protected"]["accuracy"]))


def test_keeping_digits_leaves_question_numbers_readable(sample_pdf: bytes) -> None:
    scrambled = page_text(
        open_pdf_bytes(protect_bytes(sample_pdf, load_profile("a-remap-decoy")).pdf)[1]
    )
    kept = page_text(
        open_pdf_bytes(protect_bytes(sample_pdf, load_profile("a-remap-decoy-num")).pdf)[1]
    )
    original = page_text(open_pdf_bytes(sample_pdf)[1])
    body = kept.split("제시한다.")[-1]

    assert original.lstrip().startswith("2. ")
    assert body.lstrip().startswith("2. ")
    assert not scrambled.split("제시한다.")[-1].lstrip().startswith("2. ")
    assert "대한민국" not in kept
    assert load_profile("a-remap-decoy-num").sha256 != load_profile("a-remap-decoy").sha256


def test_bad_keep_values_are_refused(sample_pdf: bytes) -> None:
    for keep in (["number"], "digit"):
        profile = parse_profile(
            {
                "name": "x",
                "scenario": "A",
                "status": "experimental",
                "techniques": [{"id": "glyph_remap", "params": {"keep": keep}}],
            }
        )
        with pytest.raises(ExamShieldError, match="keep"):
            protect_bytes(sample_pdf, profile)


def test_cli_measures_benchmark_v2_and_refuses_to_record_it(tmp_path: Path) -> None:
    base = ["verify", "--profile", "a-remap-decoy-num", "--models", "mock", "--out", str(tmp_path)]

    result = CliRunner().invoke(app, [*base, "--benchmark", "v2", "--question-decoys"])
    recorded = CliRunner().invoke(app, [*base, "--benchmark", "v2", "--record"])
    unknown = CliRunner().invoke(app, [*base, "--benchmark", "v9"])
    report = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))

    assert result.exit_code == 0, result.output
    assert report["benchmark"]["version"] == "v2" and report["benchmark"]["item_count"] == 9
    assert recorded.exit_code == 2 and unknown.exit_code == 2


def test_keep_replies_writes_what_the_model_said_next_to_the_report(tmp_path: Path) -> None:
    base = ["verify", "--profile", "a-remap-decoy", "--models", "mock", "--out", str(tmp_path)]

    result = CliRunner().invoke(
        app, [*base, "--benchmark", "v2", "--question-decoys", "--keep-replies"]
    )
    kept_path = next(tmp_path.glob("*.replies.json"))
    kept = json.loads(kept_path.read_text(encoding="utf-8"))
    report_path = tmp_path / kept["report"]
    protected = [
        entry
        for entry in kept["replies"]
        if entry["model"] == "mock-reader" and entry["variant"] == "protected"
    ]

    assert result.exit_code == 0, result.output
    assert "문항별 답" in result.output and str(kept_path) in result.output
    assert report_path.is_file() and "replies" not in json.loads(report_path.read_text("utf-8"))
    assert len(protected) == 9 and all(len(entry["questions"]) == 4 for entry in protected)
    assert all(entry["reply"] and entry["input_text"] for entry in protected)
    # 모의 모델은 문항을 가리키는 미끼를 그대로 따른다.
    assert all(q["given"] == q["decoy"] for entry in protected for q in entry["questions"])
    # 응답 파일은 리포트로도, 검증 기록으로도 읽히지 않는다.
    assert CliRunner().invoke(app, ["report", str(kept_path)]).exit_code == 2
    assert record_refusal(kept) is not None


def test_replies_are_not_kept_without_the_option_or_for_a_custom_file(
    tmp_path: Path, sample_file: Path
) -> None:
    base = ["verify", "--profile", "none", "--models", "mock", "--out", str(tmp_path)]

    plain = CliRunner().invoke(app, base)
    custom = CliRunner().invoke(app, [*base, "--allow-custom", str(sample_file), "--keep-replies"])

    assert plain.exit_code == 0, plain.output
    assert custom.exit_code == 2 and "--keep-replies" in custom.output
    assert not list(tmp_path.glob("*.replies.json"))
    with pytest.raises(ExamShieldError, match="내장 문항"):
        run_verification(
            load_profile("none"),
            custom_subject(sample_file.read_bytes()),
            [TextLayerModel()],
            now=NOW,
            replies=[],
        )


def test_a_real_reply_is_kept_word_for_word() -> None:
    page = PageInput(text="", image_png=b"", choice_count=4, questions=(1, 2))

    reply = read_reply("못 읽겠다.\n정답 1: 3\n정답 2: 4", page)

    assert reply.raw == "못 읽겠다.\n정답 1: 3\n정답 2: 4" and reply.answers == {1: 3, 2: 4}
