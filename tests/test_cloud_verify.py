import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from examshield import cli
from examshield.benchmark import load_items
from examshield.cli import app
from examshield.errors import ModelAbortError
from examshield.profiles import load_profile
from examshield.protect.pipeline import protect_bytes
from examshield.verify.models.base import FILE, IMAGE, TEXT, ModelInfo, ModelReply, PageInput
from examshield.verify.runner import Skipped, estimate_calls, run_verification
from tests.conftest import make_pdf
from tests.test_verify import NOW

runner = CliRunner()
SECRET = "sk-live-abcdefghijklmnop-WXYZ"


class FakeCloudModel:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, PageInput]] = []
        self._fail = fail
        self.info = ModelInfo(
            provider="anthropic",
            name="fake-cloud",
            version="fake-1",
            kind="cloud",
            channels=(TEXT, IMAGE, FILE),
            solves=True,
        )

    def run(self, page: PageInput, channel: str) -> ModelReply:
        self.calls.append((channel, page))
        if self._fail:
            raise ModelAbortError("키가 거부됐다")
        return ModelReply(transcription=page.text, answer=1)


@pytest.fixture
def fake_cloud(monkeypatch: pytest.MonkeyPatch) -> FakeCloudModel:
    model = FakeCloudModel()
    monkeypatch.setattr(
        cli, "_cloud_models", lambda: ([model], [Skipped("openai", "API 키가 없다.")])
    )
    return model


def verify(*args: str, answer: str | None = None) -> tuple[int, str]:
    result = runner.invoke(app, ["verify", *args], input=answer)
    return result.exit_code, result.output


def test_without_any_key_cloud_verify_explains_how_to_set_one_and_stops(tmp_path: Path) -> None:
    code, output = verify("--profile", "none", "--models", "cloud", "--out", str(tmp_path))

    assert code == 2
    for hint in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "examshield auth add"):
        assert hint in output
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("how", ("flag", "env"))
def test_offline_refuses_cloud_models_even_when_a_key_exists(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    args = ["verify", "--profile", "none", "--models", "cloud", "--yes", "--out", str(tmp_path)]
    if how == "env":
        monkeypatch.setenv("EXAMSHIELD_OFFLINE", "1")
    else:
        args = ["--offline", *args]
    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "오프라인" in result.output
    assert SECRET not in result.output
    assert list(tmp_path.iterdir()) == []


def test_the_confirmation_shows_what_leaves_and_declining_sends_nothing(
    fake_cloud: FakeCloudModel, tmp_path: Path
) -> None:
    code, output = verify(
        "--profile", "a-decoy", "--models", "cloud", "--out", str(tmp_path), answer="n\n"
    )

    assert code == 1
    assert "내장 문항 세트 v1, 12쪽 × 2벌" in output
    assert "Anthropic (fake-cloud), 예상 호출 72건" in output
    assert fake_cloud.calls == []
    assert list(tmp_path.iterdir()) == []


def test_no_answer_at_all_counts_as_declining(fake_cloud: FakeCloudModel, tmp_path: Path) -> None:
    code, _ = verify("--profile", "none", "--models", "cloud", "--out", str(tmp_path), answer="")

    assert code == 1
    assert fake_cloud.calls == []


def test_confirming_runs_exactly_the_estimated_number_of_calls(
    fake_cloud: FakeCloudModel, tmp_path: Path
) -> None:
    code, output = verify(
        "--profile", "a-decoy", "--models", "cloud", "--out", str(tmp_path), answer="y\n"
    )
    report = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))

    assert code == 0, output
    assert len(fake_cloud.calls) == 72 == estimate_calls(load_profile("a-decoy"), 12, fake_cloud)
    assert {row["channel"] for row in report["results"]} == {"text", "image", "file"}
    assert report["skipped"] == [{"model": "openai", "reason": "API 키가 없다."}]
    assert "PDF 파일" in output


def test_yes_skips_the_question(fake_cloud: FakeCloudModel, tmp_path: Path) -> None:
    code, output = verify("--profile", "none", "--models", "cloud", "--yes", "--out", str(tmp_path))

    assert code == 0, output
    assert "계속할까" not in output
    assert len(fake_cloud.calls) == 36


def test_each_page_is_uploaded_as_a_one_page_pdf(
    fake_cloud: FakeCloudModel, tmp_path: Path
) -> None:
    import pymupdf  # noqa: PLC0415

    verify("--profile", "none", "--models", "cloud", "--yes", "--out", str(tmp_path))
    files = [page.pdf for channel, page in fake_cloud.calls if channel == FILE]

    assert len(files) == 12
    assert all(pymupdf.open(stream=data, filetype="pdf").page_count == 1 for data in files)
    assert max(len(data) for data in files) < 100_000


def test_cloud_verification_only_uses_the_built_in_set_unless_a_file_is_allowed(
    fake_cloud: FakeCloudModel, tmp_path: Path
) -> None:
    custom = tmp_path / "mine.pdf"
    custom.write_bytes(make_pdf("비밀 문항: 정답이 무엇인가? ① 하나 ② 둘 ③ 셋"))
    out = tmp_path / "out"
    code, output = verify(
        "--profile", "a-decoy", "--models", "cloud", "--yes", "--out", str(out),
        "--allow-custom", str(custom),
    )  # fmt: skip
    written = next(out.glob("*.json")).read_text(encoding="utf-8")
    report = json.loads(written)

    assert code == 0, output
    assert "이 파일의 내용이 위 회사들의 서버로 전송된다" in output
    assert report["benchmark"]["version"] == "custom"
    assert report["benchmark"]["item_count"] == 1
    assert "비밀 문항" not in written
    assert all(row["accuracy"] is None for row in report["results"])
    assert any("비밀 문항" in page.text for _, page in fake_cloud.calls)


def test_without_allow_custom_nothing_but_the_built_in_items_is_sent(
    fake_cloud: FakeCloudModel, tmp_path: Path
) -> None:
    verify("--profile", "none", "--models", "cloud", "--yes", "--out", str(tmp_path))
    known = {item.text.strip() for item in load_items().items}

    assert {page.text.strip() for _, page in fake_cloud.calls} == known


def test_a_model_whose_key_is_rejected_is_dropped_and_the_rest_continue() -> None:
    good, bad = FakeCloudModel(), FakeCloudModel(fail=True)
    profile = load_profile("a-decoy")
    report = run_verification(
        profile,
        load_items(),
        [bad, good],
        now=NOW,
        transform=lambda pdf: protect_bytes(pdf, profile),
    )

    assert len(bad.calls) == 1
    assert len(report["models"]) == 1
    assert report["skipped"] == [{"model": "fake-cloud", "reason": "키가 거부됐다"}]
    assert len(report["results"]) == 6


def test_refusals_are_counted_and_noted() -> None:
    class Refuser(FakeCloudModel):
        def run(self, page: PageInput, channel: str) -> ModelReply:
            return ModelReply(transcription="", answer=None, refused=True)

    report = run_verification(load_profile("none"), load_items(), [Refuser()], now=NOW)

    assert report["results"][0]["refusals"] == 12
    assert report["results"][0]["legibility"] == 0.0
    assert any("거부" in note for note in report["notes"])


def test_keys_never_reach_the_report_or_the_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    made, skipped = cli._cloud_models()
    monkeypatch.setattr(cli, "_cloud_models", lambda: ([FakeCloudModel()], skipped))
    code, output = verify("--profile", "none", "--models", "cloud", "--yes", "--out", str(tmp_path))

    assert [model.info.provider for model in made] == ["anthropic"]
    assert {entry.model for entry in skipped} == {"openai", "google"}
    assert code == 0
    assert SECRET not in output
    assert SECRET not in next(tmp_path.glob("*.json")).read_text(encoding="utf-8")


def test_all_runs_local_models_and_lists_cloud_ones_without_keys_as_skipped(tmp_path: Path) -> None:
    code, output = verify("--profile", "none", "--models", "all", "--out", str(tmp_path))

    assert code == 0, output
    assert "text-layer" in output
    assert "건너뜀: anthropic — API 키가 없다" in output


def test_offline_with_all_still_runs_local_models_and_skips_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    result = runner.invoke(
        app, ["--offline", "verify", "--profile", "none", "--models", "all", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "text-layer" in result.output
    assert "건너뜀: cloud — 오프라인 설정" in result.output


def test_exit_code_is_1_when_every_model_was_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_cloud_models", lambda: ([FakeCloudModel(fail=True)], []))
    code, output = verify("--profile", "none", "--models", "cloud", "--yes", "--out", str(tmp_path))

    assert code == 1
    assert "측정된 값이 하나도 없다" in output
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_key_with_stray_characters_is_skipped_with_a_reason_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-한글이섞인키")
    made, skipped = cli._cloud_models()

    assert made == []
    assert any(entry.model == "openai" and "섞여" in entry.reason for entry in skipped)


@pytest.mark.parametrize("value", ("1", "y", "TRUE", "enabled"))
def test_any_unrecognised_offline_value_counts_as_offline(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examshield.netguard import offline_enabled  # noqa: PLC0415

    monkeypatch.setenv("EXAMSHIELD_OFFLINE", value)

    assert offline_enabled()
