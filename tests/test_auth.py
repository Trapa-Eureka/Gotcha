from pathlib import Path

import pytest
from typer.testing import CliRunner

from examshield.auth.keys import SERVICE, find_key, mask
from examshield.cli import app
from examshield.providers import get_provider
from tests.conftest import MemoryKeyring

runner = CliRunner()
SECRET = "sk-live-abcdefghijklmnop-WXYZ"


def test_add_stores_the_key_in_the_keychain_and_never_prints_it(
    isolated_credentials: MemoryKeyring, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = runner.invoke(app, ["auth", "add", "openai"], input=SECRET + "\n")

    assert result.exit_code == 0, result.output
    assert isolated_credentials.store == {(SERVICE, "openai"): SECRET}
    assert SECRET not in result.output
    assert list(tmp_path.iterdir()) == []


def test_list_shows_only_the_last_four_characters_and_the_source(
    isolated_credentials: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_credentials.set_password(SERVICE, "openai", SECRET)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-environment-key-9876")
    output = runner.invoke(app, ["auth", "list"]).output
    lines = {line.split()[0]: line for line in output.splitlines()}

    assert "****WXYZ" in lines["openai"] and "OS 키체인" in lines["openai"]
    assert "****9876" in lines["google"] and "환경변수 GEMINI_API_KEY" in lines["google"]
    assert "없음" in lines["anthropic"]
    assert SECRET not in output
    assert "environment-key" not in output


def test_environment_variable_wins_over_the_keychain(
    isolated_credentials: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_credentials.set_password(SERVICE, "anthropic", "from-keychain-000000")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environment-111111")
    found = find_key(get_provider("anthropic"))

    assert found is not None
    assert found.value == "from-environment-111111"


def test_google_also_reads_the_other_standard_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-standard-name-2222")
    found = find_key(get_provider("google"))

    assert found is not None and found.source == "환경변수 GOOGLE_API_KEY"


def test_remove_deletes_from_the_keychain_and_says_so_when_there_is_nothing(
    isolated_credentials: MemoryKeyring,
) -> None:
    isolated_credentials.set_password(SERVICE, "openai", SECRET)

    assert "지웠다" in runner.invoke(app, ["auth", "remove", "openai"]).output
    assert isolated_credentials.store == {}
    assert "없다" in runner.invoke(app, ["auth", "remove", "openai"]).output


def test_unknown_provider_and_empty_key_exit_with_code_2(
    isolated_credentials: MemoryKeyring,
) -> None:
    assert runner.invoke(app, ["auth", "add", "nope"]).exit_code == 2
    assert runner.invoke(app, ["auth", "add", "openai"], input="   \n").exit_code == 2
    assert isolated_credentials.store == {}


def test_short_values_are_fully_masked() -> None:
    assert mask("abcd") == "****"
    assert mask("12345678901234") == "****1234"
