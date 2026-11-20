import ast
import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.pdfio import read_pdf, render_gray
from examshield.verify.capture_sim import CaptureConditions, encode_png, simulate_capture
from tests.conftest import A_PROFILES

PACKAGE = Path(__file__).resolve().parent.parent / "examshield"
OFFLINE_ROOTS = (
    "examshield.protect.pipeline",
    "examshield.protect.copies",
    "examshield.check.compare",
    "examshield.convert",
    "examshield.trace.detect",
)
NETWORK_MODULES = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "requests",
    "httpx",
    "aiohttp",
    "openai",
    "anthropic",
    "google",
    "keyring",
}


@pytest.fixture
def no_sockets(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def refuse(name: str) -> Any:
        def blocked(*args: object, **kwargs: object) -> None:
            attempts.append(name)
            raise AssertionError(f"네트워크 호출이 있었다: {name}")

        return blocked

    for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, name, refuse(name))
    return attempts


@pytest.mark.parametrize("profile", ("none", *A_PROFILES, "b-dots"))
def test_protect_and_check_run_with_sockets_blocked(
    profile: str, sample_file: Path, tmp_path: Path, no_sockets: list[str]
) -> None:
    runner = CliRunner()
    out = tmp_path / "out"

    made = runner.invoke(
        app, ["protect", str(sample_file), "--out", str(out), "--profile", profile]
    )
    checked = runner.invoke(app, ["check", str(sample_file), str(out / "exam.pdf")])

    assert made.exit_code == 0, made.output
    assert checked.exit_code == 0, checked.output
    assert no_sockets == []


def test_copies_and_trace_run_with_sockets_blocked(
    sample_file: Path, tmp_path: Path, no_sockets: list[str]
) -> None:
    runner = CliRunner()
    seed = tmp_path / "exam.seed"
    seed.write_bytes(b"0123456789abcdef-seed")
    ids = tmp_path / "ids.csv"
    ids.write_text("A1\nA2\n")
    table = tmp_path / "map.csv"
    out = tmp_path / "out"
    made = runner.invoke(
        app,
        [
            "protect", str(sample_file), "--out", str(out), "--profile", "a-full",
            "--copies", str(ids), "--seed-file", str(seed), "--map-file", str(table),
        ],
    )  # fmt: skip
    leaked = tmp_path / "leak.pdf"
    leaked.write_bytes((out / "exam-A2.pdf").read_bytes() + b"\n")
    traced = runner.invoke(
        app,
        [
            "trace", str(leaked), "--original", str(sample_file),
            "--map-file", str(table), "--seed-file", str(seed),
        ],
    )  # fmt: skip

    assert made.exit_code == 0, made.output
    assert traced.exit_code == 0 and "A2" in traced.output, traced.output
    assert no_sockets == []


def test_the_socket_block_really_blocks(no_sockets: list[str]) -> None:
    with pytest.raises(AssertionError):
        socket.create_connection(("127.0.0.1", 9))
    assert no_sockets == ["create_connection"]


def _imports(module: str) -> set[str]:
    relative = Path(*module.split("."))
    path = PACKAGE.parent / relative.with_suffix(".py")
    if not path.is_file():
        path = PACKAGE.parent / relative / "__init__.py"
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _is_module(name: str) -> bool:
    relative = PACKAGE.parent / Path(*name.split("."))
    return relative.with_suffix(".py").is_file() or (relative / "__init__.py").is_file()


def test_protect_and_check_never_import_a_network_module() -> None:
    seen: set[str] = set()
    queue = list(OFFLINE_ROOTS)
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        for name in _imports(module):
            assert name.split(".")[0] not in NETWORK_MODULES, (
                f"{module}이(가) {name}을(를) 가져온다"
            )
            if name.startswith("examshield") and _is_module(name):
                queue.append(name)

    assert "examshield.protect.techniques.glyph_remap" in seen
    assert "examshield.verify.models.local" not in seen


def test_print_copies_photo_trace_and_calibrate_run_with_sockets_blocked(
    sample_file: Path, tmp_path: Path, no_sockets: list[str]
) -> None:
    runner = CliRunner()
    seed = tmp_path / "exam.seed"
    seed.write_bytes(b"0123456789abcdef-seed")
    ids = tmp_path / "ids.csv"
    ids.write_text("A1\nA2\n")
    table = tmp_path / "map.csv"
    out = tmp_path / "out"
    made = runner.invoke(
        app,
        [
            "protect", str(sample_file), "--out", str(out), "--profile", "b-dots",
            "--copies", str(ids), "--seed-file", str(seed), "--map-file", str(table),
        ],
    )  # fmt: skip
    printed = render_gray(read_pdf(out / "exam-A2.pdf")[0], 300)
    photo = tmp_path / "photo.png"
    photo.write_bytes(encode_png(simulate_capture(printed, CaptureConditions())))
    traced = runner.invoke(
        app,
        [
            "trace", str(photo), "--original", str(sample_file),
            "--map-file", str(table), "--seed-file", str(seed),
        ],
    )  # fmt: skip
    measured = runner.invoke(
        app, ["calibrate", str(photo), "--printed", str(out / "exam-A2.pdf"), "--profile", "b-dots"]
    )

    assert made.exit_code == 0, made.output
    assert traced.exit_code == 0 and "A2" in traced.output, traced.output
    assert measured.exit_code == 0, measured.output
    assert no_sockets == []
