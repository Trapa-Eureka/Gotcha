from collections.abc import Iterator
from pathlib import Path

import keyring
import keyring.backend
import pymupdf
import pytest

from examshield.benchmark import load_items, render_items_pdf
from examshield.netguard import OFFLINE_ENV
from examshield.pdfio import save_deterministic
from examshield.providers import PROVIDERS

A_PROFILES = ("a-rasterize", "a-glyph-remap", "a-decoy", "a-remap-decoy")


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return render_items_pdf(load_items().items[:3])


@pytest.fixture
def sample_file(tmp_path: Path, sample_pdf: bytes) -> Path:
    folder = tmp_path / "in"
    folder.mkdir()
    path = folder / "exam.pdf"
    path.write_bytes(sample_pdf)
    return path


def make_pdf(text: str, fontsize: float = 10) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="kr", fontbuffer=pymupdf.Font("korea").buffer)
    page.insert_textbox(
        pymupdf.Rect(72, 90, 523, 750), text, fontname="kr", fontsize=fontsize, lineheight=1.8
    )
    return save_deterministic(doc)


class MemoryKeyring(keyring.backend.KeyringBackend):
    """테스트용 키체인. 실제 OS 키체인을 건드리지 않는다."""

    priority = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryKeyring]:
    """어떤 테스트도 개발자의 실제 키나 키체인으로 클라우드에 닿지 않게 한다."""
    for provider in PROVIDERS:
        for name in (*provider.key_env, provider.model_env):
            monkeypatch.delenv(name, raising=False)
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "EXAMSHIELD_OLLAMA_MODEL",
        OFFLINE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    # 개발자가 띄워 둔 Ollama나 원격 Ollama 주소로 가지 않게, 아무도 듣지 않는 포트로 돌린다.
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    previous = keyring.get_keyring()
    memory = MemoryKeyring()
    keyring.set_keyring(memory)
    yield memory
    keyring.set_keyring(previous)
