import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from examshield.errors import ModelUnavailableError, NetworkRefusedError
from examshield.netguard import OFFLINE_ENV, is_loopback, require_network
from examshield.verify.models.base import IMAGE, TEXT, PageInput
from examshield.verify.models.local import OllamaModel

PAGE = PageInput(text="1. 문제\n① 가 ② 나", image_png=b"\x89PNG-fake", choice_count=4)


class FakeOllama(BaseHTTPRequestHandler):
    capabilities: list[str] = ["completion", "vision"]
    requests: list[dict[str, Any]] = []

    def log_message(self, *args: object) -> None:
        return

    def _send(self, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({"models": [{"name": "fake-vision:latest", "digest": "abcdef0123456789"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            self._send({"capabilities": type(self).capabilities})
            return
        type(self).requests.append(body)
        self._send({"response": "1. 문제\n① 가 ② 나\n정답: 2"})


@pytest.fixture
def fake_ollama() -> Iterator[str]:
    FakeOllama.requests = []
    FakeOllama.capabilities = ["completion", "vision"]
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_model_name_version_and_channels_come_from_the_server(fake_ollama: str) -> None:
    model = OllamaModel(base_url=fake_ollama)

    assert model.info.name == "fake-vision:latest"
    assert model.info.version == "abcdef012345"
    assert model.info.channels == (TEXT, IMAGE)


def test_image_channel_sends_the_image_and_parses_the_reply(fake_ollama: str) -> None:
    reply = OllamaModel(base_url=fake_ollama).run(PAGE, IMAGE)
    sent = FakeOllama.requests[-1]

    assert sent["images"] == [base64.b64encode(PAGE.image_png).decode()]
    assert sent["options"] == {"temperature": 0, "seed": 0}
    assert reply.answer == 2
    assert reply.transcription == "1. 문제\n① 가 ② 나"


def test_text_channel_sends_the_text_layer_without_an_image(fake_ollama: str) -> None:
    OllamaModel(base_url=fake_ollama).run(PAGE, TEXT)
    sent = FakeOllama.requests[-1]

    assert "images" not in sent
    assert PAGE.text in sent["prompt"]


def test_model_without_vision_only_offers_the_text_channel(fake_ollama: str) -> None:
    FakeOllama.capabilities = ["completion"]

    assert OllamaModel(base_url=fake_ollama).info.channels == (TEXT,)


def test_unknown_model_name_is_reported(fake_ollama: str) -> None:
    with pytest.raises(ModelUnavailableError, match="fake-vision"):
        OllamaModel("missing-model", base_url=fake_ollama)


def test_unreachable_server_is_reported_as_unavailable() -> None:
    with pytest.raises(ModelUnavailableError):
        OllamaModel(base_url="http://127.0.0.1:9")


def test_offline_setting_refuses_a_remote_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OFFLINE_ENV, "1")

    with pytest.raises(NetworkRefusedError):
        OllamaModel(base_url="http://ollama.example.com:11434")


def test_offline_setting_still_allows_loopback(
    fake_ollama: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OFFLINE_ENV, "1")

    assert OllamaModel(base_url=fake_ollama).run(PAGE, TEXT).answer == 2


def test_loopback_detection() -> None:
    assert is_loopback("127.0.0.1")
    assert is_loopback("localhost")
    assert is_loopback("::1")
    assert not is_loopback("10.0.0.5")
    assert not is_loopback("api.example.com")


def test_network_is_allowed_when_offline_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OFFLINE_ENV, raising=False)

    require_network("api.example.com", "테스트")
