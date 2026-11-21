import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

pytest.importorskip("anthropic", reason="클라우드 SDK가 없다. uv sync --extra cloud")
pytest.importorskip("openai")
pytest.importorskip("google.genai")

from examshield.auth.keys import FoundKey
from examshield.errors import ModelAbortError, ModelUnavailableError, NetworkRefusedError
from examshield.netguard import OFFLINE_ENV
from examshield.verify.models.anthropic import AnthropicModel
from examshield.verify.models.base import FILE, IMAGE, TEXT, ModelClient, PageInput
from examshield.verify.models.google import GoogleModel
from examshield.verify.models.openai import OpenAIModel

KEY = FoundKey(value="sk-test-secret-value-1234", source="환경변수 TEST")
PAGE = PageInput(
    text="1. 문제\n① 가 ② 나", image_png=b"\x89PNG-fake", choice_count=4, pdf=b"%PDF-fake"
)
ANSWER = "1. 문제\n① 가 ② 나\n정답: 2"


def anthropic_body(text: str, stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }


def openai_body(text: str) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test-2026-01-01",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 45,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 165,
        },
    }


def google_body(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "modelVersion": "gemini-test-001",
        # 생각하는 데 쓴 토큰(15)도 출력으로 센다.
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 15,
            "totalTokenCount": 165,
        },
    }


class FakeCloud(BaseHTTPRequestHandler):
    status = 200
    refuse = False
    requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def log_message(self, *args: object) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body, dict(self.headers)))
        if type(self).status != 200:
            reply: dict[str, Any] = {
                "error": {"type": "error", "message": "nope", "code": type(self).status}
            }
            if "/v1/messages" in self.path:
                reply = {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "nope"},
                }
        elif "/v1/messages" in self.path:
            reply = anthropic_body("", "refusal") if type(self).refuse else anthropic_body(ANSWER)
        elif "responses" in self.path:
            reply = openai_body(ANSWER)
        else:
            reply = google_body(ANSWER)
        data = json.dumps(reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def cloud() -> Iterator[str]:
    FakeCloud.status = 200
    FakeCloud.refuse = False
    FakeCloud.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeCloud)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def make(name: str, base: str) -> ModelClient:
    if name == "anthropic":
        return AnthropicModel(KEY, base_url=base)
    if name == "openai":
        return OpenAIModel(KEY, base_url=base + "/v1")
    return GoogleModel(KEY, base_url=base)


VENDORS = ("anthropic", "openai", "google")
VERSIONS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-test-2026-01-01",
    "google": "gemini-test-001",
}


@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("channel", (TEXT, IMAGE, FILE))
def test_each_vendor_answers_on_each_channel_through_its_own_sdk(
    vendor: str, channel: str, cloud: str
) -> None:
    model = make(vendor, cloud)
    assert FakeCloud.requests == []

    reply = model.run(PAGE, channel)
    sent = json.dumps(FakeCloud.requests[-1][1], ensure_ascii=False)

    assert reply.answer == 2
    assert reply.transcription == "1. 문제\n① 가 ② 나"
    assert reply.tokens == (120, 45)
    assert model.info.version == VERSIONS[vendor]
    assert model.info.kind == "cloud"
    assert (base64.b64encode(PAGE.image_png).decode() in sent) == (channel == IMAGE)
    assert (base64.b64encode(PAGE.pdf).decode() in sent) == (channel == FILE)
    assert ("① 가 ② 나" in sent) == (channel == TEXT)


def test_anthropic_request_names_the_model_and_does_not_ask_for_a_fallback(cloud: str) -> None:
    AnthropicModel(KEY, base_url=cloud).run(PAGE, FILE)
    path, body, headers = FakeCloud.requests[-1]

    assert path.startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "high"}
    assert body["messages"][0]["content"][0]["type"] == "document"
    assert "fallbacks" not in body
    assert "temperature" not in body
    assert KEY.value in json.dumps(headers)


def test_a_refusal_is_recorded_as_a_refusal_not_an_answer(cloud: str) -> None:
    FakeCloud.refuse = True
    reply = AnthropicModel(KEY, base_url=cloud).run(PAGE, TEXT)

    assert reply.refused
    assert reply.answer is None


@pytest.mark.parametrize("vendor", VENDORS)
def test_a_rejected_key_stops_that_model(vendor: str, cloud: str) -> None:
    FakeCloud.status = 401

    with pytest.raises(ModelAbortError) as caught:
        make(vendor, cloud).run(PAGE, TEXT)
    assert KEY.value not in str(caught.value)
    assert "nope" not in str(caught.value)


@pytest.mark.parametrize("vendor", VENDORS)
def test_a_bad_request_counts_as_one_failed_item(vendor: str, cloud: str) -> None:
    FakeCloud.status = 400

    with pytest.raises(ModelUnavailableError):
        make(vendor, cloud).run(PAGE, TEXT)


@pytest.mark.parametrize("vendor", VENDORS)
def test_offline_setting_refuses_the_vendor_host_even_with_a_key(
    vendor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OFFLINE_ENV, "1")
    model: ModelClient = {
        "anthropic": AnthropicModel,
        "openai": OpenAIModel,
        "google": GoogleModel,
    }[vendor](KEY)

    with pytest.raises(NetworkRefusedError):
        model.run(PAGE, TEXT)


def test_model_name_can_be_overridden_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMSHIELD_ANTHROPIC_MODEL", "claude-haiku-5-5")

    assert AnthropicModel(KEY).info.name == "claude-haiku-5-5"


@pytest.mark.parametrize("vendor", VENDORS)
def test_sdk_environment_variables_cannot_redirect_the_destination(
    vendor: str, cloud: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_GEMINI_BASE_URL"):
        monkeypatch.setenv(name, cloud)
    monkeypatch.setenv(OFFLINE_ENV, "1")
    model = {"anthropic": AnthropicModel, "openai": OpenAIModel, "google": GoogleModel}[vendor](KEY)

    assert cloud not in model.base_url
    with pytest.raises(NetworkRefusedError):
        model.run(PAGE, TEXT)
    assert FakeCloud.requests == []


@pytest.mark.parametrize("vendor", ("anthropic", "openai"))
def test_a_failed_call_is_not_retried_behind_the_users_back(vendor: str, cloud: str) -> None:
    FakeCloud.status = 429

    with pytest.raises(ModelUnavailableError):
        make(vendor, cloud).run(PAGE, FILE)
    assert len(FakeCloud.requests) == 1
