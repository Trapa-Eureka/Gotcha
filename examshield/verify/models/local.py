import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from examshield.errors import ModelUnavailableError, NetworkRefusedError
from examshield.netguard import OFFLINE_ENV, offline_enabled, require_network
from examshield.verify.models.base import (
    IMAGE,
    TEXT,
    ModelInfo,
    ModelReply,
    PageInput,
    prompt_of,
    read_reply,
    token_counts,
)

DEFAULT_HOST = "http://127.0.0.1:11434"
MODEL_ENV = "EXAMSHIELD_OLLAMA_MODEL"
HOST_ENV = "OLLAMA_HOST"
TIMEOUT_SECONDS = 300


def ollama_base_url() -> str:
    host = os.environ.get(HOST_ENV, "").strip() or DEFAULT_HOST
    return host if "://" in host else f"http://{host}"


def is_relayed(name: str, described: dict[str, Any]) -> bool:
    """Ollama가 이 PC에서 돌리지 않고 외부 서버로 넘기는 모델인지 본다."""
    if described.get("remote_host") or described.get("remote_model"):
        return True
    return name.endswith(("-cloud", ":cloud"))


class OllamaModel:
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self._base_url = (base_url or ollama_base_url()).rstrip("/")
        installed = self._call("GET", "/api/tags").get("models") or []
        if not installed:
            raise ModelUnavailableError(
                "Ollama에 설치된 모델이 없다. `ollama pull <모델>`로 받는다."
            )
        wanted = model or os.environ.get(MODEL_ENV, "").strip() or str(installed[0]["name"])
        entry = next((m for m in installed if m.get("name") == wanted), None)
        if entry is None:
            names = ", ".join(str(m.get("name")) for m in installed)
            raise ModelUnavailableError(
                f"Ollama에 모델 '{wanted}'이(가) 없다. 설치된 모델: {names}"
            )
        self._model = wanted
        shown = self._call("POST", "/api/show", {"model": wanted})
        relayed = is_relayed(wanted, entry) or is_relayed(wanted, shown)
        if relayed and offline_enabled():
            raise NetworkRefusedError(
                f"오프라인 설정({OFFLINE_ENV})이 켜져 있어 Ollama 모델 '{wanted}'을(를) 거부했다. "
                "이 모델은 이 PC에서 돌지 않고 Ollama가 외부 서버로 중계한다."
            )
        sees = "vision" in (shown.get("capabilities") or [])
        digest = str(entry.get("digest", ""))[:12] or "unknown"
        self.info = ModelInfo(
            provider="ollama",
            name=wanted,
            version=digest,
            kind="relayed" if relayed else "local",
            channels=(TEXT, IMAGE) if sees else (TEXT,),
            solves=True,
            note=" ".join(
                part
                for part in (
                    "" if sees else "비전 기능이 없는 모델이라 이미지 경로는 측정하지 않았다.",
                    "Ollama가 외부 서버로 중계하는 모델이다. 입력이 이 PC 밖으로 나간다."
                    if relayed
                    else "",
                )
                if part
            ),
        )

    def run(self, page: PageInput, channel: str) -> ModelReply:
        body: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }
        if channel == IMAGE:
            body["prompt"] = prompt_of(page)
            body["images"] = [base64.b64encode(page.image_png).decode("ascii")]
        else:
            body["prompt"] = f"{prompt_of(page)}\n\n---\n{page.text}"
        answer = self._call("POST", "/api/generate", body)
        tokens = token_counts(answer.get("prompt_eval_count"), answer.get("eval_count"))
        return read_reply(str(answer.get("response", "")), page, tokens)

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._base_url + path
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ModelUnavailableError(f"Ollama 주소가 잘못됐다: {self._base_url}")
        require_network(parts.hostname, "Ollama 로컬 모델")
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(  # noqa: S310
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
                parsed: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return parsed
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise ModelUnavailableError(
                f"Ollama({self._base_url})에 연결하지 못했다. `ollama serve`가 떠 있는지 확인한다."
            ) from error
