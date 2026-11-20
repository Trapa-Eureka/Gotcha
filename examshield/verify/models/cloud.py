import base64
import os
from urllib.parse import urlsplit

from examshield.auth.keys import FoundKey
from examshield.netguard import require_network
from examshield.providers import Provider
from examshield.verify.models.base import (
    FILE,
    IMAGE,
    TEXT,
    ModelInfo,
    ModelReply,
    PageInput,
    prompt_of,
    read_reply,
)

CHANNELS = (TEXT, IMAGE, FILE)
NO_RESPONSE_YET = "응답 없음"


def chosen_model(provider: Provider) -> str:
    return os.environ.get(provider.model_env, "").strip() or provider.default_model


def prompt_for(page: PageInput, channel: str) -> str:
    prompt = prompt_of(page)
    return f"{prompt}\n\n---\n{page.text}" if channel == TEXT else prompt


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def reply_from(text: str, page: PageInput) -> ModelReply:
    return read_reply(text, page)


def abort_reason(company: str, status: int) -> str:
    """서버가 보낸 문구는 옮기지 않는다. 키 조각이 섞여 올 수 있다."""
    meaning = {
        401: "API 키가 거부됐다",
        403: "이 키로는 권한이 없다",
        404: "모델 이름을 찾지 못했다",
    }
    return f"{company}: {meaning.get(status, '호출을 계속할 수 없다')} (HTTP {status})."


REFUSED = ModelReply(transcription="", answer=None, refused=True)


class CloudModel:
    """세 제공사 연결의 공통 부분. 만들 때는 네트워크를 쓰지 않는다. 호출 직전에 오프라인 설정을 본다."""

    def __init__(self, provider: Provider, key: FoundKey, model: str | None, base_url: str | None):
        self.provider = provider
        self.model = model or chosen_model(provider)
        self.key_source = key.source
        # 접속 주소를 항상 직접 정한다. SDK가 환경변수(OPENAI_BASE_URL 등)를 읽어
        # 화면에 보여준 회사와 다른 곳으로 보내는 일을 막는다.
        self.base_url = base_url or provider.base_url
        self._host = urlsplit(self.base_url).hostname or ""
        self._version = NO_RESPONSE_YET

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider.id,
            name=self.model,
            version=self._version,
            kind="cloud",
            channels=CHANNELS,
            solves=True,
            note="버전은 응답에 적혀 온 모델 이름이다. 같은 이름의 모델도 예고 없이 바뀔 수 있다.",
        )

    def before_call(self) -> None:
        require_network(self._host, f"{self.provider.company} 클라우드 모델")

    def saw_version(self, version: object) -> None:
        if version:
            self._version = str(version)


__all__ = ["CHANNELS", "FILE", "IMAGE", "REFUSED", "TEXT", "CloudModel"]
