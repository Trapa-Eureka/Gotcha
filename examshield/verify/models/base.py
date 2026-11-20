from dataclasses import dataclass
from typing import Protocol

TEXT = "text"
IMAGE = "image"
FILE = "file"

PROMPT = (
    "다음은 시험 문항 한 개다.\n"
    "1) 보이는 글자를 빠짐없이 그대로 옮겨 적어라. 설명을 덧붙이지 마라.\n"
    "2) 마지막 줄에 '정답: N' 형식으로 정답 번호 하나를 적어라."
)


@dataclass(frozen=True)
class PageInput:
    text: str
    image_png: bytes
    choice_count: int
    pdf: bytes = b""


@dataclass(frozen=True)
class ModelReply:
    transcription: str
    answer: int | None
    refused: bool = False


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    name: str
    version: str
    kind: str
    channels: tuple[str, ...]
    solves: bool
    note: str = ""


class ModelClient(Protocol):
    info: ModelInfo

    def run(self, page: PageInput, channel: str) -> ModelReply: ...
