from dataclasses import dataclass, field
from typing import Protocol

from examshield.verify.metrics import parse_answer, parse_answers, strip_answer_line

TEXT = "text"
IMAGE = "image"
FILE = "file"

PROMPT = (
    "다음은 시험 문항 한 개다.\n"
    "1) 보이는 글자를 빠짐없이 그대로 옮겨 적어라. 설명을 덧붙이지 마라.\n"
    "2) 마지막 줄에 '정답: N' 형식으로 정답 번호 하나를 적어라."
)


MULTI_PROMPT = (
    "다음은 시험지 한 쪽이다. 문항 {numbers}이(가) 차례로 있다.\n"
    "1) 보이는 글자를 빠짐없이 그대로 옮겨 적어라. 설명을 덧붙이지 마라.\n"
    "2) 마지막에 문항마다 한 줄씩 '정답 <문항 번호>: N' 형식으로 정답 번호를 적어라."
)


@dataclass(frozen=True)
class PageInput:
    text: str
    image_png: bytes
    choice_count: int
    pdf: bytes = b""
    # 이 쪽에 있는 문항 번호. 비어 있으면 문항이 하나인 쪽이다.
    questions: tuple[int, ...] = ()


def prompt_of(page: PageInput) -> str:
    """쪽에 문항이 여럿이면 문항 번호를 알려 주고 문항마다 답을 받는다. 응시자도 문항 번호는 안다."""
    if not page.questions:
        return PROMPT
    return MULTI_PROMPT.format(numbers=", ".join(str(number) for number in page.questions))


@dataclass(frozen=True)
class ModelReply:
    transcription: str
    answer: int | None
    refused: bool = False
    # 쪽에 문항이 여럿일 때 문항 번호 → 답.
    answers: dict[int, int] = field(default_factory=dict)
    # 모델이 돌려준 글 그대로. verify --keep-replies로 남길 때만 쓴다.
    raw: str = ""
    # 이 호출에 쓴 토큰 (입력, 출력). 알려 주지 않는 모델은 None.
    tokens: tuple[int, int] | None = None
    # 이 호출에 걸린 시간(초). 받아 둔 응답을 다시 쓸 때 원래 걸린 시간을 전하려고 둔다.
    seconds: float | None = None


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


def token_counts(given: object, produced: object) -> tuple[int, int] | None:
    """응답에 적혀 온 토큰 수. 숫자가 아니면 모르는 것으로 둔다."""
    if isinstance(given, bool) or isinstance(produced, bool):
        return None
    if not isinstance(given, int) or not isinstance(produced, int):
        return None
    return (given, produced) if given >= 0 and produced >= 0 else None


def read_reply(text: str, page: PageInput, tokens: tuple[int, int] | None = None) -> ModelReply:
    """모델이 돌려준 글에서 옮겨 적은 글자와 답을 나눠 읽는다."""
    if page.questions:
        return ModelReply(
            transcription=strip_answer_line(text),
            answer=None,
            answers=parse_answers(text, page.questions, page.choice_count),
            raw=text,
            tokens=tokens,
        )
    return ModelReply(
        transcription=strip_answer_line(text),
        answer=parse_answer(text, page.choice_count),
        raw=text,
        tokens=tokens,
    )
