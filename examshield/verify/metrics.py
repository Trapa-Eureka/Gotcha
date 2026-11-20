import re
from collections import Counter
from difflib import SequenceMatcher

from examshield.pdfio import normalize_text

LONG_TEXT = 1000
_ANSWER_LINE = re.compile(r"정답\s*(?:은|:|：)?\s*([1-9①-⑨])")
_NUMBERED_ANSWER = re.compile(r"정답\s*(\d{1,3})\s*번?\s*[:：]\s*([1-9①-⑨])")
_CIRCLED = {mark: number for number, mark in enumerate("①②③④⑤⑥⑦⑧⑨", start=1)}


def legibility(transcription: str, truth: str) -> float:
    left = normalize_text(transcription)
    right = normalize_text(truth)
    if not right:
        return 1.0 if not left else 0.0
    if max(len(left), len(right)) > LONG_TEXT:
        return _pair_overlap(left, right)
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _pair_overlap(left: str, right: str) -> float:
    """긴 글용 근사값. 이웃한 두 글자 묶음이 얼마나 겹치는지 본다(순서는 보지 않는다)."""
    first = Counter(left[index : index + 2] for index in range(len(left) - 1))
    second = Counter(right[index : index + 2] for index in range(len(right) - 1))
    total = sum(first.values()) + sum(second.values())
    return 2 * sum((first & second).values()) / total if total else 0.0


def parse_answer(reply: str, choice_count: int) -> int | None:
    matches = _ANSWER_LINE.findall(reply)
    if not matches:
        return None
    token = matches[-1]
    number = _CIRCLED.get(token) or int(token)
    return number if 1 <= number <= choice_count else None


def parse_answers(reply: str, questions: tuple[int, ...], choice_count: int) -> dict[int, int]:
    """'정답 12: 3' 꼴의 줄에서 문항별 답을 읽는다. 같은 문항이 여러 번 나오면 마지막 것을 쓴다."""
    found: dict[int, int] = {}
    for number, token in _NUMBERED_ANSWER.findall(reply):
        answer = _CIRCLED.get(token) or int(token)
        if int(number) in questions and 1 <= answer <= choice_count:
            found[int(number)] = answer
    return found


def strip_answer_line(reply: str) -> str:
    lines = [line for line in reply.splitlines() if not _ANSWER_LINE.search(line)]
    return "\n".join(lines)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
