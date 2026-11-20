import re
from difflib import SequenceMatcher

from examshield.pdfio import normalize_text

_ANSWER_LINE = re.compile(r"정답\s*(?:은|:|：)?\s*([1-9①-⑨])")
_CIRCLED = {mark: number for number, mark in enumerate("①②③④⑤⑥⑦⑧⑨", start=1)}


def legibility(transcription: str, truth: str) -> float:
    left = normalize_text(transcription)
    right = normalize_text(truth)
    if not right:
        return 1.0 if not left else 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def parse_answer(reply: str, choice_count: int) -> int | None:
    matches = _ANSWER_LINE.findall(reply)
    if not matches:
        return None
    token = matches[-1]
    number = _CIRCLED.get(token) or int(token)
    return number if 1 <= number <= choice_count else None


def strip_answer_line(reply: str) -> str:
    lines = [line for line in reply.splitlines() if not _ANSWER_LINE.search(line)]
    return "\n".join(lines)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
