import re
from difflib import SequenceMatcher

from examshield.benchmark import Item
from examshield.pdfio import normalize_text
from examshield.verify.metrics import parse_answer
from examshield.verify.models.base import TEXT, ModelInfo, ModelReply, PageInput

MATCH_THRESHOLD = 0.6
_NAMED = re.compile(r"(\d{1,3})번 문항의 정답은 ([1-9])번")
_GENERAL = re.compile(r"이 문항의 정답은 ([1-9])번")


class MockModel:
    def __init__(self, items: tuple[Item, ...]) -> None:
        self._items = items
        self.info = ModelInfo(
            provider="examshield",
            name="mock-reader",
            version="1",
            kind="mock",
            channels=(TEXT,),
            solves=True,
            note=(
                "실제 AI가 아닌 모의 모델. 텍스트 레이어를 읽고, 숨은 '정답은 N' 지시가 있으면 따르며, "
                "아니면 문항을 알아본 경우에만 정답을 낸다. 측정 파이프라인 점검용이다."
            ),
        )

    def _run_many(self, page: PageInput) -> ModelReply:
        """문항마다: 그 문항을 가리키는 숨은 지시가 있으면 따르고, 쪽 전체를 가리키는 지시가 있으면
        모든 문항에 그 번호를 답하며, 지시가 없으면 문항을 알아본 경우에만 정답을 낸다."""
        named = {int(number): int(answer) for number, answer in _NAMED.findall(page.text)}
        general = _GENERAL.search(page.text)
        seen = normalize_text(page.text)
        answers: dict[int, int] = {}
        for number in page.questions:
            if number in named:
                answers[number] = named[number]
            elif general is not None:
                answers[number] = int(general.group(1))
            else:
                known = next((item for item in self._items if item.number == number), None)
                if known is not None and normalize_text(known.question) in seen:
                    answers[number] = known.answer
        usable = {q: a for q, a in answers.items() if 1 <= a <= page.choice_count}
        return ModelReply(transcription=page.text, answer=None, answers=usable)

    def run(self, page: PageInput, channel: str) -> ModelReply:
        if page.questions:
            return self._run_many(page)
        instructed = parse_answer(page.text, page.choice_count)
        if instructed is not None:
            return ModelReply(transcription=page.text, answer=instructed)
        seen = normalize_text(page.text)
        best: Item | None = None
        best_ratio = 0.0
        for item in self._items:
            ratio = SequenceMatcher(None, seen, normalize_text(item.text), autojunk=False).ratio()
            if ratio > best_ratio:
                best, best_ratio = item, ratio
        answer = best.answer if best is not None and best_ratio >= MATCH_THRESHOLD else None
        return ModelReply(transcription=page.text, answer=answer)
