from difflib import SequenceMatcher

from examshield.benchmark import Item
from examshield.pdfio import normalize_text
from examshield.verify.metrics import parse_answer
from examshield.verify.models.base import TEXT, ModelInfo, ModelReply, PageInput

MATCH_THRESHOLD = 0.6


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

    def run(self, page: PageInput, channel: str) -> ModelReply:
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
