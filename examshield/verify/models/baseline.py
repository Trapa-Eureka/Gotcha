from examshield.ocr import load_tesseract
from examshield.verify.models.base import IMAGE, TEXT, ModelInfo, ModelReply, PageInput


class TextLayerModel:
    def __init__(self) -> None:
        self.info = ModelInfo(
            provider="examshield",
            name="text-layer",
            version="1",
            kind="baseline",
            channels=(TEXT,),
            solves=False,
            note="PDF의 텍스트 레이어를 그대로 추출한 결과. 복사해서 붙여넣는 공격자가 얻는 글자다.",
        )

    def run(self, page: PageInput, channel: str) -> ModelReply:
        return ModelReply(transcription=page.text, answer=None)


class TesseractModel:
    def __init__(self, executable: str | None = None) -> None:
        self._engine = load_tesseract(executable)
        note = "OCR 기준선."
        if self._engine.missing:
            missing = ", ".join(self._engine.missing)
            note += f" 언어 데이터 {missing}이(가) 없어 해당 글자의 판독률이 낮게 나온다."
        self.info = ModelInfo(
            provider="tesseract",
            name="tesseract-" + "+".join(self._engine.languages),
            version=self._engine.version,
            kind="baseline",
            channels=(IMAGE,),
            solves=False,
            note=note,
        )

    def run(self, page: PageInput, channel: str) -> ModelReply:
        return ModelReply(transcription=self._engine.read(page.image_png), answer=None)
