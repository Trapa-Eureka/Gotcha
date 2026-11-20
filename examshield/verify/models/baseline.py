import shutil
import subprocess

from examshield.errors import ModelUnavailableError
from examshield.verify.models.base import IMAGE, TEXT, ModelInfo, ModelReply, PageInput

WANTED_LANGUAGES = ("kor", "eng")


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
        found = executable or shutil.which("tesseract")
        if not found:
            raise ModelUnavailableError("Tesseract가 설치돼 있지 않다.")
        self._executable = found
        installed = self._installed_languages()
        self._languages = [lang for lang in WANTED_LANGUAGES if lang in installed]
        if not self._languages:
            raise ModelUnavailableError("Tesseract에 kor, eng 언어 데이터가 없다.")
        missing = [lang for lang in WANTED_LANGUAGES if lang not in installed]
        note = "OCR 기준선."
        if missing:
            note += (
                f" 언어 데이터 {', '.join(missing)}이(가) 없어 해당 글자의 판독률이 낮게 나온다."
            )
        self.info = ModelInfo(
            provider="tesseract",
            name="tesseract-" + "+".join(self._languages),
            version=self._version(),
            kind="baseline",
            channels=(IMAGE,),
            solves=False,
            note=note,
        )

    def run(self, page: PageInput, channel: str) -> ModelReply:
        result = subprocess.run(
            [self._executable, "stdin", "stdout", "-l", "+".join(self._languages), "--psm", "6"],
            input=page.image_png,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ModelUnavailableError("Tesseract 실행에 실패했다.")
        return ModelReply(transcription=result.stdout.decode("utf-8", "replace"), answer=None)

    def _installed_languages(self) -> set[str]:
        result = subprocess.run(
            [self._executable, "--list-langs"], capture_output=True, text=True, check=False
        )
        return {line.strip() for line in result.stdout.splitlines()[1:] if line.strip()}

    def _version(self) -> str:
        result = subprocess.run(
            [self._executable, "--version"], capture_output=True, text=True, check=False
        )
        first = (result.stdout or result.stderr).splitlines()[:1]
        return first[0].replace("tesseract", "").strip() if first else "unknown"
