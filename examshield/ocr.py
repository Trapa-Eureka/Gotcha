import shutil
import subprocess
from dataclasses import dataclass

from examshield.errors import ModelUnavailableError

WANTED_LANGUAGES = ("kor", "eng")


@dataclass(frozen=True)
class Tesseract:
    executable: str
    languages: tuple[str, ...]
    missing: tuple[str, ...]
    version: str

    def read(self, image_png: bytes) -> str:
        result = subprocess.run(
            [self.executable, "stdin", "stdout", "-l", "+".join(self.languages), "--psm", "6"],
            input=image_png,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ModelUnavailableError("Tesseract 실행에 실패했다.")
        return result.stdout.decode("utf-8", "replace")


def load_tesseract(executable: str | None = None) -> Tesseract:
    found = executable or shutil.which("tesseract")
    if not found:
        raise ModelUnavailableError("Tesseract가 설치돼 있지 않다.")
    listed = subprocess.run([found, "--list-langs"], capture_output=True, text=True, check=False)
    installed = {line.strip() for line in listed.stdout.splitlines()[1:] if line.strip()}
    languages = tuple(lang for lang in WANTED_LANGUAGES if lang in installed)
    if not languages:
        raise ModelUnavailableError("Tesseract에 kor, eng 언어 데이터가 없다.")
    shown = subprocess.run([found, "--version"], capture_output=True, text=True, check=False)
    first = (shown.stdout or shown.stderr).splitlines()[:1]
    return Tesseract(
        executable=found,
        languages=languages,
        missing=tuple(lang for lang in WANTED_LANGUAGES if lang not in installed),
        version=first[0].replace("tesseract", "").strip() if first else "unknown",
    )


def find_tesseract() -> Tesseract | None:
    try:
        return load_tesseract()
    except ModelUnavailableError:
        return None
