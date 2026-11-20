import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from examshield.errors import ExamShieldError

WORK_FOLDER_PREFIX = ".examshield-tmp-"
PDF_SUFFIX = ".pdf"
WORD_SUFFIX = ".docx"
SUPPORTED_SUFFIXES = (PDF_SUFFIX, WORD_SUFFIX)
CONVERT_TIMEOUT_SECONDS = 180


@contextmanager
def work_folder(base: Path) -> Iterator[Path]:
    base.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=WORK_FOLDER_PREFIX, dir=base))
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def find_office() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def read_document(path: Path, work_base: Path) -> bytes:
    """원본을 읽기 전용으로 열어 PDF 바이트로 돌려준다.

    Word 파일일 때만 work_base 아래에 임시 폴더를 만들고, 끝나면 그 폴더만 지운다.
    """
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        with path.open("rb") as handle:
            return handle.read()
    if suffix == WORD_SUFFIX:
        with work_folder(work_base) as work:
            return _word_to_pdf(path, work)
    raise ExamShieldError(f"지원하지 않는 파일 형식이다: {path.name} (PDF, .docx만 받는다)")


def _word_to_pdf(path: Path, work: Path) -> bytes:
    office = find_office()
    if office is None:
        raise ExamShieldError(
            "Word 파일을 PDF로 바꾸려면 LibreOffice가 필요하다. 설치하거나 PDF로 내보낸 파일을 쓴다."
        )
    source = work / f"input{WORD_SUFFIX}"
    with path.open("rb") as handle:
        source.write_bytes(handle.read())
    profile = work / "office-profile"
    result = subprocess.run(
        [
            office,
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(work),
            str(source),
        ],
        capture_output=True,
        check=False,
        timeout=CONVERT_TIMEOUT_SECONDS,
    )
    converted = work / "input.pdf"
    if result.returncode != 0 or not converted.is_file():
        raise ExamShieldError(f"LibreOffice가 {path.name}을(를) PDF로 바꾸지 못했다.")
    return converted.read_bytes()
