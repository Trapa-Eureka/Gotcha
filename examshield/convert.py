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
# 한글(HWP, HWPX) 문서. LibreOffice가 읽을 수 있는 것만 바뀐다.
HWP_SUFFIXES = (".hwp", ".hwpx")
OFFICE_SUFFIXES = (WORD_SUFFIX, *HWP_SUFFIXES)
SUPPORTED_SUFFIXES = (PDF_SUFFIX, *OFFICE_SUFFIXES)
HWP_NOTICE = (
    "한글 문서는 LibreOffice가 PDF로 바꾼 결과를 원본으로 삼는다. LibreOffice는 한글 문서의 "
    "모양(글꼴, 표, 쪽 나눔)을 그대로 옮기지 못할 수 있다. 복사본을 열어 한글에서 본 것과 "
    "같은지 확인한다. 다르면 한글에서 PDF로 내보낸 파일을 쓴다."
)
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

    Word나 한글 파일일 때만 work_base 아래에 임시 폴더를 만들고, 끝나면 그 폴더만 지운다.
    """
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        with path.open("rb") as handle:
            return handle.read()
    if suffix in OFFICE_SUFFIXES:
        with work_folder(work_base) as work:
            return _office_to_pdf(path, suffix, work)
    raise ExamShieldError(
        f"지원하지 않는 파일 형식이다: {path.name} (PDF, .docx, .hwp, .hwpx만 받는다)"
    )


def needs_review(path: Path) -> bool:
    """변환한 PDF의 모양을 사람이 확인해야 하는 형식인지."""
    return path.suffix.lower() in HWP_SUFFIXES


def _office_to_pdf(path: Path, suffix: str, work: Path) -> bytes:
    office = find_office()
    if office is None:
        raise ExamShieldError(
            f"{suffix} 파일을 PDF로 바꾸려면 LibreOffice가 필요하다. "
            "설치하거나 PDF로 내보낸 파일을 쓴다."
        )
    source = work / f"input{suffix}"
    with path.open("rb") as handle:
        source.write_bytes(handle.read())
    profile = work / "office-profile"
    try:
        result = _run_office(office, profile, work, source)
    except subprocess.TimeoutExpired as error:
        raise ExamShieldError(
            f"LibreOffice가 {CONVERT_TIMEOUT_SECONDS}초 안에 {path.name}을(를) 바꾸지 못했다."
        ) from error
    converted = work / "input.pdf"
    if result.returncode != 0 or not converted.is_file():
        hint = (
            " 한글 문서는 LibreOffice 버전에 따라 열리지 않는다. 한글에서 PDF로 내보낸 파일을 쓴다."
            if suffix in HWP_SUFFIXES
            else ""
        )
        raise ExamShieldError(f"LibreOffice가 {path.name}을(를) PDF로 바꾸지 못했다.{hint}")
    return converted.read_bytes()


def _run_office(
    office: str, profile: Path, work: Path, source: Path
) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(
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
