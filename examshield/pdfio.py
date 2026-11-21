import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]

# MuPDF가 깨진 파일을 만나면 표준 출력에 직접 쓰는 진단 글을 끈다. 오류는 예외로 받는다.
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

_WHITESPACE = re.compile(r"\s+")


def open_pdf_bytes(data: bytes) -> Any:
    return pymupdf.open(stream=data, filetype="pdf")


def read_pdf(path: Path) -> Any:
    with path.open("rb") as handle:
        return open_pdf_bytes(handle.read())


def own_contents(doc: Any, page: Any) -> None:
    """쪽의 내용 줄기 목록을 이 쪽만 쓰게 한다. 쪽에 무언가를 그려 넣기 전에 부른다.

    펼침면을 쪽 여럿으로 나눈 문서는 쪽들이 내용 줄기 목록 하나를 같이 쓴다. 그대로 그려 넣으면
    한 쪽에 넣은 것이 같이 쓰는 모든 쪽에 들어간다.
    """
    kind, value = doc.xref_get_key(page.xref, "Contents")
    if kind != "xref":
        return
    shared = doc.xref_object(int(value.split()[0]), compressed=True)
    if shared.lstrip().startswith("["):
        doc.xref_set_key(page.xref, "Contents", shared)


def save_deterministic(doc: Any) -> bytes:
    doc.set_metadata({})
    return bytes(doc.tobytes(garbage=4, deflate=True, no_new_id=True))


def page_text(page: Any) -> str:
    return str(page.get_text())


def render_gray(page: Any, dpi: int) -> GrayImage:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    flat = np.frombuffer(pix.samples, dtype=np.uint8)
    rows = flat.reshape(pix.height, pix.stride)
    return np.ascontiguousarray(rows[:, : pix.width])


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def render_rgb(page: Any, dpi: int) -> NDArray[np.uint8]:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    flat = np.frombuffer(pix.samples, dtype=np.uint8)
    rows = flat.reshape(pix.height, pix.stride)
    return np.ascontiguousarray(rows[:, : pix.width * 3]).reshape(pix.height, pix.width, 3)
