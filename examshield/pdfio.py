import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]

_WHITESPACE = re.compile(r"\s+")


def open_pdf_bytes(data: bytes) -> Any:
    return pymupdf.open(stream=data, filetype="pdf")


def read_pdf(path: Path) -> Any:
    with path.open("rb") as handle:
        return open_pdf_bytes(handle.read())


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
