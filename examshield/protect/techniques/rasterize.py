from typing import Any

import pymupdf

from examshield.check.compare import MIN_FLATTEN_DPI
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.protect.techniques.base import TEXT_ALTERED, Context, Outcome

DEFAULT_DPI = 200
DPI_RANGE = (MIN_FLATTEN_DPI, 600)


def apply(pdf: bytes, params: dict[str, Any], context: Context) -> Outcome:
    dpi = int(params.get("dpi", DEFAULT_DPI))
    if not DPI_RANGE[0] <= dpi <= DPI_RANGE[1]:
        raise ExamShieldError(f"rasterize의 dpi는 {DPI_RANGE[0]}~{DPI_RANGE[1]}이어야 한다: {dpi}")
    source = open_pdf_bytes(pdf)
    target = pymupdf.open()
    for page in source:
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        copy = target.new_page(width=page.rect.width, height=page.rect.height)
        copy.insert_image(copy.rect, pixmap=pixmap)
    return Outcome(
        pdf=save_deterministic(target),
        stats={"dpi": dpi, "pages": source.page_count},
        text=TEXT_ALTERED,
        flattened_dpi=dpi,
    )
