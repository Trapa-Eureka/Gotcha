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
        # 그림의 화소가 그 해상도의 화면 화소와 하나씩 맞게 놓는다. 쪽 크기는 화소로 딱 떨어지지
        # 않는 일이 많아서, 그림을 쪽에 꼭 맞게 줄이면 다시 그릴 때 글자 가장자리가 달라진다.
        # 그림은 쪽보다 한 화소 미만으로 넘치고, 넘친 부분은 쪽 밖이라 보이지 않는다.
        corner = copy.rect.tl
        area = pymupdf.Rect(
            corner.x,
            corner.y,
            corner.x + pixmap.width * 72.0 / dpi,
            corner.y + pixmap.height * 72.0 / dpi,
        )
        copy.insert_image(area, pixmap=pixmap, keep_proportion=False)
    return Outcome(
        pdf=save_deterministic(target),
        stats={"dpi": dpi, "pages": source.page_count},
        text=TEXT_ALTERED,
        flattened_dpi=dpi,
    )
