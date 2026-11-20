import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from examshield.errors import ExamShieldError
from examshield.pdfio import save_deterministic

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_VERSION = "v1"
CHOICE_MARKS = "①②③④⑤"
PAGE_RECT = (72.0, 90.0, 523.0, 750.0)


@dataclass(frozen=True)
class Item:
    id: str
    number: int
    question: str
    choices: tuple[str, ...]
    answer: int

    @property
    def text(self) -> str:
        lines = [f"{self.number}. {self.question}"]
        lines += [f"{CHOICE_MARKS[i]} {choice}" for i, choice in enumerate(self.choices)]
        return "\n".join(lines)


@dataclass(frozen=True)
class ItemSet:
    version: str
    sha256: str
    items: tuple[Item, ...]
    # 한 쪽에 넣는 문항 수.
    per_page: int = 1

    @property
    def pages(self) -> tuple[tuple[Item, ...], ...]:
        size = self.per_page
        return tuple(self.items[start : start + size] for start in range(0, len(self.items), size))


def load_items(version: str = DEFAULT_VERSION) -> ItemSet:
    path = BENCHMARK_DIR / version / "items.yaml"
    if not path.is_file():
        raise ExamShieldError(f"문항 세트 {version}을(를) 찾지 못했다.")
    raw = path.read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw)
    items = tuple(
        Item(
            id=str(entry["id"]),
            number=number,
            question=str(entry["question"]),
            choices=tuple(str(choice) for choice in entry["choices"]),
            answer=int(entry["answer"]),
        )
        for number, entry in enumerate(data["items"], start=1)
    )
    per_page = int(data.get("per_page", 1))
    if not 1 <= per_page <= 6:
        raise ExamShieldError("문항 세트의 per_page는 1~6이어야 한다.")
    return ItemSet(
        version=str(data["version"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        items=items,
        per_page=per_page,
    )


def page_text_of(items: tuple[Item, ...]) -> str:
    return "\n\n".join(item.text for item in items)


def render_items_pdf(items: tuple[Item, ...], per_page: int = 1) -> bytes:
    font = pymupdf.Font("korea")
    doc = pymupdf.open()
    # 한 쪽에 여러 문항을 넣을 때는 글자를 조금 줄여야 들어간다.
    size, spacing = (14, 1.8) if per_page == 1 else (12, 1.6)
    for start in range(0, len(items), per_page):
        group = items[start : start + per_page]
        page = doc.new_page()
        page.insert_font(fontname="kr", fontbuffer=font.buffer)
        leftover = page.insert_textbox(
            pymupdf.Rect(*PAGE_RECT),
            page_text_of(group),
            fontname="kr",
            fontsize=size,
            lineheight=spacing,
        )
        if leftover < 0:
            raise ExamShieldError(f"문항 {group[0].id}부터가 한 쪽에 들어가지 않는다.")
    # 쓰는 글자만 남긴다. 쪽 하나를 통째로 모델에 올릴 때 파일이 작아야 한다.
    doc.subset_fonts()
    return save_deterministic(doc)
