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
    return ItemSet(
        version=str(data["version"]), sha256=hashlib.sha256(raw).hexdigest(), items=items
    )


def render_items_pdf(items: tuple[Item, ...]) -> bytes:
    font = pymupdf.Font("korea")
    doc = pymupdf.open()
    for item in items:
        page = doc.new_page()
        page.insert_font(fontname="kr", fontbuffer=font.buffer)
        leftover = page.insert_textbox(
            pymupdf.Rect(*PAGE_RECT), item.text, fontname="kr", fontsize=14, lineheight=1.8
        )
        if leftover < 0:
            raise ExamShieldError(f"문항 {item.id}이(가) 한 쪽에 들어가지 않는다.")
    return save_deterministic(doc)
