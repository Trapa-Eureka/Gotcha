import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KOREAN = (ROOT / "README.md").read_text(encoding="utf-8")
ENGLISH = (ROOT / "README.en.md").read_text(encoding="utf-8")


def shape(text: str) -> dict[str, int]:
    lines = text.splitlines()
    return {
        "h2": sum(line.startswith("## ") for line in lines),
        "h3": sum(line.startswith("### ") for line in lines),
        "table_rows": sum(line.startswith("|") for line in lines),
        "bullets": sum(line.startswith("- ") for line in lines),
        "done": len(re.findall(r"^- \[x\]", text, re.M)),
        "todo": len(re.findall(r"^- \[ \]", text, re.M)),
        "code_fences": text.count("```"),
    }


def test_both_readmes_have_the_same_structure() -> None:
    assert shape(KOREAN) == shape(ENGLISH)


def test_readmes_link_to_each_other() -> None:
    assert "[English](README.en.md)" in KOREAN
    assert "[한국어](README.md)" in ENGLISH
