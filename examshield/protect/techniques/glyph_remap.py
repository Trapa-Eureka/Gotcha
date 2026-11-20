import re
from typing import Any

from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.protect.seed import SeedStream
from examshield.protect.techniques.base import TEXT_ALTERED, TEXT_PRESERVED, Context, Outcome

Mapping = dict[bytes, bytes]

_TOKEN = re.compile(rb"<([0-9A-Fa-f\s]*)>|(\[)|(\])")
_SECTION = re.compile(rb"begin(bfchar|bfrange)(.*?)end(?:bfchar|bfrange)", re.S)
_CODESPACE = re.compile(rb"begincodespacerange(.*?)endcodespacerange", re.S)
_COMMENT = re.compile(rb"%[^\r\n]*")
_BLOCK = 100
_PROBE_COLUMNS = 256
_PROBE_CELL = 16
_CLASSES: tuple[tuple[str, int, int], ...] = (
    ("hangul", 0xAC00, 0xD7A3),
    ("digit", 0x30, 0x39),
    ("upper", 0x41, 0x5A),
    ("lower", 0x61, 0x7A),
    ("circled", 0x2460, 0x2473),
)


def parse_tounicode(cmap: bytes) -> Mapping:
    """읽지 못하는 형식(다른 CMap을 끌어다 쓰는 usecmap)은 빈 결과를 낸다. 그러면 호출자가 글꼴을 직접 조사한다."""
    cmap = _COMMENT.sub(b"", cmap)
    if b"usecmap" in cmap:
        return {}
    mapping: Mapping = {}
    for section in _SECTION.finditer(cmap):
        tokens = _tokens(section.group(2))
        if section.group(1) == b"bfchar":
            for code, target in zip(tokens[0::2], tokens[1::2], strict=False):
                if isinstance(code, bytes) and isinstance(target, bytes):
                    mapping[code] = target
        else:
            _read_ranges(tokens, mapping)
    return mapping


def _tokens(body: bytes) -> list[bytes | list[bytes]]:
    tokens: list[bytes | list[bytes]] = []
    group: list[bytes] | None = None
    for match in _TOKEN.finditer(body):
        if match.group(2):
            group = []
        elif match.group(3):
            if group is not None:
                tokens.append(group)
            group = None
        else:
            digits = b"".join(match.group(1).split())
            value = bytes.fromhex((digits + b"0" * (len(digits) % 2)).decode("ascii"))
            if group is not None:
                group.append(value)
            else:
                tokens.append(value)
    return tokens


def _read_ranges(tokens: list[bytes | list[bytes]], mapping: Mapping) -> None:
    for start, end, target in zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=False):
        if not isinstance(start, bytes) or not isinstance(end, bytes) or len(start) != len(end):
            continue
        first, last = int.from_bytes(start, "big"), int.from_bytes(end, "big")
        if last < first or last - first > 0xFFFF:
            continue
        for offset in range(last - first + 1):
            code = (first + offset).to_bytes(len(start), "big")
            if isinstance(target, list):
                if offset < len(target):
                    mapping[code] = target[offset]
            elif target:
                value = (int.from_bytes(target, "big") + offset) % (1 << (8 * len(target)))
                mapping[code] = value.to_bytes(len(target), "big")


def _class_of(target: bytes) -> str | None:
    if len(target) != 2:
        return None
    point = int.from_bytes(target, "big")
    for name, low, high in _CLASSES:
        if low <= point <= high:
            return name
    return None


def scramble(mapping: Mapping, stream: SeedStream) -> tuple[Mapping, int]:
    groups: dict[str, set[bytes]] = {}
    for target in mapping.values():
        name = _class_of(target)
        if name is not None:
            groups.setdefault(name, set()).add(target)
    swap: dict[bytes, bytes] = {}
    for name, _, _ in _CLASSES:
        targets = sorted(groups.get(name, ()))
        if len(targets) < 2:
            continue
        order = stream.shuffled(targets)
        for index, target in enumerate(order):
            swap[target] = order[(index + 1) % len(order)]
    changed = sum(1 for target in mapping.values() if target in swap)
    return {code: swap.get(target, target) for code, target in mapping.items()}, changed


def build_tounicode(mapping: Mapping, original: bytes) -> bytes:
    codespace = _CODESPACE.search(original)
    if codespace:
        ranges = b" ".join(codespace.group(1).split())
        count = len(_TOKEN.findall(codespace.group(1))) // 2
        space = b"%d begincodespacerange\n%s\nendcodespacerange\n" % (count, ranges)
    else:
        widths = sorted({len(code) for code in mapping}) or [2]
        space = b"%d begincodespacerange\n%sendcodespacerange\n" % (
            len(widths),
            b"".join(b"<%s> <%s>\n" % (b"00" * width, b"FF" * width) for width in widths),
        )
    lines = [
        b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n",
        space,
    ]
    entries = sorted(mapping.items(), key=lambda entry: (len(entry[0]), entry[0]))
    for start in range(0, len(entries), _BLOCK):
        block = entries[start : start + _BLOCK]
        lines.append(b"%d beginbfchar\n" % len(block))
        for code, target in block:
            lines.append(
                b"<%s> <%s>\n" % (code.hex().upper().encode(), target.hex().upper().encode())
            )
        lines.append(b"endbfchar\n")
    lines.append(b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n")
    return b"".join(lines)


def probe_font(pdf: bytes, font_xref: int, code_bytes: int) -> Mapping:
    """ToUnicode가 없는 글꼴의 코드별 유니코드를 MuPDF에 직접 물어서 얻는다.

    글꼴만 쓰는 시험용 쪽을 사본에 만들어 코드를 하나씩 찍고, 추출 결과를 읽는다.
    사본은 버리므로 결과 파일에는 흔적이 남지 않는다.
    """
    doc = open_pdf_bytes(pdf)
    rows = 1 if code_bytes == 1 else _PROBE_COLUMNS
    width, height = _PROBE_COLUMNS * _PROBE_CELL, rows * _PROBE_CELL
    page = doc.new_page(width=width, height=height)
    parts = [b"BT /Fp 1 Tf\n"]
    for code in range(_PROBE_COLUMNS * rows):
        x = (code % _PROBE_COLUMNS) * _PROBE_CELL + _PROBE_CELL // 4
        y = height - (code // _PROBE_COLUMNS) * _PROBE_CELL - _PROBE_CELL // 2
        parts.append(b"1 0 0 1 %d %d Tm <%0*X> Tj\n" % (x, y, code_bytes * 2, code))
    parts.append(b"ET\n")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, b"".join(parts))
    doc.xref_set_key(page.xref, "Resources", f"<</Font<</Fp {font_xref} 0 R>>>>")
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    number = doc.page_count - 1
    doc = open_pdf_bytes(doc.tobytes())
    cells: dict[int, list[int]] = {}
    for span in doc[number].get_texttrace():
        for char in span["chars"]:
            column = int(char[2][0] // _PROBE_CELL)
            row = int(char[2][1] // _PROBE_CELL)
            if 0 <= column < _PROBE_COLUMNS and 0 <= row < rows:
                cells.setdefault(row * _PROBE_COLUMNS + column, []).append(int(char[0]))
    mapping: Mapping = {}
    for code, found in cells.items():
        if len(found) == 1 and 0x20 <= found[0] <= 0xFFFF and found[0] != 0xFFFD:
            mapping[code.to_bytes(code_bytes, "big")] = found[0].to_bytes(2, "big")
    return mapping


def apply(pdf: bytes, params: dict[str, Any], context: Context) -> Outcome:
    doc = open_pdf_bytes(pdf)
    fonts: list[int] = []
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "Type")[1] != "/Font":
            continue
        if doc.xref_get_key(xref, "Subtype")[1] in ("/CIDFontType0", "/CIDFontType2"):
            continue
        fonts.append(xref)
    done: dict[int, int] = {}
    scrambled = 0
    changed = 0
    unreadable = 0
    for index, xref in enumerate(fonts):
        kind, value = doc.xref_get_key(xref, "ToUnicode")
        existing = int(value.split()[0]) if kind == "xref" else None
        if existing in done:
            scrambled += 1
            changed += done[existing]
            continue
        original = doc.xref_stream(existing) if existing is not None else None
        mapping = parse_tounicode(original) if original else {}
        if not mapping:
            composite = doc.xref_get_key(xref, "Subtype")[1] == "/Type0"
            mapping = probe_font(pdf, xref, 2 if composite else 1)
        if not mapping:
            unreadable += 1
            continue
        mapping, count = scramble(mapping, SeedStream(context.seed, f"glyph_remap:{index}"))
        if count == 0:
            continue
        target = existing
        if target is None:
            target = doc.get_new_xref()
            doc.update_object(target, "<<>>")
            doc.xref_set_key(xref, "ToUnicode", f"{target} 0 R")
        else:
            done[target] = count
        doc.update_stream(target, build_tounicode(mapping, original or b""))
        scrambled += 1
        changed += count
    return Outcome(
        pdf=save_deterministic(doc),
        stats={
            "fonts": len(fonts),
            "fonts_scrambled": scrambled,
            "fonts_without_readable_codes": unreadable,
            "codes_changed": changed,
        },
        text=TEXT_ALTERED if changed else TEXT_PRESERVED,
    )
