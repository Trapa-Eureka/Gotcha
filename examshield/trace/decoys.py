"""답안이 숨은 미끼와 얼마나 일치하는지 본다.

텍스트를 복사해 AI에 넣은 응시자는 문제를 못 읽은 AI가 미끼 번호를 그대로 답하므로, 답안이 미끼와
여러 문항에서 일치한다(Claude로 잰 결과 12문항 중 12문항). 이 모듈은 그 일치가 우연일 확률을 낸다.
근거 자료일 뿐 증명이 아니다. 화면 캡처나 파일 올리기로 푼 경우는 미끼를 따르지 않아 잡히지 않는다.
"""

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examshield.errors import ExamShieldError
from examshield.protect.pipeline import AnswerKey

MAX_CHOICE = 9
DEFAULT_CHOICES = 4
# 다른 응시자들이 그 번호를 고른 비율로 우연의 확률을 잡으려면 이만큼은 있어야 한다.
MIN_COHORT = 20
DEFAULT_ALPHA = 0.001
MIN_ITEMS = 5
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


Item = tuple[int, int | None]
# (쪽, 문항 번호 또는 None) → 미끼 번호. 문항 번호가 None인 것은 그 쪽 전체에 하나 넣은 미끼다.
DecoyTable = dict[Item, int]


@dataclass(frozen=True)
class Response:
    examinee_id: str
    page: int
    answer: int
    question: int | None = None

    @property
    def item(self) -> Item:
        return (self.page, self.question)


@dataclass(frozen=True)
class Finding:
    examinee_id: str
    items: int
    matches: int
    # 미끼와 무관하게 답했을 때 이만큼 이상 일치할 확률의 위쪽 한계.
    chance: float
    flagged: bool


@dataclass(frozen=True)
class Borrowed:
    """어떤 응시자의 답이 다른 응시자의 복사본에 든 미끼와 일치한 경우."""

    examinee_id: str
    source_id: str
    items: int
    matches: int
    chance: float


@dataclass(frozen=True)
class Analysis:
    findings: tuple[Finding, ...]
    borrowed: tuple[Borrowed, ...]
    dropped: int
    unmatched: int
    per_examinee: bool
    base: str
    threshold: float
    notes: tuple[str, ...]


def read_decoys(job_file: Path) -> tuple[DecoyTable, bool]:
    """작업 기록에서 미끼 번호와 정답표를 썼는지를 읽는다."""
    try:
        record: dict[str, Any] = json.loads(job_file.read_text(encoding="utf-8"))
        table: DecoyTable = {}
        if record.get("decoy_items"):
            for entry in record["decoy_items"]:
                question = entry.get("question")
                key = (int(entry["page"]), None if question is None else int(question))
                table[key] = int(entry["answer"])
        else:
            # 문항별 미끼가 생기기 전에 만든 기록. 쪽마다 미끼가 하나다.
            for page, answer in enumerate(record["decoy_answers"], start=1):
                table[(page, None)] = int(answer)
        if not table or any(not 1 <= value <= MAX_CHOICE for value in table.values()):
            raise ValueError("미끼가 없거나 번호가 범위를 벗어났다")
        return table, bool(record.get("answer_key", False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
        raise ExamShieldError(
            f"작업 기록에서 미끼 번호를 읽지 못했다: {job_file}. "
            "미끼가 든 프로파일(a-remap-decoy, a-full)로 만든 복사본의 .job.json을 준다."
        ) from error


def read_responses(path: Path) -> list[Response]:
    """답안 CSV를 읽는다. 한 줄에 '수험번호,쪽,답' 또는 '수험번호,쪽,문항,답'.

    답이 비었거나 숫자가 아니면 무응답으로 본다.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ExamShieldError(f"답안 파일을 읽지 못했다: {path}") from error
    responses: list[Response] = []
    widths: set[int] = set()
    for number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        if len(cells) not in (3, 4):
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: '수험번호,쪽,답' 또는 '수험번호,쪽,문항,답' "
                "형식이어야 한다. 예: 2026-0001,3,2"
            )
        if number == 1 and not cells[1].isdigit():
            continue
        widths.add(len(cells))
        if not _ID.fullmatch(cells[0]) or not cells[1].isdigit() or int(cells[1]) < 1:
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: 수험번호나 쪽 번호가 올바르지 않다."
            )
        question = None
        if len(cells) == 4:
            if not cells[2].isdigit() or int(cells[2]) < 1:
                raise ExamShieldError(f"{path.name} {number}번째 줄: 문항 번호가 올바르지 않다.")
            question = int(cells[2])
        if not cells[-1].isdigit() or not 1 <= int(cells[-1]) <= MAX_CHOICE:
            continue
        responses.append(Response(cells[0], int(cells[1]), int(cells[-1]), question))
    if len(widths) > 1:
        raise ExamShieldError(
            f"{path.name}: 문항 번호가 있는 줄과 없는 줄이 섞여 있다. 한 가지 형식으로 쓴다."
        )
    if not responses:
        raise ExamShieldError(f"답안 파일에 읽을 답이 없다: {path}")
    return responses


def tail(probabilities: list[float], at_least: int) -> float:
    """문항마다 일치할 확률이 다를 때, at_least개 이상 일치할 확률."""
    counts = [1.0]
    for chance in probabilities:
        updated = [0.0] * (len(counts) + 1)
        for hits, weight in enumerate(counts):
            updated[hits] += weight * (1.0 - chance)
            updated[hits + 1] += weight * chance
        counts = updated
    return min(1.0, sum(counts[at_least:]))


def decoy_for(table: DecoyTable, response: Response) -> int | None:
    """이 답에 해당하는 미끼 번호. 문항별 미끼가 있으면 그것을, 없으면 그 쪽의 미끼를 쓴다."""
    if response.question is not None and response.item in table:
        return table[response.item]
    return table.get((response.page, None))


def _coincides(decoy: int, response: Response, answer_key: AnswerKey | None) -> bool:
    """미끼가 정답과 같은 문항인지. 그런 문항은 정직한 응시자도 일치하므로 근거가 되지 않는다."""
    if answer_key is None:
        return False
    for entry in answer_key:
        if entry.page != response.page or entry.answer != decoy:
            continue
        if entry.question is None or response.question is None:
            return True
        if entry.question == response.question:
            return True
    return False


def analyse(
    responses: list[Response],
    decoys: DecoyTable | dict[str, DecoyTable],
    key_used: bool,
    answer_key: AnswerKey | None = None,
    choices: int = DEFAULT_CHOICES,
    alpha: float = DEFAULT_ALPHA,
) -> Analysis:
    """decoys는 모두에게 같은 미끼 표 하나이거나, 수험번호마다의 미끼 표다."""
    if not 2 <= choices <= MAX_CHOICE:
        raise ExamShieldError(f"선택지 수는 2~{MAX_CHOICE}이어야 한다.")
    if not 0.0 < alpha < 1.0:
        raise ExamShieldError("판정 기준은 0과 1 사이여야 한다.")
    tables: dict[str, DecoyTable] | None = None
    shared: DecoyTable = {}
    if decoys and all(isinstance(key, str) for key in decoys):
        tables = decoys  # type: ignore[assignment]
    else:
        shared = decoys  # type: ignore[assignment]
    notes: list[str] = []
    if answer_key is None and not key_used:
        notes.append(
            "복사본을 정답표 없이 만들었고 여기에도 정답표를 주지 않았다. 미끼가 우연히 정답과 같은 "
            "문항이 섞여 있으면 정직한 응시자도 일치가 많아진다. --answer-key로 정답표를 준다."
        )

    by_examinee: dict[str, list[Response]] = {}
    for entry in responses:
        by_examinee.setdefault(entry.examinee_id, []).append(entry)
    unknown = sorted(name for name in by_examinee if tables is not None and name not in tables)
    if unknown:
        notes.append(
            f"매핑표에 없는 수험번호의 답안은 보지 않았다: {', '.join(unknown[:10])}"
            + (" 외" if len(unknown) > 10 else "")
        )
        for name in unknown:
            del by_examinee[name]

    # 문항마다 누가 어떤 번호를 골랐는지. 우연의 기준과 서로 견주는 데 쓴다.
    tally: dict[Item, dict[int, int]] = {}
    own: dict[str, dict[Item, dict[int, int]]] = {}
    for name, entries in by_examinee.items():
        mine = own.setdefault(name, {})
        for entry in entries:
            slot = tally.setdefault(entry.item, {})
            slot[entry.answer] = slot.get(entry.answer, 0) + 1
            kept = mine.setdefault(entry.item, {})
            kept[entry.answer] = kept.get(entry.answer, 0) + 1

    # 한 문항에서 미끼 번호를 우연히 고를 확률. 늘 틀리는 응시자가 오답을 고르게 찍는 경우를
    # 위쪽 한계로 잡는다. 실제 응시자는 정답도 고르므로 이보다 낮다.
    floor = 1.0 / (choices - 1) if choices > 2 else 0.5
    cohort = len(by_examinee) >= MIN_COHORT

    def chance_of(item: Item, decoy: int, *excluded: str) -> float:
        if not cohort:
            return floor
        # 견주는 당사자를 뺀 나머지 응시자가 이 문항에서 그 번호를 고른 비율. 그 번호가 원래 인기
        # 있는 오답이면 이 값이 커져서 섣불리 지목하지 않게 된다.
        total = sum(tally[item].values())
        hits = tally[item].get(decoy, 0)
        for name in excluded:
            mine = own[name].get(item, {})
            total -= sum(mine.values())
            hits -= mine.get(decoy, 0)
        return max(floor, (hits + 1) / (total + 2))

    def compare_with(
        name: str, table: DecoyTable, *excluded: str
    ) -> tuple[int, int, float, int, int]:
        """name의 답을 미끼 표와 견준다. (본 답, 일치, 우연일 확률, 정답과 겹쳐 뺀 답, 미끼 없는 답)"""
        chances: list[float] = []
        matches = dropped = unmatched = 0
        for entry in by_examinee[name]:
            decoy = decoy_for(table, entry)
            if decoy is None:
                unmatched += 1
                continue
            if _coincides(decoy, entry, answer_key):
                dropped += 1
                continue
            chances.append(chance_of(entry.item, decoy, name, *excluded))
            matches += 1 if entry.answer == decoy else 0
        return len(chances), matches, tail(chances, matches), dropped, unmatched

    threshold = alpha / max(1, len(by_examinee))
    findings = []
    dropped_total = unmatched_total = 0
    for name in by_examinee:
        table = tables[name] if tables is not None else shared
        items, matches, chance, dropped, unmatched = compare_with(name, table)
        dropped_total += dropped
        unmatched_total += unmatched
        findings.append(
            Finding(name, items, matches, chance, items >= MIN_ITEMS and chance <= threshold)
        )
    findings.sort(key=lambda found: (found.chance, found.examinee_id))
    if any(found.items < MIN_ITEMS for found in findings):
        notes.append(f"답이 {MIN_ITEMS}개보다 적은 응시자는 일치가 많아도 지목하지 않는다.")

    borrowed: list[Borrowed] = []
    if tables is not None and len(by_examinee) > 1:
        # 자기 복사본이 아니라 남의 복사본에 든 미끼와 일치하는 답안을 찾는다. 그 사람의 답을
        # 베꼈거나 그 사람의 복사본을 받아 AI에 넣은 경우다. 짝의 수가 많으므로 기준을 더 엄하게 둔다.
        people = len(by_examinee)
        suspects = {found.examinee_id for found in findings if found.flagged}
        for name in by_examinee:
            for source in by_examinee:
                if source == name:
                    continue
                # 이미 지목된 응시자의 미끼와 견줄 때는 견주는 짝이 적으므로 기준이 덜 엄하다.
                # 그 밖의 모든 짝은 사람 수의 제곱만큼 견주므로 그만큼 엄하게 둔다.
                pairs = people * len(suspects) if source in suspects else people * (people - 1)
                items, matches, chance, _, _ = compare_with(name, tables[source], source)
                if items >= MIN_ITEMS and chance <= alpha / pairs:
                    borrowed.append(Borrowed(name, source, items, matches, chance))
        borrowed.sort(key=lambda found: (found.chance, found.examinee_id, found.source_id))
    return Analysis(
        findings=tuple(findings),
        borrowed=tuple(borrowed),
        dropped=dropped_total,
        unmatched=unmatched_total,
        per_examinee=tables is not None,
        base=(
            f"다른 응시자들이 그 번호를 고른 비율(문항마다, 최소 {floor:.0%})"
            if cohort
            else f"문항마다 {floor:.0%} (선택지 {choices}개에서 오답을 고르게 찍는다고 본 위쪽 한계)"
        ),
        threshold=threshold,
        notes=tuple(notes),
    )


def load_tables(rows: list[tuple[str, str]], jobs_dir: Path) -> tuple[dict[str, DecoyTable], bool]:
    """매핑표의 (수험번호, 복사본 파일 이름)으로 수험번호마다의 미끼 표를 읽는다."""
    tables: dict[str, DecoyTable] = {}
    key_used = True
    for examinee_id, file_name in rows:
        job = jobs_dir / (Path(file_name).stem + ".job.json")
        if not job.is_file():
            raise ExamShieldError(f"{examinee_id}의 작업 기록을 찾지 못했다: {job}")
        tables[examinee_id], used = read_decoys(job)
        key_used = key_used and used
    if not tables:
        raise ExamShieldError("매핑표에 수험번호가 없다.")
    return tables, key_used


def _chance(value: float) -> str:
    if value >= 0.001:
        return f"{value:.1%}"
    return f"{value:.1e} (약 {round(1 / value):,}분의 1)" if value > 0 else "0"


def format_analysis(analysis: Analysis, limit: int = 30) -> str:
    lines = [
        "미끼         "
        + ("수험번호마다 다르다" if analysis.per_examinee else "모든 응시자의 복사본에서 같다"),
        f"우연의 기준  {analysis.base}",
        f"지목 기준    우연일 확률 {analysis.threshold:.1e} 이하 "
        f"(응시자 {len(analysis.findings)}명을 함께 보는 것을 감안한 값)",
    ]
    if analysis.dropped:
        lines.append(f"뺀 답        미끼가 정답과 같은 문항의 답 {analysis.dropped}개")
    if analysis.unmatched:
        lines.append(f"보지 않은 답 미끼가 없는 문항의 답 {analysis.unmatched}개")
    lines += ["", f"{'수험번호':<16}  {'답한 문항':>8}  {'미끼 일치':>8}  우연일 확률"]
    for found in analysis.findings[:limit]:
        mark = "  ← 미끼와 강하게 일치" if found.flagged else ""
        lines.append(
            f"{found.examinee_id:<20}  {found.items:>8}  {found.matches:>8}  "
            f"{_chance(found.chance)}{mark}"
        )
    if len(analysis.findings) > limit:
        lines.append(f"… 나머지 {len(analysis.findings) - limit}명은 일치가 더 적다.")
    flagged = [found.examinee_id for found in analysis.findings if found.flagged]
    lines.append("")
    lines.append(
        f"자기 복사본의 미끼와 강하게 일치: {len(flagged)}명"
        + (f" ({', '.join(flagged[:20])})" if flagged else "")
    )
    if analysis.per_examinee:
        lines.append(f"다른 응시자의 복사본에 든 미끼와 강하게 일치: {len(analysis.borrowed)}건")
        for pair in analysis.borrowed[:limit]:
            lines.append(
                f"  {pair.examinee_id}의 답이 {pair.source_id}의 미끼와 "
                f"{pair.items}문항 중 {pair.matches}문항 일치, 우연일 확률 {_chance(pair.chance)}"
            )
        if analysis.borrowed:
            lines.append(
                "  뒤에 적힌 응시자의 답을 베꼈거나, 그 응시자의 복사본을 받아 AI에 넣었을 수 있다."
            )
    lines += [f"참고: {note}" for note in analysis.notes]
    lines.append(
        "참고: 근거 자료일 뿐 증명이 아니다. 화면 캡처나 파일 올리기로 푼 경우는 잡히지 않는다."
        + (
            ""
            if analysis.per_examinee
            else " 미끼 번호가 모두에게 같아서, 미끼를 따른 답안을 베낀 응시자도 함께 일치한다."
        )
    )
    return "\n".join(lines)
