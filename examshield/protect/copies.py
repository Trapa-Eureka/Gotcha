import csv
import io
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from examshield.check.compare import CheckResult, Expectation, Ocr, compare
from examshield.errors import ExamShieldError
from examshield.profiles import Profile
from examshield.protect.pipeline import (
    KEYED,
    TECHNIQUES,
    AnswerKey,
    Protected,
    protect_bytes,
    sha256_hex,
)
from examshield.protect.techniques import dots, watermark
from examshield.protect.techniques.base import Context

WATERMARK = "watermark"
DOTS = "dots"
MAP_COLUMNS = ("examinee_id", "code", "file", "output_sha256", "cell_pt", "kind")
CELL_RANGES = {WATERMARK: watermark.CELL_RANGE, DOTS: dots.CELL_RANGE}
HEADER_WORDS = {"수험번호", "examinee_id", "examinee", "id", "번호", "학번"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MIN_SEED_BYTES = 16


@dataclass(frozen=True)
class MapRow:
    examinee_id: str
    code: str
    file: str
    output_sha256: str
    cell_pt: int
    # 무늬의 종류. watermark는 화면용 옅은 바탕 무늬, dots는 인쇄용 점 무늬다.
    kind: str = WATERMARK


def read_examinee_ids(path: Path) -> list[str]:
    """CSV의 첫 열을 수험번호로 읽는다. 첫 줄이 제목이면 건너뛴다."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ExamShieldError(f"수험번호 목록을 읽지 못했다: {path}") from error
    ids: list[str] = []
    for number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        value = row[0].strip() if row else ""
        if not value or (number == 1 and value.lower() in HEADER_WORDS):
            continue
        if not _ID.fullmatch(value):
            raise ExamShieldError(
                f"{path.name} {number}번째 줄: 수험번호는 영문, 숫자, '-', '_'만 64자까지 쓸 수 있다."
            )
        ids.append(value)
    if not ids:
        raise ExamShieldError(f"수험번호 목록이 비어 있다: {path}")
    # 대소문자만 다른 번호도 겹친 것으로 본다. 파일 이름이 겹쳐 한쪽 복사본이 사라진다.
    folded = [value.casefold() for value in ids]
    repeated = sorted({value for value in ids if folded.count(value.casefold()) > 1})
    if repeated:
        raise ExamShieldError(
            f"수험번호가 겹친다(대소문자만 다른 것 포함): {', '.join(repeated[:5])}"
        )
    return ids


def check_seed(seed: bytes | None) -> bytes:
    if seed is None:
        raise ExamShieldError(
            "--copies에는 --seed-file이 필요하다. 시드는 워터마크의 비밀값이라 기관이 직접 만들어 보관한다. "
            "예: head -c 32 /dev/urandom > secrets/exam.seed"
        )
    if len(seed) < MIN_SEED_BYTES:
        raise ExamShieldError(f"시드 파일이 너무 짧다. {MIN_SEED_BYTES}바이트 이상이어야 한다.")
    return seed


def keyed_technique(profile: Profile) -> tuple[str, dict[str, object]]:
    """수험번호마다 다른 무늬를 넣는 기법과 그 설정값."""
    for entry in profile.techniques:
        if entry.id in KEYED:
            return entry.id, entry.params
    raise ExamShieldError(
        f"프로파일 '{profile.name}'에는 수험번호별 무늬를 넣는 기법(watermark, dots)이 없다. "
        "--copies는 a-watermark, a-full, b-dots 같은 프로파일과 함께 쓴다."
    )


@dataclass(frozen=True)
class Copy:
    examinee_id: str
    protected: Protected
    check: CheckResult


def protect_copies(
    pdf: bytes,
    profile: Profile,
    seed: bytes,
    examinee_ids: list[str],
    ocr: Ocr | None = None,
    answer_key: AnswerKey | None = None,
) -> tuple[CheckResult, Iterator[Copy]]:
    """워터마크 전 단계는 한 번만 만들고 검사한다. 그 위에 수험번호별 무늬만 따로 덧씌운다."""
    kind, params = keyed_technique(profile)
    base_profile = replace(
        profile, techniques=tuple(e for e in profile.techniques if e.id not in KEYED)
    )
    base = protect_bytes(pdf, base_profile, seed, None, answer_key)
    base_check = compare(pdf, base.pdf, base.expectation, ocr)

    def make() -> Iterator[Copy]:
        context = Context(seed=b"", original_texts=())
        for examinee_id in examinee_ids:
            key = watermark.derive_key(seed, examinee_id)
            outcome = TECHNIQUES[kind](base.pdf, params, replace(context, watermark_key=key))
            tolerance = outcome.pixel_tolerance or base.expectation.pixel_tolerance
            overlay = Expectation(
                overlay_images=outcome.overlay_images,
                pixel_tolerance=tolerance,
                overlay_cell_pt=None if outcome.dots else int(outcome.stats["cell_pt"]),
                dots_cell_pt=outcome.dots["cell_pt"] if outcome.dots else None,
                dots_ink=outcome.dots["ink"] if outcome.dots else None,
            )
            # 무늬를 넣기 전 문서와 비교한다. 달라도 되는 것은 덧씌운 무늬 한 장뿐이다.
            check = compare(base.pdf, outcome.pdf, overlay)
            expectation = replace(
                base.expectation,
                overlay_images=overlay.overlay_images,
                pixel_tolerance=overlay.pixel_tolerance,
                overlay_cell_pt=overlay.overlay_cell_pt,
                dots_cell_pt=overlay.dots_cell_pt,
                dots_ink=overlay.dots_ink,
            )
            record = dict(base.record)
            record["techniques"] = [
                *base.record["techniques"],
                {"id": kind, "params": params, "stats": outcome.stats},
            ]
            record["output_sha256"] = sha256_hex(outcome.pdf)
            record["expect"] = expectation.as_dict()
            record["check"] = base_check.as_dict()
            record["watermark_check"] = check.as_dict()
            yield Copy(
                examinee_id=examinee_id,
                protected=Protected(
                    pdf=outcome.pdf,
                    record=record,
                    expectation=expectation,
                    decoy_answers=base.decoy_answers,
                ),
                check=check,
            )

    return base_check, make()


def map_row(copy: Copy, seed: bytes, file_name: str) -> MapRow:
    key = watermark.derive_key(seed, copy.examinee_id)
    last = copy.protected.record["techniques"][-1]
    stats = last["stats"]
    return MapRow(
        kind=str(last["id"]),
        examinee_id=copy.examinee_id,
        code=watermark.code_of(key),
        file=file_name,
        output_sha256=copy.protected.record["output_sha256"],
        cell_pt=int(stats["cell_pt"]),
    )


def write_map(path: Path, rows: list[MapRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MAP_COLUMNS)
        for row in rows:
            writer.writerow(
                [row.examinee_id, row.code, row.file, row.output_sha256, row.cell_pt, row.kind]
            )


def read_map(path: Path) -> list[MapRow]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [
                MapRow(
                    examinee_id=entry["examinee_id"],
                    code=entry["code"],
                    file=entry["file"],
                    output_sha256=entry["output_sha256"],
                    cell_pt=int(entry["cell_pt"]),
                    # 이 열이 없는 매핑표는 도트 무늬가 생기기 전에 만든 것이다.
                    kind=entry.get("kind") or WATERMARK,
                )
                for entry in csv.DictReader(handle)
            ]
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ExamShieldError(f"매핑표를 읽지 못했다: {path}") from error
    if not rows:
        raise ExamShieldError(f"매핑표가 비어 있다: {path}")
    if any(row.kind not in CELL_RANGES for row in rows):
        raise ExamShieldError(f"매핑표의 kind 값을 모른다: {path}")
    for row in rows:
        low, high = CELL_RANGES[row.kind]
        if not low <= row.cell_pt <= high:
            raise ExamShieldError(f"매핑표의 cell_pt 값이 범위({low}~{high})를 벗어났다: {path}")
    return rows
