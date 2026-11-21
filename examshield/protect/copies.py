import csv
import hashlib
import io
import json
import multiprocessing
import os
import re
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from pathlib import Path

from examshield.check.compare import CheckResult, Expectation, Ocr, compare
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.profiles import Profile
from examshield.protect.pipeline import (
    JOB_SUFFIX,
    KEYED,
    TECHNIQUES,
    AnswerKey,
    Protected,
    decoy_items_record,
    key_for_pages,
    protect_bytes,
    sha256_hex,
)
from examshield.protect.techniques import decoy, dots, watermark
from examshield.protect.techniques.base import Context, Outcome

WATERMARK = "watermark"
DECOY = "decoy"
DOTS = "dots"
MAP_COLUMNS = ("examinee_id", "code", "file", "output_sha256", "cell_pt", "kind")
CELL_RANGES = {WATERMARK: watermark.CELL_RANGE, DOTS: dots.CELL_RANGE}
HEADER_WORDS = {"수험번호", "examinee_id", "examinee", "id", "번호", "학번"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MIN_SEED_BYTES = 16
MIN_COPIES_FOR_PROCESSES = 8
COPIES_PER_PROCESS = 2
MAX_AUTO_JOBS = 8


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


def decoy_seed(key: bytes, input_sha256: str) -> bytes:
    """수험번호마다 다른 미끼 번호를 고르는 시드."""
    return hashlib.sha256(b"examshield-decoy\x00" + key + input_sha256.encode()).digest()


@dataclass(frozen=True)
class _Shared:
    """복사본 하나를 만드는 데 드는 것 가운데 수험번호와 무관한 것. 작업 프로세스에 한 번만 보낸다.

    시드는 넣지 않는다. 수험번호별 키는 부모 프로세스가 만들어 작업마다 하나씩 준다.
    """

    profile_name: str
    profile_sha256: str
    kind: str
    params: dict[str, object]
    decoy_params: dict[str, object] | None
    base: Protected
    base_check: dict[str, object]
    originals: tuple[str, ...]
    avoid: tuple[frozenset[int], ...] | None
    questions: tuple[tuple[tuple[int, int], ...], ...] | None
    input_sha256: str
    answer_key: bool


def _one_copy(shared: _Shared, examinee_id: str, key: bytes) -> Copy:
    base = shared.base
    personal = shared.decoy_params is not None
    current = base.pdf
    lured: Outcome | None = None
    if shared.decoy_params is not None:
        lured = decoy.apply(
            current,
            shared.decoy_params,
            Context(
                seed=decoy_seed(key, shared.input_sha256),
                original_texts=shared.originals,
                avoid_answers=shared.avoid,
                questions=shared.questions,
            ),
        )
        current = lured.pdf
    outcome = TECHNIQUES[shared.kind](
        current, shared.params, Context(seed=b"", original_texts=(), watermark_key=key)
    )
    tolerance = outcome.pixel_tolerance or base.expectation.pixel_tolerance
    texts = lured.decoy_texts if lured is not None and lured.decoy_texts else ()
    overlay = Expectation(
        decoys=texts,
        overlay_images=outcome.overlay_images,
        pixel_tolerance=tolerance,
        overlay_cell_pt=None if outcome.dots else int(outcome.stats["cell_pt"]),
        dots_cell_pt=outcome.dots["cell_pt"] if outcome.dots else None,
        dots_ink=outcome.dots["ink"] if outcome.dots else None,
    )
    # 수험번호와 무관한 단계까지 만든 문서와 비교한다. 달라도 되는 것은 기록된 미끼 문구와
    # 덧씌운 무늬 한 장뿐이다.
    check = compare(base.pdf, outcome.pdf, overlay)
    expectation = replace(
        base.expectation,
        decoys=texts if personal else base.expectation.decoys,
        overlay_images=overlay.overlay_images,
        pixel_tolerance=overlay.pixel_tolerance,
        overlay_cell_pt=overlay.overlay_cell_pt,
        dots_cell_pt=overlay.dots_cell_pt,
        dots_ink=overlay.dots_ink,
    )
    answers = lured.decoy_answers if lured is not None else base.decoy_answers
    items = lured.decoy_items if lured is not None else base.decoy_items
    record = dict(base.record)
    record["techniques"] = [
        *base.record["techniques"],
        *(
            [{"id": DECOY, "params": shared.decoy_params, "stats": lured.stats}]
            if lured is not None
            else []
        ),
        {"id": shared.kind, "params": shared.params, "stats": outcome.stats},
    ]
    record["profile"] = {"name": shared.profile_name, "sha256": shared.profile_sha256}
    record["output_sha256"] = sha256_hex(outcome.pdf)
    record["expect"] = expectation.as_dict()
    record["check"] = shared.base_check
    record["watermark_check"] = check.as_dict()
    record["answer_key"] = shared.answer_key
    record["decoy_answers"] = list(answers) if answers is not None else None
    record["decoy_items"] = decoy_items_record(items)
    record["decoys_per_examinee"] = personal
    return Copy(
        examinee_id=examinee_id,
        protected=Protected(
            pdf=outcome.pdf,
            record=record,
            expectation=expectation,
            decoy_answers=answers,
            decoy_items=items,
        ),
        check=check,
    )


_worker_shared: _Shared | None = None


def _start_worker(shared: _Shared) -> None:
    global _worker_shared
    _worker_shared = shared


def _work(task: tuple[str, bytes]) -> Copy:
    if _worker_shared is None:
        raise RuntimeError("작업 프로세스가 준비되지 않았다.")
    return _one_copy(_worker_shared, *task)


def choose_jobs(requested: int | None, copies: int) -> int:
    """복사본을 만들 프로세스 수. 정하지 않으면 코어 수와 복사본 수를 보고 고른다."""
    if requested is not None and requested < 1:
        raise ExamShieldError("--jobs는 1 이상이어야 한다.")
    if requested is not None:
        return min(requested, max(copies, 1))
    # 프로세스를 띄우는 데도 시간이 든다. 복사본이 적으면 하나씩 만드는 편이 빠르다.
    if copies < MIN_COPIES_FOR_PROCESSES:
        return 1
    return max(1, min(os.cpu_count() or 1, MAX_AUTO_JOBS, copies // COPIES_PER_PROCESS))


def _in_processes(shared: _Shared, tasks: Iterator[tuple[str, bytes]], jobs: int) -> Iterator[Copy]:
    """수험번호 목록의 순서대로 돌려준다. 받는 쪽이 느리면 더 만들지 않고 기다린다."""
    # fork는 macOS에서 안전하지 않고, 단일 실행 파일에서도 spawn만 쓸 수 있다.
    pool = ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(shared,),
    )
    waiting: deque[Future[Copy]] = deque()
    try:
        for task in tasks:
            waiting.append(pool.submit(_work, task))
            if len(waiting) >= jobs * 2:
                yield waiting.popleft().result()
        while waiting:
            yield waiting.popleft().result()
    except BrokenProcessPool as error:
        raise ExamShieldError(
            "복사본을 만들던 작업 프로세스가 죽었다. 메모리가 모자랐을 수 있다. "
            "--jobs를 줄여서 다시 해 본다."
        ) from error
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def protect_copies(
    pdf: bytes,
    profile: Profile,
    seed: bytes,
    examinee_ids: list[str],
    ocr: Ocr | None = None,
    answer_key: AnswerKey | None = None,
    jobs: int | None = 1,
) -> tuple[CheckResult, Iterator[Copy]]:
    """수험번호와 무관한 단계는 한 번만 만들고 검사한다. 그 위에 수험번호별 미끼와 무늬를 얹는다.

    미끼 번호도 수험번호마다 다르게 고른다. 그래야 미끼를 따른 답안을 베낀 응시자와, 자기 복사본을
    AI에 넣은 응시자를 가릴 수 있다. 미끼가 맨 마지막(무늬 바로 앞) 기법일 때만 그렇게 한다.

    jobs가 2 이상이면 수험번호별 단계를 여러 프로세스로 나눠 한다. None이면 알아서 고른다.
    어느 쪽이든 나오는 복사본은 바이트까지 같다.
    """
    kind, params = keyed_technique(profile)
    plain = [entry for entry in profile.techniques if entry.id not in KEYED]
    personal = bool(plain) and plain[-1].id == DECOY
    shared_steps = plain[:-1] if personal else plain
    base_profile = replace(profile, techniques=tuple(shared_steps))
    base = protect_bytes(pdf, base_profile, seed, None, None if personal else answer_key)
    base_check = compare(pdf, base.pdf, base.expectation, ocr)
    originals = tuple(page_text(page) for page in open_pdf_bytes(pdf))
    avoid, questions = key_for_pages(answer_key if personal else None, len(originals), profile)
    shared = _Shared(
        profile_name=profile.name,
        profile_sha256=profile.sha256,
        kind=kind,
        params=params,
        decoy_params=plain[-1].params if personal else None,
        base=base,
        base_check=base_check.as_dict(),
        originals=originals,
        avoid=avoid,
        questions=questions,
        input_sha256=sha256_hex(pdf),
        answer_key=answer_key is not None,
    )
    workers = choose_jobs(jobs, len(examinee_ids))
    tasks = ((examinee, watermark.derive_key(seed, examinee)) for examinee in examinee_ids)

    def make() -> Iterator[Copy]:
        if workers > 1 and base_check.passed:
            yield from _in_processes(shared, tasks, workers)
            return
        for examinee_id, key in tasks:
            yield _one_copy(shared, examinee_id, key)

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


def append_map(path: Path, row: MapRow) -> None:
    """만든 복사본을 하나씩 매핑표에 보탠다. 중간에 끊겨도 그때까지 만든 것은 추적할 수 있다."""
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            [row.examinee_id, row.code, row.file, row.output_sha256, row.cell_pt, row.kind]
        )


def reusable_rows(
    table: Path,
    out_dir: Path,
    stem: str,
    examinee_ids: list[str],
    seed: bytes,
    pdf: bytes,
    profile: Profile,
    answer_key: bool,
) -> dict[str, MapRow]:
    """끊긴 실행이 남긴 복사본 가운데 그대로 써도 되는 것. 수험번호 → 매핑표의 행.

    매핑표에 적혀 있고, 그 코드가 지금 시드에서 나온 것이며, 파일이 적힌 해시 그대로이고,
    작업 기록의 원본과 프로파일이 지금 것과 같을 때만 쓴다. 하나라도 어긋나면 다시 만든다.
    """
    try:
        rows = read_map(table)
    except ExamShieldError:
        return {}
    kind, _ = keyed_technique(profile)
    wanted = set(examinee_ids)
    input_sha256 = sha256_hex(pdf)
    kept: dict[str, MapRow] = {}
    for row in rows:
        examinee = row.examinee_id
        if examinee not in wanted or row.kind != kind:
            continue
        if row.file != f"{stem}-{examinee}.pdf":
            continue
        if row.code != watermark.code_of(watermark.derive_key(seed, examinee)):
            continue
        path = out_dir / row.file
        record_path = path.with_name(path.stem + JOB_SUFFIX)
        try:
            if sha256_hex(path.read_bytes()) != row.output_sha256:
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            same = (
                record["output_sha256"] == row.output_sha256
                and record["input_sha256"] == input_sha256
                and record["profile"]["sha256"] == profile.sha256
                and bool(record.get("answer_key")) == answer_key
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if same:
            kept[examinee] = row
    return kept


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
