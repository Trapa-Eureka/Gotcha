"""수험번호별 복사본을 폴더에 만들어 쓰는 일. 명령줄과 웹 화면이 함께 쓴다."""

import glob
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from examshield.check.compare import Ocr, format_result
from examshield.clearzone import THIN_PAGE_CELLS, TRACE_MIN_CELLS, blank_cells
from examshield.errors import ExamShieldError
from examshield.pdfio import open_pdf_bytes
from examshield.profiles import Profile
from examshield.protect.copies import (
    DOTS,
    MapRow,
    append_map,
    check_seed,
    choose_jobs,
    keyed_technique,
    map_row,
    protect_copies,
    reusable_rows,
    write_map,
)
from examshield.protect.pipeline import AnswerKey, append_log, job_path, write_outputs
from examshield.protect.techniques import dots, watermark


class CopiesFailed(ExamShieldError):
    """복사본이 검사를 통과하지 못했다. detail에 검사 결과가 있다."""

    def __init__(self, message: str, detail: str) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True)
class Batch:
    rows: list[MapRow]
    table: Path
    kept: int
    made: int
    # 저시력 수험생용으로 만든 복사본 수와, 그 복사본들에서 무늬를 넣을 수 있었던 칸의 비율(가장 낮은 값).
    low_vision: int = 0
    open_share: float | None = None
    # 빈 바탕이 적어 그림 한 장으로는 추적하기 어려운 쪽(1부터 센다)과 전체 쪽수, 문서 전체의 빈 칸 수.
    thin_pages: tuple[int, ...] = ()
    pages: int = 0
    blank_cells: int = 0


def inside(path: Path, folder: Path) -> bool:
    """path가 folder 안(또는 folder 자체)인지 본다. 대소문자나 유니코드 표기만 다른 경로도 잡는다."""
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if folder.exists() and candidate.exists() and candidate.samefile(folder):
            return True
    wanted = unicodedata.normalize("NFC", str(folder.resolve())).casefold()
    return any(
        unicodedata.normalize("NFC", str(candidate)).casefold() == wanted
        for candidate in (resolved, *resolved.parents)
    )


def make_copies(
    *,
    stem: str,
    load: Callable[[], bytes],
    out: Path,
    profile: Profile,
    seed_file: Path | None,
    ids: list[str],
    table: Path,
    answers: AnswerKey | None = None,
    jobs: int | None = None,
    resume: bool = False,
    force: bool = False,
    ocr: Ocr | None = None,
    low_vision: list[str] | None = None,
    keep_out: tuple[tuple[str, Path], ...] = (),
    on_start: Callable[[int, int], None] | None = None,
    on_step: Callable[[], None] | None = None,
) -> Batch:
    """복사본을 out에 쓰고 매핑표를 table에 쓴다.

    load는 원본 PDF를 돌려주는 함수다. 경로 확인이 끝난 뒤에 부른다.
    keep_out은 출력 폴더 안에 있으면 안 되는 그 밖의 파일(이름, 경로)이다.
    on_start(이미 만든 수, 전체 수)는 만들기 직전에, on_step은 복사본 하나를 쓸 때마다 부른다.
    """
    keyed_technique(profile)
    low = frozenset(low_vision or ())
    unknown = sorted(low - set(ids))
    if unknown:
        raise ExamShieldError(
            "저시력 수험번호 목록에 전체 수험번호 목록에 없는 번호가 있다: "
            + ", ".join(unknown[:5])
        )
    if seed_file is not None and not seed_file.is_file():
        raise ExamShieldError(f"시드 파일을 찾지 못했다: {seed_file}")
    seed = check_seed(seed_file.read_bytes() if seed_file is not None else None)
    choose_jobs(jobs, len(ids))
    if out.exists() and not out.is_dir():
        raise ExamShieldError(f"출력 폴더 자리에 파일이 있다: {out}")
    # S5: 매핑표와 시드는 배포하는 폴더에 두지 않는다.
    for label, path in (("매핑표", table), ("시드 파일", seed_file), *keep_out):
        if path is not None and inside(path, out):
            raise ExamShieldError(
                f"{label}를 출력 폴더 안에 둘 수 없다. 출력 폴더는 배포하는 곳이다: {path}"
            )
    targets = [out / f"{stem}-{examinee}.pdf" for examinee in ids]
    existing = [str(path) for path in targets if path.exists() or job_path(path).exists()]
    if table.exists():
        existing.append(str(table))
    if existing and not force and not resume:
        shown = ", ".join(existing[:5]) + (" 외" if len(existing) > 5 else "")
        raise ExamShieldError(
            "이미 같은 이름의 파일이 있다. 끊긴 실행을 이어 하려면 이어 하기(--resume)를, "
            f"처음부터 덮어쓰려면 덮어쓰기(--force)를 고른다: {shown}"
        )
    # 이번 목록에 없는 예전 복사본이 남으면 매핑표와 폴더가 어긋난다. 지우지는 않고 멈춘다.
    wanted = {path.name for path in targets}
    stale = sorted(
        path.name
        for path in (out.glob(f"{glob.escape(stem)}-*.pdf") if out.is_dir() else [])
        if path.name not in wanted
    )
    if stale:
        raise ExamShieldError(
            "출력 폴더에 이번 수험번호 목록에 없는 복사본이 있다. 치우거나 새 폴더를 쓴다: "
            + ", ".join(stale[:5])
        )
    original = load()
    kind, params = keyed_technique(profile)
    cell_pt = (dots if kind == DOTS else watermark).read_params(params)[0]
    blank = blank_cells(open_pdf_bytes(original), cell_pt)
    kept = (
        reusable_rows(table, out, stem, ids, seed, original, profile, answers is not None, low)
        if resume
        else {}
    )
    # 매핑표를 쓸 수 있는지 복사본을 만들기 전에 확인한다. 복사본만 남고 매핑표가 없으면 추적할 수 없다.
    # 만드는 동안에도 복사본마다 한 줄씩 보탠다. 끊겨도 그때까지 만든 것은 추적하고 이어 할 수 있다.
    write_map(table, list(kept.values()))
    todo = [examinee for examinee in ids if examinee not in kept]
    if on_start is not None:
        on_start(len(kept), len(ids))
    made_rows = dict(kept)
    shares: list[float] = []
    if todo:
        base_check, made = protect_copies(original, profile, seed, todo, ocr, answers, jobs, low)
        if not base_check.passed:
            raise CopiesFailed("검사 실패. 복사본을 쓰지 않았다.", format_result(base_check))
        for copy in made:
            append_log(out, copy.protected, copy.check.passed, datetime.now().astimezone())
            if not copy.check.passed:
                raise CopiesFailed(
                    f"{copy.examinee_id}: 워터마크 검사 실패. 중단한다.", format_result(copy.check)
                )
            written = write_outputs(copy.protected, out, f"{stem}-{copy.examinee_id}")
            row = map_row(copy, seed, written.name)
            append_map(table, row)
            made_rows[copy.examinee_id] = row
            share = copy.protected.record["techniques"][-1]["stats"].get("open_share")
            if share is not None:
                shares.append(float(share))
            if on_step is not None:
                on_step()
    rows = [made_rows[examinee] for examinee in ids]
    write_map(table, rows)
    return Batch(
        rows=rows,
        table=table,
        kept=len(kept),
        made=len(todo),
        low_vision=sum(1 for row in rows if row.clear_pt),
        open_share=min(shares) if shares else None,
        thin_pages=tuple(
            number for number, count in enumerate(blank, start=1) if count < THIN_PAGE_CELLS
        ),
        pages=len(blank),
        blank_cells=sum(blank),
    )


# 무늬를 넣을 수 있는 칸이 이보다 적으면 그림 한 장으로는 추적하기 어렵다고 알린다.
THIN_OPEN_SHARE = 0.2


def low_vision_lines(batch: Batch) -> list[str]:
    """저시력 수험생용 복사본에 대해 사용자에게 알릴 말. 명령줄과 웹 화면이 함께 쓴다."""
    if not batch.low_vision:
        return []
    lines = [
        f"저시력 수험생용 복사본 {batch.low_vision}개: 글자와 그림 둘레를 비우고 여백에만 무늬를 넣었다. "
        "화면의 일부만 잘라 낸 그림으로는 추적하기 어렵다."
    ]
    if batch.open_share is not None:
        lines.append(f"  무늬를 넣은 자리는 쪽 넓이의 {batch.open_share:.0%}다.")
        if batch.open_share < THIN_OPEN_SHARE:
            lines.append(
                "  주의: 여백이 적은 문서다. 쪽 전체가 나온 그림이나 PDF가 아니면 추적하지 못할 수 있다."
            )
    return lines


def thin_page_lines(batch: Batch) -> list[str]:
    """빈 바탕이 적은 쪽에 대해 알릴 말. 무늬는 글자나 그림이 없는 칸에서만 읽을 수 있다."""
    if not batch.thin_pages:
        return []
    shown = ", ".join(str(number) for number in batch.thin_pages[:10])
    more = " 외" if len(batch.thin_pages) > 10 else ""
    lines = [
        f"주의: 빈 바탕이 적은 쪽이 {batch.pages}쪽 가운데 {len(batch.thin_pages)}쪽 있다({shown}쪽{more}). "
        "쪽 전체가 그림이나 색으로 덮인 쪽이다. 이런 쪽은 화면 그림 한 장이나 사진 한 장으로는 "
        "누구의 복사본인지 찾지 못할 수 있다."
    ]
    if batch.blank_cells < TRACE_MIN_CELLS:
        lines.append(
            "  문서 전체에도 빈 바탕이 모자라다. PDF 전체가 유출돼도 무늬로는 찾지 못한다. "
            "손대지 않은 파일의 해시로만 찾을 수 있다."
        )
    else:
        lines.append("  PDF 전체나 빈 바탕이 있는 다른 쪽이 함께 있으면 찾는다.")
    return lines


def notice_lines(batch: Batch) -> list[str]:
    """복사본을 만든 뒤 사용자에게 알릴 말. 명령줄과 웹 화면이 함께 쓴다."""
    return [*low_vision_lines(batch), *thin_page_lines(batch)]
