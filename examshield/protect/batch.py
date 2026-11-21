"""수험번호별 복사본을 폴더에 만들어 쓰는 일. 명령줄과 웹 화면이 함께 쓴다."""

import glob
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from examshield.check.compare import Ocr, format_result
from examshield.errors import ExamShieldError
from examshield.profiles import Profile
from examshield.protect.copies import (
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
    kept = (
        reusable_rows(table, out, stem, ids, seed, original, profile, answers is not None)
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
    if todo:
        base_check, made = protect_copies(original, profile, seed, todo, ocr, answers, jobs)
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
            if on_step is not None:
                on_step()
    rows = [made_rows[examinee] for examinee in ids]
    write_map(table, rows)
    return Batch(rows=rows, table=table, kept=len(kept), made=len(todo))
