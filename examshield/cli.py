import glob
import importlib
import os
import sys
import unicodedata
import webbrowser
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import typer

from examshield import __version__
from examshield.auth.keys import delete_key, find_key, setup_help, store_key, usable
from examshield.benchmark import load_items
from examshield.check.compare import Expectation, compare, format_result
from examshield.convert import HWP_NOTICE, SUPPORTED_SUFFIXES, needs_review, read_document
from examshield.errors import ExamShieldError, ModelUnavailableError
from examshield.netguard import OFFLINE_ENV, offline_enabled
from examshield.ocr import find_tesseract
from examshield.profiles import Profile, list_profile_names, load_profile
from examshield.progress import Progress
from examshield.protect.copies import (
    append_map,
    check_seed,
    choose_jobs,
    keyed_technique,
    map_row,
    protect_copies,
    read_examinee_ids,
    read_map,
    reusable_rows,
    write_map,
)
from examshield.protect.pipeline import (
    AnswerKey,
    KeyEntry,
    append_log,
    job_path,
    load_expectation,
    protect_and_check,
    protect_bytes,
    read_answer_key,
    write_outputs,
)
from examshield.providers import PROVIDERS, Provider, get_provider
from examshield.trace.decoys import (
    DEFAULT_ALPHA,
    DEFAULT_CHOICES,
    DecoyTable,
    analyse,
    format_analysis,
    load_tables,
    read_decoys,
    read_responses,
)
from examshield.trace.detect import format_trace
from examshield.trace.detect import trace as trace_document
from examshield.ui.server import make_server
from examshield.verify.calibrate import (
    decode_photo,
    format_comparison,
    measure,
    reference_pages,
    simulated,
    suggest,
)
from examshield.verify.capture_sim import CaptureConditions, conditions_from
from examshield.verify.checkpoint import Checkpoint, Tracked, checkpoint_path
from examshield.verify.drift import (
    DEFAULT_REPORT_DIR,
    NOT_APPLICABLE,
    RECORD_DIR,
    Assessment,
    Current,
    assess,
    check_report,
    find_latest,
    format_assessment,
    record_refusal,
    write_record,
)
from examshield.verify.models.base import ModelClient
from examshield.verify.models.baseline import TesseractModel, TextLayerModel
from examshield.verify.models.cloud import chosen_model
from examshield.verify.models.local import OllamaModel
from examshield.verify.models.mock import MockModel
from examshield.verify.report import (
    display_width,
    format_replies,
    format_report,
    load_report,
    write_replies,
    write_report,
)
from examshield.verify.runner import (
    Skipped,
    Subject,
    builtin_subject,
    custom_subject,
    estimate_calls,
    run_verification,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=(
        "시험 문서의 내용은 그대로 둔 채, 텍스트를 복사해 AI에 넣는 것을 막고 유출되면 추적할 수 있는 "
        "복사본을 만든다. 효과는 수치로 잰다."
    ),
)


class ModelGroup(StrEnum):
    local = "local"
    cloud = "cloud"
    all = "all"
    mock = "mock"


def _fail(error: ExamShieldError) -> typer.Exit:
    typer.echo(f"오류: {error}", err=True)
    return typer.Exit(error.exit_code)


def _local_models(local_model: str | None) -> tuple[list[ModelClient], list[Skipped]]:
    models: list[ModelClient] = [TextLayerModel()]
    skipped: list[Skipped] = []
    try:
        models.append(TesseractModel())
    except ModelUnavailableError as error:
        skipped.append(Skipped("tesseract", str(error)))
    try:
        models.append(OllamaModel(local_model))
    except ExamShieldError as error:
        skipped.append(Skipped("ollama", str(error)))
    return models, skipped


VERIFY_WATERMARK_KEY = b"examshield-verify-sample"
CLOUD_CLASSES = {
    "openai": ("examshield.verify.models.openai", "OpenAIModel"),
    "anthropic": ("examshield.verify.models.anthropic", "AnthropicModel"),
    "google": ("examshield.verify.models.google", "GoogleModel"),
}
auth_app = typer.Typer(no_args_is_help=True, help="클라우드 모델용 API 키를 OS 키체인에 보관한다.")
app.add_typer(auth_app, name="auth")
profile_app = typer.Typer(no_args_is_help=True, help="프로파일 목록과 최신 검증 상태를 보여준다.")
app.add_typer(profile_app, name="profile")


def _current() -> Current:
    items = load_items()
    return Current(__version__, items.version, items.sha256)


def _assess(profile: Profile, reports: Path | None = None) -> Assessment:
    """저장소의 검증 기록과 reports 폴더에서 가장 최근 리포트를 찾아 판정한다."""
    now = datetime.now().astimezone()
    found, source, changed = find_latest(profile, [RECORD_DIR, reports or DEFAULT_REPORT_DIR], now)
    return assess(profile, found, now, _current(), changed=changed, source=source)


def _report_verdict(report: dict[str, Any]) -> str | None:
    """리포트 하나를 지금의 프로파일, 도구, 문항 세트에 비춰 본 판정. 해당 없으면 None."""
    now = datetime.now().astimezone()
    try:
        checked = check_report(report, now)
        profile = load_profile(checked.profile_name)
        if not profile.techniques:
            return None
        refusal = record_refusal(report, now)
        if refusal is not None:
            return f"참고: 프로파일의 검증 기록으로 쓰지 않는다. {refusal}"
        changed = checked.profile_sha256 != profile.sha256
        assessment = assess(profile, None if changed else report, now, _current(), changed=changed)
    except (ValueError, ExamShieldError) as error:
        return f"참고: 이 리포트로는 판정하지 못한다. {error}"
    return None if assessment.state == NOT_APPLICABLE else format_assessment(assessment)


@app.callback()
def main(
    offline: bool = typer.Option(
        False,
        "--offline",
        help=f"이 PC 밖으로 나가는 접속을 모두 거부한다. 환경변수 {OFFLINE_ENV}=1과 같다.",
    ),
) -> None:
    if offline:
        os.environ[OFFLINE_ENV] = "1"


def _cloud_model(provider: Provider) -> ModelClient | Skipped:
    key = find_key(provider)
    if key is None:
        return Skipped(provider.id, "API 키가 없다. " + setup_help(provider))
    if not usable(key.value):
        return Skipped(provider.id, f"{key.source}의 키에 한글이나 공백 같은 글자가 섞여 있다.")
    module_name, class_name = CLOUD_CLASSES[provider.id]
    try:
        module = importlib.import_module(module_name)
        client: ModelClient = getattr(module, class_name)(key)
    except ImportError:
        return Skipped(
            provider.id,
            f"{provider.package} 패키지가 없다. `uv sync --extra cloud` 또는 "
            "`uv tool install 'examshield[cloud]'`로 설치한다.",
        )
    return client


OFFLINE_REASON = (
    f"오프라인 설정({OFFLINE_ENV} 또는 --offline)이 켜져 있어 클라우드 모델을 부르지 않는다. "
    "API 키가 있어도 마찬가지다."
)


def _cloud_models() -> tuple[list[ModelClient], list[Skipped]]:
    if offline_enabled():
        return [], [Skipped("cloud", OFFLINE_REASON)]
    models: list[ModelClient] = []
    skipped: list[Skipped] = []
    for provider in PROVIDERS:
        made = _cloud_model(provider)
        if isinstance(made, Skipped):
            skipped.append(made)
        else:
            models.append(made)
    return models, skipped


def _confirm_cloud(
    profile: Profile,
    subject: Subject,
    models: list[ModelClient],
    yes: bool,
    only: tuple[str, ...] | None = None,
) -> None:
    pages = len(subject.truths)
    copies = 2 if profile.techniques else 1
    what = (
        "사용자가 --allow-custom으로 준 파일"
        if subject.custom
        else f"내장 문항 세트 {subject.version}"
    )
    typer.echo("클라우드 모델을 호출한다. 아래 내용이 이 PC 밖으로 나간다.")
    typer.echo(
        f"  보낼 문서: {what}, {pages}쪽 × {copies}벌(대조군{', 변환본' if copies == 2 else ''})"
    )
    total = 0
    for model in models:
        calls = estimate_calls(profile, pages, model, only)
        total += calls
        company = get_provider(model.info.provider).company
        typer.echo(f"  대상: {company} ({model.info.name}), 예상 호출 {calls}건")
    typer.echo(f"  예상 호출 수 합계: {total}건. 요금은 각 회사가 사용자 키에 청구한다.")
    if subject.custom:
        typer.echo(
            "  주의: 이 파일의 내용이 위 회사들의 서버로 전송된다. 실제 시험지는 보내지 않는다."
        )
    if yes:
        return
    if not typer.confirm("계속할까?", default=False):
        typer.echo("취소했다. 아무것도 보내지 않았다.", err=True)
        raise typer.Exit(1)


@app.command(help="프로파일의 저항 효과를 측정하고 리포트를 쓴다. 기본 대상은 내장 문항 세트다.")
def verify(
    profile: str = typer.Option(..., "--profile", help="프로파일 이름. 예: none, a-decoy"),
    models: ModelGroup = typer.Option(
        ModelGroup.local,
        "--models",
        help="local, cloud, all. mock은 실제 AI가 아닌 점검용 모의 모델.",
    ),
    out: Path = typer.Option(Path("out/reports"), "--out", help="리포트를 쓸 폴더"),
    local_model: str | None = typer.Option(
        None, "--local-model", help="Ollama 모델 이름. 없으면 설치된 첫 모델을 쓴다."
    ),
    yes: bool = typer.Option(False, "--yes", help="클라우드 호출 전 확인 질문을 건너뛴다."),
    allow_custom: Path | None = typer.Option(
        None,
        "--allow-custom",
        help="내장 문항 세트 대신 이 PDF로 측정한다. 클라우드 모델과 함께 쓰면 파일 내용이 밖으로 나간다.",
    ),
    channels: str | None = typer.Option(
        None,
        "--channels",
        help="잴 경로를 고른다. text, image, file을 쉼표로 잇는다. 없으면 프로파일이 재는 경로 전부.",
    ),
    avoid_answers: bool = typer.Option(
        False,
        "--avoid-answers",
        help="내장 문항의 정답을 정답표로 써서, protect --answer-key를 준 것처럼 잰다.",
    ),
    benchmark: str = typer.Option(
        "v1",
        "--benchmark",
        help="내장 문항 세트. v1은 쪽마다 문항 하나(12문항), v2는 쪽마다 문항 넷(36문항).",
    ),
    question_decoys: bool = typer.Option(
        False,
        "--question-decoys",
        help="--avoid-answers에 더해, 정답표에 문항 번호를 적은 것처럼 문항마다 미끼를 넣어 잰다.",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="결과를 이 프로파일의 최신 검증 기록으로 저장소(examshield/profiles/verified)에 남긴다.",
    ),
    keep_replies: bool = typer.Option(
        False,
        "--keep-replies",
        help="모델이 돌려준 글과 문항별 답을 리포트 옆 파일에 남긴다. 내장 문항 세트로 잴 때만 된다.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="끊기거나 일부 호출이 실패한 실행을 이어 한다. 24시간 안에 받아 둔 응답은 다시 부르지 않는다.",
    ),
) -> None:
    replies: list[dict[str, Any]] | None = [] if keep_replies else None
    try:
        loaded = load_profile(profile)
        if resume and allow_custom is not None:
            raise ExamShieldError(
                "--resume은 --allow-custom과 함께 쓸 수 없다. "
                "사용자의 파일로 잴 때는 응답을 파일에 적어 두지 않는다."
            )
        if keep_replies and allow_custom is not None:
            raise ExamShieldError(
                "--keep-replies는 --allow-custom과 함께 쓸 수 없다. "
                "응답에는 문서를 옮겨 적은 글이 들어 있어서 사용자의 파일로 잴 때는 남기지 않는다."
            )
        if record and getattr(sys, "frozen", False):
            raise ExamShieldError(
                "단일 실행 파일에서는 --record를 쓸 수 없다. 기록할 저장소가 없다. "
                "리포트는 --out 폴더에 남는다."
            )
        only = tuple(part.strip() for part in channels.split(",")) if channels else None
        if record and only is not None:
            raise ExamShieldError("--record는 --channels와 함께 쓸 수 없다. 경로를 모두 재야 한다.")
        if record and benchmark != "v1":
            raise ExamShieldError("--record는 기본 문항 세트(v1)로 잰 결과에만 쓴다.")
        if record and (models is ModelGroup.mock or allow_custom is not None):
            raise ExamShieldError(
                "--record는 내장 문항 세트를 실제 AI 모델로 잰 결과에만 쓴다. "
                "--models mock이나 --allow-custom과 함께 쓸 수 없다."
            )
        clients: list[ModelClient] = []
        skipped: list[Skipped] = []
        cloud: list[ModelClient] = []
        if models is ModelGroup.mock:
            clients = [TextLayerModel(), MockModel(load_items(benchmark).items)]
        if models in (ModelGroup.local, ModelGroup.all):
            clients, skipped = _local_models(local_model)
        if models in (ModelGroup.cloud, ModelGroup.all):
            cloud, cloud_skipped = _cloud_models()
            skipped += cloud_skipped
            if not cloud and models is ModelGroup.cloud:
                reasons = "\n".join(f"  - {entry.reason}" for entry in cloud_skipped)
                raise ExamShieldError(f"쓸 수 있는 클라우드 모델이 없다.\n{reasons}")
        if allow_custom is not None:
            if not allow_custom.is_file():
                raise ExamShieldError(f"--allow-custom 파일을 찾지 못했다: {allow_custom}")
            subject = custom_subject(allow_custom.read_bytes())
        else:
            subject = builtin_subject(load_items(benchmark))
        key: AnswerKey | None = None
        if avoid_answers or question_decoys:
            if subject.custom:
                raise ExamShieldError("--avoid-answers는 정답을 아는 내장 문항 세트에만 쓴다.")
            if record:
                raise ExamShieldError(
                    "--record는 --avoid-answers와 함께 쓸 수 없다. 검증 기록은 정답표 없이 잰다."
                )
            key = tuple(
                KeyEntry(page, number if question_decoys else None, answer)
                for page, truth in enumerate(subject.truths, start=1)
                for number, answer in truth.questions
            )
        started = datetime.now().astimezone()
        # 응답에는 문항을 옮겨 적은 글이 있다. 사용자의 파일로 잴 때는 적어 두지 않는다.
        checkpoint = (
            None
            if subject.custom
            else Checkpoint(checkpoint_path(out, loaded.name), started, resume)
        )
        if resume and checkpoint is not None:
            typer.echo(
                f"이어 하기: 받아 둔 응답 {len(checkpoint)}건이 있다. 같은 호출은 다시 하지 않는다."
            )
        if cloud:
            _confirm_cloud(loaded, subject, cloud, yes, only)
        everyone = clients + cloud
        progress = Progress(
            sum(estimate_calls(loaded, len(subject.truths), model, only) for model in everyone),
            "모델 호출",
        )
        try:
            report = run_verification(
                loaded,
                subject,
                [
                    cast(ModelClient, Tracked(model, checkpoint, progress.step))
                    for model in everyone
                ],
                now=started,
                skipped=skipped,
                # 워터마크가 든 프로파일은 고정된 가짜 번호의 무늬로 잰다. 무늬가 판독에 주는 영향을 보기 위해서다.
                transform=lambda pdf: protect_bytes(pdf, loaded, None, VERIFY_WATERMARK_KEY, key),
                answer_key=key is not None,
                only=only,
                replies=replies,
            )
        finally:
            progress.close()
        failed_calls = sum(row["errors"] for row in report["results"])
        if checkpoint is not None:
            if checkpoint.reused:
                report["notes"].append(
                    f"이어 하기로, 앞선 실행(24시간 안)에서 받아 둔 응답 {checkpoint.reused}건을 썼다."
                )
            if not failed_calls:
                checkpoint.discard()
        path = write_report(report, out)
        replies_path = write_replies(replies, path) if replies is not None else None
    except ExamShieldError as error:
        raise _fail(error) from error
    typer.echo(format_report(report))
    verdict = _report_verdict(report)
    if verdict:
        typer.echo("\n" + verdict)
    typer.echo(f"\n리포트: {path}")
    if checkpoint is not None and failed_calls:
        typer.echo(
            f"오류가 난 호출이 {failed_calls}건 있다. 같은 명령에 --resume을 붙여 다시 실행하면 "
            "받은 응답은 그대로 쓰고 그 호출만 다시 한다.",
            err=True,
        )
    if replies and replies_path is not None:
        typer.echo("\n" + format_replies(replies))
        typer.echo(f"응답 글: {replies_path}")
    if not any(row["errors"] < row["items"] for row in report["results"]):
        typer.echo("오류: 측정된 값이 하나도 없다. 위의 건너뜀과 오류 수를 확인한다.", err=True)
        raise typer.Exit(1)
    if record:
        refusal = record_refusal(report)
        if refusal is not None:
            typer.echo(f"오류: 검증 기록으로 남기지 않았다. {refusal}", err=True)
            raise typer.Exit(1)
        try:
            typer.echo(f"검증 기록: {write_record(report, RECORD_DIR)}")
        except OSError as error:
            raise _fail(ExamShieldError(f"검증 기록을 쓰지 못했다: {error}")) from error


@auth_app.command("add", help="API 키를 물어서 OS 키체인에 저장한다. 키는 화면에 보이지 않는다.")
def auth_add(provider: str = typer.Argument(..., help="openai, anthropic, google")) -> None:
    try:
        target = get_provider(provider)
        value = typer.prompt(f"{target.company} API 키", hide_input=True)
        store_key(target, value)
    except ExamShieldError as error:
        raise _fail(error) from error
    typer.echo(f"{target.company} 키를 OS 키체인에 저장했다.")


@auth_app.command("list", help="제공사별로 키가 어디에 있는지와 끝 4자리를 보여준다.")
def auth_list() -> None:
    for target in PROVIDERS:
        found = find_key(target)
        state = f"{found.masked}  ({found.source})" if found else "없음"
        typer.echo(f"{target.id:<10} {chosen_model(target):<20} {state}")


@auth_app.command("remove", help="OS 키체인에서 키를 지운다. 환경변수는 건드리지 않는다.")
def auth_remove(provider: str = typer.Argument(..., help="openai, anthropic, google")) -> None:
    try:
        target = get_provider(provider)
        removed = delete_key(target)
    except ExamShieldError as error:
        raise _fail(error) from error
    typer.echo(
        f"{target.company} 키를 OS 키체인에서 지웠다."
        if removed
        else f"OS 키체인에 {target.company} 키가 없다."
    )


@app.command(help="verify가 쓴 리포트 파일을 표로 보여준다.")
def report(report_file: Path = typer.Argument(..., help="리포트 JSON 파일")) -> None:
    try:
        loaded = load_report(report_file)
        typer.echo(format_report(loaded))
        verdict = _report_verdict(loaded)
    except ExamShieldError as error:
        raise _fail(error) from error
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise _fail(ExamShieldError(f"리포트의 형식이 맞지 않는다: {error!r}")) from error
    if verdict:
        typer.echo("\n" + verdict)


@profile_app.command("list", help="프로파일 이름, 시나리오, 상태, 최신 검증 상태를 보여준다.")
def profile_list(
    reports: Path | None = typer.Option(
        None, "--reports", help="verify 리포트가 있는 폴더. 기본값 out/reports"
    ),
) -> None:
    try:
        rows = [("이름", "시나리오", "상태", "검증", "설명")]
        for name in list_profile_names():
            loaded = load_profile(name)
            rows.append(
                (
                    name,
                    loaded.scenario,
                    loaded.status,
                    _assess(loaded, reports).label,
                    loaded.description,
                )
            )
    except ExamShieldError as error:
        raise _fail(error) from error
    widths = [max(display_width(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        cells = [cell + " " * (widths[i] - display_width(cell)) for i, cell in enumerate(row[:4])]
        typer.echo("  ".join([*cells, row[4]]).rstrip())


@profile_app.command("show", help="프로파일 하나의 기법, 해시, 최신 검증 결과와 경고를 보여준다.")
def profile_show(
    name: str = typer.Argument(..., help="프로파일 이름. 예: a-remap-decoy"),
    reports: Path | None = typer.Option(
        None, "--reports", help="verify 리포트가 있는 폴더. 기본값 out/reports"
    ),
) -> None:
    try:
        loaded = load_profile(name)
        assessment = _assess(loaded, reports)
    except ExamShieldError as error:
        raise _fail(error) from error
    typer.echo(f"이름         {loaded.name}")
    typer.echo(f"시나리오     {loaded.scenario}")
    typer.echo(f"상태         {loaded.status}")
    typer.echo(f"설명         {loaded.description}")
    typer.echo(f"해시         {loaded.sha256}")
    typer.echo("기법" + ("" if loaded.techniques else "         없음 (대조군)"))
    for technique in loaded.techniques:
        params = ", ".join(f"{key}={value}" for key, value in technique.params.items())
        typer.echo(f"  - {technique.id}" + (f" ({params})" if params else ""))
    if loaded.capture:
        shown = ", ".join(f"{key}={value}" for key, value in loaded.capture.items())
        typer.echo(f"촬영 조건    {shown}")
    typer.echo(format_assessment(assessment))


def _inputs(source: Path) -> list[Path]:
    if source.is_dir():
        found = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not found:
            raise ExamShieldError(f"폴더에 PDF, .docx, .hwp, .hwpx 파일이 없다: {source}")
        # 대소문자나 유니코드 표기만 다른 이름도 겹친 것으로 본다. 결과 파일이 서로 덮어쓴다.
        stems = [unicodedata.normalize("NFC", path.stem).casefold() for path in found]
        if len(set(stems)) != len(stems):
            raise ExamShieldError(
                "확장자만 다르고 이름이 같은 파일이 함께 있다. 결과 파일 이름이 겹친다."
            )
        return found
    if source.is_file():
        return [source]
    raise ExamShieldError(f"입력을 찾지 못했다: {source}")


@app.command(help="시험 문서의 복사본을 만들고, 원본과 내용이 같은지 바로 검사한다.")
def protect(
    source: Path = typer.Argument(
        ..., metavar="입력", help="PDF, .docx, .hwp, .hwpx 파일, 또는 폴더"
    ),
    out: Path = typer.Option(..., "--out", help="복사본을 쓸 폴더. 원본이 있는 폴더는 안 된다."),
    profile: str = typer.Option(..., "--profile", help="프로파일 이름. 예: a-glyph-remap"),
    seed_file: Path | None = typer.Option(
        None, "--seed-file", help="시드 파일. 같은 입력, 프로파일, 시드면 결과가 같다."
    ),
    force: bool = typer.Option(
        False, "--force", help="출력 폴더에 같은 이름의 파일이 있어도 덮어쓴다."
    ),
    copies: Path | None = typer.Option(
        None,
        "--copies",
        help="수험번호 CSV. 번호마다 고유 워터마크가 든 복사본을 만든다. --seed-file이 필요하다.",
    ),
    answer_key: Path | None = typer.Option(
        None,
        "--answer-key",
        help="정답표 CSV('쪽,정답'). 미끼가 그 쪽의 정답을 가리키지 않게 한다. 출력 폴더 안은 안 된다.",
    ),
    map_file: Path | None = typer.Option(
        None,
        "--map-file",
        help="수험번호 매핑표를 쓸 곳. 기본값 secrets/<이름>.watermark-map.csv. 출력 폴더 안은 안 된다.",
    ),
    jobs: int | None = typer.Option(
        None,
        "--jobs",
        help="--copies에서 복사본을 만들 프로세스 수. 없으면 코어 수와 복사본 수를 보고 고른다. 1이면 하나씩 만든다.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="--copies가 중간에 끊겼을 때 이미 만든 복사본은 두고 나머지만 만든다. 입력, 프로파일, 시드가 같아야 한다.",
    ),
) -> None:
    failed = 0
    loaded: Profile | None = None
    try:
        loaded = load_profile(profile)
        sources = _inputs(source)
        answers = None
        if answer_key is not None:
            if len(sources) != 1:
                raise ExamShieldError("--answer-key는 파일 하나에만 쓴다. 폴더는 받지 않는다.")
            # S5와 같은 이유다. 정답표가 배포하는 폴더에 섞여 나가면 안 된다.
            if _inside(answer_key, out):
                raise ExamShieldError(f"정답표를 출력 폴더 안에 둘 수 없다: {answer_key}")
            answers = read_answer_key(answer_key)
        if copies is not None:
            _protect_copies(
                sources, out, loaded, seed_file, force, copies, map_file, answers, jobs, resume
            )
            return
        if map_file is not None:
            raise ExamShieldError("--map-file은 --copies와 함께 쓴다.")
        if jobs is not None:
            raise ExamShieldError("--jobs는 --copies와 함께 쓴다.")
        if resume:
            raise ExamShieldError("--resume은 --copies와 함께 쓴다.")
        if out.exists() and not out.is_dir():
            raise ExamShieldError(f"--out은 폴더여야 한다: {out}")
        # 대소문자나 유니코드 정규화만 다른 경로도 같은 폴더로 잡으려고 samefile로 비교한다.
        if out.is_dir() and any(out.samefile(path.parent) for path in sources):
            raise ExamShieldError("출력 폴더가 원본이 있는 폴더와 같다. 다른 폴더를 지정한다.")
        if seed_file is not None and not seed_file.is_file():
            raise ExamShieldError(f"시드 파일을 찾지 못했다: {seed_file}")
        seed = seed_file.read_bytes() if seed_file is not None else None
        targets = [out / f"{path.stem}.pdf" for path in sources]
        existing = [str(path) for path in targets if path.exists() or job_path(path).exists()]
        if existing and not force:
            raise ExamShieldError(
                "출력 폴더에 이미 같은 이름의 파일이 있다. 덮어쓰려면 --force를 준다: "
                + ", ".join(existing)
            )
        engine = find_tesseract()
        for path in sources:
            original = read_document(path, out)
            protected, result = protect_and_check(
                original, loaded, seed, engine.read if engine else None, answers
            )
            append_log(out, protected, result.passed, datetime.now().astimezone())
            if not result.passed:
                failed += 1
                typer.echo(f"{path.name}: 검사 실패. 복사본을 쓰지 않았다.", err=True)
                typer.echo(format_result(result), err=True)
                continue
            written = write_outputs(protected, out, path.stem)
            similarity = protected.record["text_layer_similarity"]
            typer.echo(
                f"{path.name} → {written}  검사 통과, "
                f"추출 텍스트가 원문과 일치하는 정도 {similarity:.1%}"
            )
            for note in result.notes:
                typer.echo(f"  참고: {note}")
            if needs_review(path):
                typer.echo(f"  주의: {HWP_NOTICE}")
    except ExamShieldError as error:
        raise _fail(error) from error
    except OSError as error:
        raise _fail(ExamShieldError(f"파일을 읽거나 쓰지 못했다: {error}")) from error
    except typer.Exit:
        raise
    except Exception as error:
        # 깨진 PDF를 PDF 라이브러리가 처리하다 낸 오류. 내용은 내보내지 않는다.
        raise _fail(
            ExamShieldError(f"파일을 처리하지 못했다. 깨진 파일일 수 있다({type(error).__name__}).")
        ) from error
    finally:
        # 경고는 결과를 막지 않는다. 복사본은 시각과 무관하게 같은 바이트로 나온다.
        if loaded is not None:
            _warn_drift(loaded)
    if failed:
        raise typer.Exit(1)


def _warn_drift(profile: Profile) -> None:
    try:
        warnings = _assess(profile).warnings
    except Exception:  # noqa: BLE001 - 경고를 못 만들었다고 protect의 결과를 바꾸지 않는다.
        return
    for warning in warnings:
        typer.echo(f"경고: {warning}", err=True)


def _inside(path: Path, folder: Path) -> bool:
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


def _protect_copies(
    sources: list[Path],
    out: Path,
    profile: Profile,
    seed_file: Path | None,
    force: bool,
    copies: Path,
    map_file: Path | None,
    answers: AnswerKey | None = None,
    jobs: int | None = None,
    resume: bool = False,
) -> None:
    if len(sources) != 1:
        raise ExamShieldError("--copies는 파일 하나에만 쓴다. 폴더는 받지 않는다.")
    source = sources[0]
    keyed_technique(profile)
    if seed_file is not None and not seed_file.is_file():
        raise ExamShieldError(f"시드 파일을 찾지 못했다: {seed_file}")
    seed = check_seed(seed_file.read_bytes() if seed_file is not None else None)
    ids = read_examinee_ids(copies)
    choose_jobs(jobs, len(ids))
    if out.exists() and not out.is_dir():
        raise ExamShieldError(f"--out은 폴더여야 한다: {out}")
    if out.is_dir() and out.samefile(source.parent):
        raise ExamShieldError("출력 폴더가 원본이 있는 폴더와 같다. 다른 폴더를 지정한다.")
    table = map_file or Path("secrets") / f"{source.stem}.watermark-map.csv"
    # S5: 매핑표와 시드는 배포하는 폴더에 두지 않는다.
    for label, path in (("매핑표", table), ("시드 파일", seed_file), ("수험번호 목록", copies)):
        if path is not None and _inside(path, out):
            raise ExamShieldError(
                f"{label}를 출력 폴더 안에 둘 수 없다. 출력 폴더는 배포하는 곳이다: {path}"
            )
    targets = [out / f"{source.stem}-{examinee}.pdf" for examinee in ids]
    existing = [str(path) for path in targets if path.exists() or job_path(path).exists()]
    if table.exists():
        existing.append(str(table))
    if existing and not force and not resume:
        shown = ", ".join(existing[:5]) + (" 외" if len(existing) > 5 else "")
        raise ExamShieldError(
            "이미 같은 이름의 파일이 있다. 끊긴 실행을 이어 하려면 --resume을, "
            f"처음부터 덮어쓰려면 --force를 준다: {shown}"
        )
    # 이번 목록에 없는 예전 복사본이 남으면 매핑표와 폴더가 어긋난다. 지우지는 않고 멈춘다.
    wanted = {path.name for path in targets}
    stale = sorted(
        path.name
        for path in (out.glob(f"{glob.escape(source.stem)}-*.pdf") if out.is_dir() else [])
        if path.name not in wanted
    )
    if stale:
        raise ExamShieldError(
            "출력 폴더에 이번 수험번호 목록에 없는 복사본이 있다. 치우거나 새 폴더를 쓴다: "
            + ", ".join(stale[:5])
        )
    engine = find_tesseract()
    original = read_document(source, out)
    kept = (
        reusable_rows(table, out, source.stem, ids, seed, original, profile, answers is not None)
        if resume
        else {}
    )
    # 매핑표를 쓸 수 있는지 복사본을 만들기 전에 확인한다. 복사본만 남고 매핑표가 없으면 추적할 수 없다.
    # 만드는 동안에도 복사본마다 한 줄씩 보탠다. 끊겨도 그때까지 만든 것은 추적하고 이어 할 수 있다.
    write_map(table, list(kept.values()))
    todo = [examinee for examinee in ids if examinee not in kept]
    if resume:
        typer.echo(
            f"이어 하기: 이미 만든 복사본 {len(kept)}개는 그대로 두고 {len(todo)}개를 만든다."
        )
    made_rows = dict(kept)
    if todo:
        base_check, made = protect_copies(
            original, profile, seed, todo, engine.read if engine else None, answers, jobs
        )
        if not base_check.passed:
            typer.echo(f"{source.name}: 검사 실패. 복사본을 쓰지 않았다.", err=True)
            typer.echo(format_result(base_check), err=True)
            raise typer.Exit(1)
        progress = Progress(len(ids), "복사본", already=len(kept))
        try:
            for copy in made:
                append_log(out, copy.protected, copy.check.passed, datetime.now().astimezone())
                if not copy.check.passed:
                    progress.close()
                    typer.echo(f"{copy.examinee_id}: 워터마크 검사 실패. 중단한다.", err=True)
                    typer.echo(format_result(copy.check), err=True)
                    raise typer.Exit(1)
                written = write_outputs(copy.protected, out, f"{source.stem}-{copy.examinee_id}")
                row = map_row(copy, seed, written.name)
                append_map(table, row)
                made_rows[copy.examinee_id] = row
                progress.step()
        finally:
            progress.close()
    rows = [made_rows[examinee] for examinee in ids]
    write_map(table, rows)
    typer.echo(f"{source.name} → {out}  복사본 {len(rows)}개, 모두 검사 통과")
    if needs_review(source):
        typer.echo(f"주의: {HWP_NOTICE}")
    typer.echo(
        f"수험번호: {ids[0]} ~ {ids[-1]} ({len(ids)}개). 목록의 제목 줄이 번호로 읽히지 않았는지 확인한다."
    )
    typer.echo(f"매핑표: {table}")
    typer.echo("매핑표와 시드 파일은 유출본을 추적할 때 필요하다. 배포하지 말고 따로 보관한다.")


@app.command(
    help="유출된 PDF나 화면 캡처가 어느 수험번호의 복사본인지 찾는다. 못 찾으면 종료 코드 1."
)
def trace(
    leaked: Path = typer.Argument(..., metavar="유출본", help="PDF 또는 그림 파일(png, jpg 등)"),
    original: Path = typer.Option(..., "--original", help="변환 전 원본(PDF, .docx, .hwp, .hwpx)"),
    map_file: Path = typer.Option(..., "--map-file", help="protect --copies가 쓴 매핑표"),
    seed_file: Path = typer.Option(..., "--seed-file", help="protect --copies에 쓴 시드 파일"),
) -> None:
    try:
        for path in (leaked, original, map_file, seed_file):
            if not path.is_file():
                raise ExamShieldError(f"파일을 찾지 못했다: {path}")
        result = trace_document(
            leaked.read_bytes(),
            leaked.suffix,
            read_document(original, leaked.resolve().parent),
            read_map(map_file),
            seed_file.read_bytes(),
        )
    except ExamShieldError as error:
        raise _fail(error) from error
    except OSError as error:
        raise _fail(ExamShieldError(f"파일을 읽거나 쓰지 못했다: {error}")) from error
    except typer.Exit:
        raise
    except Exception as error:
        # 깨진 PDF를 PDF 라이브러리가 처리하다 낸 오류. 내용은 내보내지 않는다.
        raise _fail(
            ExamShieldError(f"파일을 처리하지 못했다. 깨진 파일일 수 있다({type(error).__name__}).")
        ) from error
    typer.echo(format_trace(result))
    if not result.found:
        raise typer.Exit(1)


@app.command(
    help="응시자들의 답안이 숨은 미끼와 얼마나 일치하는지, 그 일치가 우연일 확률을 보여준다."
)
def decoys(
    answers_file: Path = typer.Argument(
        ..., metavar="답안", help="답안 CSV. 한 줄에 '수험번호,쪽,답' 또는 '수험번호,쪽,문항,답'"
    ),
    job: Path | None = typer.Option(
        None, "--job", help="복사본의 작업 기록(.job.json). 모든 응시자가 같은 복사본을 받았을 때"
    ),
    jobs_dir: Path | None = typer.Option(
        None, "--jobs-dir", help="protect --copies의 출력 폴더. --map-file과 함께 쓴다."
    ),
    map_file: Path | None = typer.Option(
        None, "--map-file", help="protect --copies가 쓴 매핑표. --jobs-dir과 함께 쓴다."
    ),
    answer_key: Path | None = typer.Option(
        None,
        "--answer-key",
        help="정답표 CSV. 미끼가 우연히 정답과 같은 문항을 빼고 본다.",
    ),
    choices: int = typer.Option(
        DEFAULT_CHOICES,
        "--choices",
        min=2,
        max=9,
        help="문항의 선택지 수. 적게 줄수록 덜 지목한다.",
    ),
    alpha: float = typer.Option(
        DEFAULT_ALPHA, "--alpha", help="전체에서 정직한 응시자를 잘못 지목할 확률의 한도"
    ),
) -> None:
    try:
        if (job is None) == (jobs_dir is None and map_file is None):
            raise ExamShieldError("--job 하나를 주거나, --jobs-dir과 --map-file을 함께 준다.")
        if job is None and (jobs_dir is None or map_file is None):
            raise ExamShieldError("--jobs-dir과 --map-file은 함께 준다.")
        for path in (answers_file, job, map_file):
            if path is not None and not path.is_file():
                raise ExamShieldError(f"파일을 찾지 못했다: {path}")
        tables: DecoyTable | dict[str, DecoyTable]
        if job is not None:
            tables, key_used = read_decoys(job)
        else:
            assert jobs_dir is not None and map_file is not None
            if not jobs_dir.is_dir():
                raise ExamShieldError(f"폴더를 찾지 못했다: {jobs_dir}")
            rows = [(row.examinee_id, row.file) for row in read_map(map_file)]
            tables, key_used = load_tables(rows, jobs_dir)
        result = analyse(
            read_responses(answers_file),
            tables,
            key_used,
            read_answer_key(answer_key) if answer_key is not None else None,
            choices,
            alpha,
        )
    except ExamShieldError as error:
        raise _fail(error) from error
    typer.echo(format_analysis(result))


@app.command(
    help="인쇄해서 찍은 사진의 촬영 조건을 재서 프로파일의 촬영 시뮬레이션과 나란히 보여준다."
)
def calibrate(
    photo: Path = typer.Argument(..., metavar="사진", help="인쇄물 한 쪽을 찍은 사진(jpg, png 등)"),
    printed: Path = typer.Option(..., "--printed", help="인쇄에 쓴 PDF(원본 또는 복사본)"),
    profile: str = typer.Option(
        "none-print", "--profile", help="촬영 조건을 견줄 시나리오 B 프로파일"
    ),
    page: int | None = typer.Option(
        None, "--page", min=1, help="사진에 찍힌 쪽 번호. 없으면 가장 잘 맞는 쪽을 찾는다."
    ),
) -> None:
    try:
        loaded = load_profile(profile)
        if loaded.scenario != "B":
            raise ExamShieldError(
                f"프로파일 '{profile}'은 시나리오 {loaded.scenario}용이라 촬영 조건이 없다. "
                "none-print나 b-dots 같은 시나리오 B 프로파일을 준다."
            )
        for path in (photo, printed):
            if not path.is_file():
                raise ExamShieldError(f"파일을 찾지 못했다: {path}")
        conditions = conditions_from(loaded.capture) if loaded.capture else CaptureConditions()
        document = printed.read_bytes()
        pages, texts = reference_pages(document)
        engine = find_tesseract()
        read = engine.read if engine else None
        real = measure(
            decode_photo(photo.read_bytes()), pages, texts, read, None if page is None else page - 1
        )
        model = simulated(document, real.page_index, conditions, read)
    except ExamShieldError as error:
        raise _fail(error) from error
    except OSError as error:
        raise _fail(ExamShieldError(f"파일을 읽지 못했다: {error}")) from error
    typer.echo(format_comparison(real, model))
    typer.echo(f"\n이 사진에 맞춘 촬영 조건(어림값). 프로파일 '{profile}'의 capture에 넣어 쓴다.")
    typer.echo("capture:")
    for key, value in suggest(real, conditions).items():
        typer.echo(f"  {key}: {value}")
    if read is None:
        typer.echo("참고: Tesseract가 없어 OCR 판독률은 재지 않았다.")


@app.command(help="이 PC에서만 열리는 웹 화면을 띄운다. 복사본 만들기와 동일성 검사를 할 수 있다.")
def ui(
    port: int = typer.Option(0, "--port", min=0, max=65535, help="포트. 0이면 빈 포트를 고른다."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="브라우저를 열지 않고 주소만 보여준다."
    ),
    work_dir: Path = typer.Option(
        Path("out/ui-work"),
        "--work-dir",
        help="Word·한글 파일을 PDF로 바꿀 때 임시 파일을 두는 폴더. 끝나면 지운다.",
    ),
) -> None:
    try:
        server, address = make_server(work_dir, port)
    except OSError as error:
        raise _fail(ExamShieldError(f"웹 화면을 띄우지 못했다: {error}")) from error
    typer.echo(f"웹 화면: {address}")
    typer.echo(
        "이 PC에서만 열린다(127.0.0.1). 주소에 든 토큰이 있어야 쓸 수 있다. 끝내려면 Ctrl+C."
    )
    if not no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\n끝냈다.")
    finally:
        server.server_close()


@app.command("license", help="라이선스와 소스 코드를 받는 곳을 보여준다.")
def show_license() -> None:
    typer.echo(f"examshield {__version__}")
    typer.echo("GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)")
    typer.echo(
        "이 프로그램은 자유 소프트웨어다. 보증은 없다. 고쳐서 배포하거나 네트워크로 서비스하면"
    )
    typer.echo("고친 소스를 같은 조건으로 공개해야 한다.")
    typer.echo("소스 코드: https://github.com/Trapa-Eureka/Gotcha")
    typer.echo(
        "PDF 처리에 PyMuPDF(AGPL-3.0, Artifex Software)를 쓴다: https://github.com/pymupdf/PyMuPDF"
    )
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "LICENSE"
    if bundled.is_file():
        typer.echo(f"라이선스 전문: {bundled}")
        if getattr(sys, "frozen", False):
            typer.echo("")
            typer.echo(bundled.read_text(encoding="utf-8"))


@app.command(help="원본과 복사본의 내용이 같은지 검사한다. 다르면 종료 코드 1.")
def check(
    original: Path = typer.Argument(..., metavar="원본", help="원본(PDF, .docx, .hwp, .hwpx)"),
    copy: Path = typer.Argument(..., metavar="복사본", help="protect가 만든 PDF"),
    job: Path | None = typer.Option(
        None, "--job", help="작업 기록(.job.json). 없으면 복사본 옆의 같은 이름 파일을 찾는다."
    ),
) -> None:
    try:
        for path in (original, copy):
            if not path.is_file():
                raise ExamShieldError(f"파일을 찾지 못했다: {path}")
        copy_bytes = copy.read_bytes()
        job_file = job if job is not None else job_path(copy)
        if job is not None and not job.is_file():
            raise ExamShieldError(f"작업 기록을 찾지 못했다: {job}")
        expect = load_expectation(job_file, copy_bytes) if job_file.is_file() else Expectation()
        original_bytes = read_document(original, copy.resolve().parent)
        engine = find_tesseract()
        result = compare(original_bytes, copy_bytes, expect, engine.read if engine else None)
    except ExamShieldError as error:
        raise _fail(error) from error
    except OSError as error:
        raise _fail(ExamShieldError(f"파일을 읽거나 쓰지 못했다: {error}")) from error
    except typer.Exit:
        raise
    except Exception as error:
        # 깨진 PDF를 PDF 라이브러리가 처리하다 낸 오류. 내용은 내보내지 않는다.
        raise _fail(
            ExamShieldError(f"파일을 처리하지 못했다. 깨진 파일일 수 있다({type(error).__name__}).")
        ) from error
    typer.echo(format_result(result))
    if not result.passed:
        raise typer.Exit(1)
