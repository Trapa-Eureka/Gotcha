from datetime import datetime
from enum import StrEnum
from pathlib import Path

import typer

from examshield.benchmark import load_items
from examshield.errors import ExamShieldError, ModelUnavailableError
from examshield.profiles import load_profile
from examshield.verify.models.base import ModelClient
from examshield.verify.models.baseline import TesseractModel, TextLayerModel
from examshield.verify.models.local import OllamaModel
from examshield.verify.models.mock import MockModel
from examshield.verify.report import format_report, load_report, write_report
from examshield.verify.runner import Skipped, run_verification

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="시험 문서 AI 판독 저항 CLI. 내용이 같은 복사본을 만들고 저항 효과를 수치로 잰다.",
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


@app.command(help="내장 문항 세트로 프로파일의 저항 효과를 측정하고 리포트를 쓴다.")
def verify(
    profile: str = typer.Option(..., "--profile", help="프로파일 이름. 예: none, none-print"),
    models: ModelGroup = typer.Option(
        ModelGroup.local,
        "--models",
        help="local, cloud, all. mock은 실제 AI가 아닌 점검용 모의 모델.",
    ),
    out: Path = typer.Option(Path("out/reports"), "--out", help="리포트를 쓸 폴더"),
    local_model: str | None = typer.Option(
        None, "--local-model", help="Ollama 모델 이름. 없으면 설치된 첫 모델을 쓴다."
    ),
) -> None:
    try:
        if models in (ModelGroup.cloud, ModelGroup.all):
            raise ExamShieldError(
                "클라우드 모델 검증은 아직 구현되지 않았다(Phase 3). --models local을 쓴다."
            )
        loaded = load_profile(profile)
        item_set = load_items()
        if models is ModelGroup.mock:
            clients: list[ModelClient] = [TextLayerModel(), MockModel(item_set.items)]
            skipped: list[Skipped] = []
        else:
            clients, skipped = _local_models(local_model)
        report = run_verification(
            loaded, item_set, clients, now=datetime.now().astimezone(), skipped=skipped
        )
        path = write_report(report, out)
    except ExamShieldError as error:
        raise _fail(error) from error
    typer.echo(format_report(report))
    typer.echo(f"\n리포트: {path}")


@app.command(help="verify가 쓴 리포트 파일을 표로 보여준다.")
def report(report_file: Path = typer.Argument(..., help="리포트 JSON 파일")) -> None:
    try:
        typer.echo(format_report(load_report(report_file)))
    except ExamShieldError as error:
        raise _fail(error) from error
