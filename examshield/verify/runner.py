import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pymupdf

from examshield import __version__
from examshield.benchmark import ItemSet, page_text_of, render_items_pdf
from examshield.errors import ExamShieldError, ModelAbortError, ModelUnavailableError
from examshield.pdfio import open_pdf_bytes, page_text, render_gray, save_deterministic
from examshield.profiles import Profile
from examshield.protect.pipeline import Protected
from examshield.protect.techniques.decoy import DEFAULT_CHOICES, choice_count
from examshield.verify.capture_sim import (
    CaptureConditions,
    conditions_from,
    encode_png,
    simulate_capture,
)
from examshield.verify.metrics import legibility, mean
from examshield.verify.models.base import (
    FILE,
    IMAGE,
    TEXT,
    ModelClient,
    ModelReply,
    PageInput,
)

REPORT_SCHEMA = 1
SCREEN_DPI = 150
CONTROL = "control"
PROTECTED = "protected"

Transform = Callable[[bytes], Protected]


@dataclass(frozen=True)
class Skipped:
    model: str
    reason: str


@dataclass(frozen=True)
class Truth:
    text: str
    answer: int | None
    choice_count: int
    # 이 쪽에 있는 문항들의 (번호, 정답). 사용자가 준 파일에는 없다.
    questions: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Subject:
    """측정에 쓰는 문서. 내장 문항 세트이거나 사용자가 --allow-custom으로 준 파일이다."""

    version: str
    sha256: str
    pdf: bytes
    truths: tuple[Truth, ...]
    custom: bool = False
    # 한 쪽에 문항이 여럿인 문항 세트인지. 그러면 문항마다 답을 받고 문항마다 채점한다.
    multi: bool = False


def builtin_subject(item_set: ItemSet) -> Subject:
    multi = item_set.per_page > 1
    return Subject(
        version=item_set.version,
        sha256=item_set.sha256,
        pdf=render_items_pdf(item_set.items, item_set.per_page),
        truths=tuple(
            Truth(
                text=page_text_of(group),
                answer=None if multi else group[0].answer,
                choice_count=max(len(item.choices) for item in group),
                questions=tuple((item.number, item.answer) for item in group),
            )
            for group in item_set.pages
        ),
        multi=multi,
    )


def custom_subject(pdf: bytes) -> Subject:
    try:
        doc = open_pdf_bytes(pdf)
    except Exception as error:
        raise ExamShieldError("--allow-custom 파일을 PDF로 열지 못했다.") from error
    texts = [page_text(page) for page in doc]
    if not texts:
        raise ExamShieldError("--allow-custom 파일에 쪽이 없다.")
    return Subject(
        version="custom",
        sha256=hashlib.sha256(pdf).hexdigest(),
        pdf=pdf,
        truths=tuple(
            Truth(text=text, answer=None, choice_count=choice_count(text, DEFAULT_CHOICES))
            for text in texts
        ),
        custom=True,
    )


def channels_for(profile: Profile) -> tuple[str, ...]:
    return (IMAGE,) if profile.scenario == "B" else (TEXT, IMAGE, FILE)


def chosen_channels(profile: Profile, only: tuple[str, ...] | None) -> tuple[str, ...]:
    """프로파일이 재는 경로 가운데 only에 든 것만. only가 없으면 전부."""
    available = channels_for(profile)
    if only is None:
        return available
    unknown = [channel for channel in only if channel not in (TEXT, IMAGE, FILE)]
    if unknown:
        raise ExamShieldError(
            f"모르는 경로다: {', '.join(unknown)}. text, image, file 중에서 고른다."
        )
    picked = tuple(channel for channel in available if channel in only)
    if not picked:
        raise ExamShieldError(
            f"프로파일 '{profile.name}'은 고른 경로를 재지 않는다. "
            f"이 프로파일이 재는 경로: {', '.join(available)}"
        )
    return picked


def estimate_calls(
    profile: Profile, pages: int, model: ModelClient, only: tuple[str, ...] | None = None
) -> int:
    variants = 2 if profile.techniques else 1
    used = [channel for channel in chosen_channels(profile, only) if channel in model.info.channels]
    return pages * variants * len(used)


def run_verification(
    profile: Profile,
    subject: Subject | ItemSet,
    models: list[ModelClient],
    *,
    now: datetime,
    skipped: list[Skipped] | None = None,
    transform: Transform | None = None,
    only: tuple[str, ...] | None = None,
    answer_key: bool = False,
    replies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """replies에 목록을 주면 호출마다 모델이 돌려준 글과 문항별 채점을 거기에 담는다."""
    if isinstance(subject, ItemSet):
        subject = builtin_subject(subject)
    if replies is not None and subject.custom:
        # 응답에는 문서를 옮겨 적은 글이 들어 있다. 사용자의 시험지를 파일로 남기지 않는다.
        raise ExamShieldError("응답 글은 내장 문항 세트로 잴 때만 남긴다.")
    if profile.techniques and transform is None:
        raise ExamShieldError(
            f"프로파일 '{profile.name}'에는 교란 기법이 있는데, 이를 적용할 변환 함수가 없다."
        )
    conditions = _capture_conditions(profile)
    channels = chosen_channels(profile, only)
    variants = {CONTROL: subject.pdf}
    decoys: dict[str, tuple[int, ...] | None] = {CONTROL: None}
    tables: dict[str, dict[tuple[int, int | None], int] | None] = {CONTROL: None}
    protection: dict[str, Any] | None = None
    if transform is not None and profile.techniques:
        protected = transform(subject.pdf)
        variants[PROTECTED] = protected.pdf
        decoys[PROTECTED] = protected.decoy_answers
        tables[PROTECTED] = (
            {(page, question): answer for page, question, answer in protected.decoy_items}
            if protected.decoy_items is not None
            else None
        )
        protection = {
            "techniques": protected.record["techniques"],
            "text_layer_similarity": protected.record["text_layer_similarity"],
        }

    pages = {name: _page_inputs(pdf, subject, conditions) for name, pdf in variants.items()}
    all_skipped = list(skipped or [])
    results: list[dict[str, Any]] = []
    measured: list[ModelClient] = []
    for model in models:
        rows: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        try:
            for channel in channels:
                if channel not in model.info.channels:
                    continue
                for variant, inputs in pages.items():
                    rows.append(
                        _measure(
                            model,
                            channel,
                            variant,
                            inputs,
                            subject,
                            decoys[variant],
                            tables[variant],
                            kept if replies is not None else None,
                        )
                    )
        except ModelAbortError as error:
            all_skipped.append(Skipped(model.info.name, str(error)))
            continue
        results += rows
        measured.append(model)
        if replies is not None:
            replies += kept

    notes = []
    if PROTECTED not in variants:
        notes.append("변환 없는 프로파일이라 대조군만 측정했다.")
    if subject.custom:
        notes.append("사용자가 준 파일로 측정했다. 정답을 모르므로 정답률은 없다.")
    limited = channels != channels_for(profile)
    if limited:
        notes.append(
            f"경로를 골라서 쟀다({', '.join(channels)}). 재지 않은 경로가 있어 프로파일의 "
            "검증 기록으로는 쓰지 않는다."
        )
    if answer_key:
        notes.append(
            "정답표를 준 것처럼 쟀다. 미끼가 정답을 가리키지 않는다. 검증 기록으로는 쓰지 않는다."
        )
    refusals = sum(row["refusals"] for row in results)
    if refusals:
        notes.append(
            f"모델이 답하기를 거부한 호출이 {refusals}건 있다. 판독률 0, 오답, 미끼에 걸리지 않음으로 셌다."
        )
    return {
        "schema": REPORT_SCHEMA,
        "created_at": now.isoformat(timespec="seconds"),
        "tool_version": __version__,
        "profile": {
            "name": profile.name,
            "sha256": profile.sha256,
            "scenario": profile.scenario,
            "status": profile.status,
        },
        "benchmark": {
            "version": subject.version,
            "sha256": subject.sha256,
            "item_count": len(subject.truths),
        },
        "simulation": conditions.as_dict() if conditions else None,
        "channels_limited": limited,
        "answer_key": answer_key,
        "protection": protection,
        "models": [
            {
                "provider": model.info.provider,
                "name": model.info.name,
                "version": model.info.version,
                "kind": model.info.kind,
                "note": model.info.note,
            }
            for model in measured
        ],
        "results": results,
        "skipped": [{"model": entry.model, "reason": entry.reason} for entry in all_skipped],
        "notes": notes,
    }


def _capture_conditions(profile: Profile) -> CaptureConditions | None:
    if profile.scenario != "B":
        return None
    return conditions_from(profile.capture) if profile.capture else CaptureConditions()


def _page_inputs(
    pdf: bytes, subject: Subject, conditions: CaptureConditions | None
) -> list[PageInput]:
    doc = open_pdf_bytes(pdf)
    if doc.page_count != len(subject.truths):
        raise ExamShieldError("변환 후 쪽수가 원래 쪽수와 다르다.")
    inputs = []
    for number, (page, truth) in enumerate(zip(doc, subject.truths, strict=True)):
        if conditions is None:
            image = render_gray(page, SCREEN_DPI)
        else:
            image = simulate_capture(render_gray(page, conditions.print_dpi), conditions)
        single = pymupdf.open()
        single.insert_pdf(doc, from_page=number, to_page=number)
        inputs.append(
            PageInput(
                text=page_text(page),
                image_png=encode_png(image),
                choice_count=truth.choice_count,
                pdf=save_deterministic(single),
                questions=tuple(number for number, _ in truth.questions) if subject.multi else (),
            )
        )
    return inputs


def _decoy_for(
    index: int,
    number: int | None,
    multi: bool,
    decoys: tuple[int, ...] | None,
    table: dict[tuple[int, int | None], int] | None,
) -> int | None:
    if multi:
        if table is None:
            return None
        return table.get((index + 1, number), table.get((index + 1, None)))
    return decoys[index] if decoys is not None else None


def _kept(
    model: ModelClient,
    channel: str,
    variant: str,
    index: int,
    page: PageInput,
    truth: Truth,
    reply: ModelReply,
    decoys: tuple[int, ...] | None,
    table: dict[tuple[int, int | None], int] | None,
) -> dict[str, Any]:
    """호출 한 번의 기록. 왜 그런 숫자가 나왔는지 사람이 읽어 보기 위한 것이다."""
    multi = bool(page.questions)
    questions = []
    for number, answer in truth.questions:
        questions.append(
            {
                "number": number,
                "answer": answer,
                "decoy": _decoy_for(index, number, multi, decoys, table),
                "given": reply.answers.get(number) if multi else reply.answer,
            }
        )
    return {
        "model": model.info.name,
        "solves": model.info.solves,
        "channel": channel,
        "variant": variant,
        "page": index + 1,
        "refused": reply.refused,
        # 텍스트 복사 경로에서는 모델이 받은 글도 남긴다. 미끼가 어디에 어떻게 보였는지 볼 수 있다.
        "input_text": page.text if channel == TEXT else None,
        "reply": reply.raw or reply.transcription,
        "questions": questions,
    }


def _measure(
    model: ModelClient,
    channel: str,
    variant: str,
    inputs: list[PageInput],
    subject: Subject,
    decoys: tuple[int, ...] | None,
    table: dict[tuple[int, int | None], int] | None = None,
    kept: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scores: list[float] = []
    correct: list[float] = []
    lured: list[float] = []
    errors = 0
    refusals = 0
    seconds = 0.0
    tokens: list[tuple[int, int]] = []
    for index, (page, truth) in enumerate(zip(inputs, subject.truths, strict=True)):
        began = time.perf_counter()
        try:
            reply = model.run(page, channel)
        except ModelUnavailableError:
            seconds += time.perf_counter() - began
            errors += 1
            continue
        # 받아 둔 응답을 다시 쓴 호출은 처음 받을 때 걸린 시간을 센다.
        seconds += reply.seconds if reply.seconds is not None else time.perf_counter() - began
        if reply.tokens is not None:
            tokens.append(reply.tokens)
        refusals += 1 if reply.refused else 0
        scores.append(legibility(reply.transcription, truth.text))
        if kept is not None:
            kept.append(_kept(model, channel, variant, index, page, truth, reply, decoys, table))
        if subject.multi:
            # 문항마다 채점한다. 미끼는 그 문항을 가리키는 것이 있으면 그것을, 없으면 그 쪽의 것을 본다.
            for number, answer in truth.questions:
                given = reply.answers.get(number)
                correct.append(1.0 if given == answer else 0.0)
                decoy = None
                if table is not None:
                    decoy = table.get((index + 1, number), table.get((index + 1, None)))
                if decoy is not None and decoy != answer:
                    lured.append(1.0 if given == decoy else 0.0)
            continue
        if truth.answer is not None:
            correct.append(1.0 if reply.answer == truth.answer else 0.0)
        # 미끼가 우연히 정답을 가리킨 문항은 유도됐는지 알 수 없어 뺀다.
        if decoys is not None and decoys[index] != truth.answer:
            lured.append(1.0 if reply.answer == decoys[index] else 0.0)
    return {
        "model": model.info.name,
        "channel": channel,
        "variant": variant,
        "items": len(inputs),
        "questions": sum(len(truth.questions) or 1 for truth in subject.truths),
        "errors": errors,
        "refusals": refusals,
        "legibility": mean(scores),
        "accuracy": mean(correct) if model.info.solves else None,
        # 정답률과 미끼 적중률의 분자와 분모. 문항이 적으면 비율만 봐서는 크기를 알 수 없다.
        "correct": int(sum(correct)) if model.info.solves and correct else None,
        "scored": len(correct) if model.info.solves and correct else None,
        "decoy_hit_rate": mean(lured) if model.info.solves and decoys is not None else None,
        "decoy_hits": int(sum(lured)) if model.info.solves and decoys is not None else None,
        "decoy_items": len(lured) if decoys is not None else None,
        "seconds": round(seconds, 1),
        # 토큰 수를 알려 준 호출만 더한다. 하나도 없으면 None.
        "input_tokens": sum(pair[0] for pair in tokens) if tokens else None,
        "output_tokens": sum(pair[1] for pair in tokens) if tokens else None,
        "token_calls": len(tokens),
    }
