from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from examshield import __version__
from examshield.benchmark import ItemSet, render_items_pdf
from examshield.errors import ExamShieldError, ModelUnavailableError
from examshield.pdfio import open_pdf_bytes, page_text, render_gray
from examshield.profiles import Profile
from examshield.verify.capture_sim import (
    CaptureConditions,
    conditions_from,
    encode_png,
    simulate_capture,
)
from examshield.verify.metrics import legibility, mean
from examshield.verify.models.base import IMAGE, TEXT, ModelClient, PageInput

REPORT_SCHEMA = 1
SCREEN_DPI = 150
CONTROL = "control"
PROTECTED = "protected"

Transform = Callable[[bytes], bytes]


@dataclass(frozen=True)
class Skipped:
    model: str
    reason: str


def run_verification(
    profile: Profile,
    item_set: ItemSet,
    models: list[ModelClient],
    *,
    now: datetime,
    skipped: list[Skipped] | None = None,
    transform: Transform | None = None,
) -> dict[str, Any]:
    if profile.techniques and transform is None:
        raise ExamShieldError(
            f"프로파일 '{profile.name}'에는 교란 기법이 있는데, 이를 적용하는 protect가 아직 없다."
        )
    conditions = _capture_conditions(profile)
    control_pdf = render_items_pdf(item_set.items)
    variants = {CONTROL: control_pdf}
    if transform is not None and profile.techniques:
        variants[PROTECTED] = transform(control_pdf)

    pages = {name: _page_inputs(pdf, item_set, conditions) for name, pdf in variants.items()}
    channels = (IMAGE,) if profile.scenario == "B" else (TEXT, IMAGE)

    results: list[dict[str, Any]] = []
    for model in models:
        for channel in channels:
            if channel not in model.info.channels:
                continue
            for variant, inputs in pages.items():
                results.append(_measure(model, channel, variant, inputs, item_set))

    notes = []
    if PROTECTED not in variants:
        notes.append("변환 없는 프로파일이라 대조군만 측정했다.")
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
            "version": item_set.version,
            "sha256": item_set.sha256,
            "item_count": len(item_set.items),
        },
        "simulation": conditions.as_dict() if conditions else None,
        "models": [
            {
                "provider": model.info.provider,
                "name": model.info.name,
                "version": model.info.version,
                "kind": model.info.kind,
                "note": model.info.note,
            }
            for model in models
        ],
        "results": results,
        "skipped": [{"model": entry.model, "reason": entry.reason} for entry in skipped or []],
        "notes": notes,
    }


def _capture_conditions(profile: Profile) -> CaptureConditions | None:
    if profile.scenario != "B":
        return None
    return conditions_from(profile.capture) if profile.capture else CaptureConditions()


def _page_inputs(
    pdf: bytes, item_set: ItemSet, conditions: CaptureConditions | None
) -> list[PageInput]:
    doc = open_pdf_bytes(pdf)
    if doc.page_count != len(item_set.items):
        raise ExamShieldError("변환 후 쪽수가 문항 수와 다르다.")
    inputs = []
    for page, item in zip(doc, item_set.items, strict=True):
        if conditions is None:
            image = render_gray(page, SCREEN_DPI)
        else:
            image = simulate_capture(render_gray(page, conditions.print_dpi), conditions)
        inputs.append(
            PageInput(
                text=page_text(page), image_png=encode_png(image), choice_count=len(item.choices)
            )
        )
    return inputs


def _measure(
    model: ModelClient, channel: str, variant: str, inputs: list[PageInput], item_set: ItemSet
) -> dict[str, Any]:
    scores: list[float] = []
    correct: list[float] = []
    errors = 0
    for page, item in zip(inputs, item_set.items, strict=True):
        try:
            reply = model.run(page, channel)
        except ModelUnavailableError:
            errors += 1
            continue
        scores.append(legibility(reply.transcription, item.text))
        correct.append(1.0 if reply.answer == item.answer else 0.0)
    return {
        "model": model.info.name,
        "channel": channel,
        "variant": variant,
        "items": len(inputs),
        "errors": errors,
        "legibility": mean(scores),
        "accuracy": mean(correct) if model.info.solves else None,
        "decoy_hit_rate": None,
    }
