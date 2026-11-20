from dataclasses import dataclass, field
from typing import Any

TEXT_PRESERVED = "preserved"
TEXT_ALTERED = "altered"


@dataclass(frozen=True)
class Context:
    seed: bytes
    original_texts: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    pdf: bytes
    stats: dict[str, Any] = field(default_factory=dict)
    text: str = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoy_texts: tuple[str, ...] | None = None
    decoy_answers: tuple[int, ...] | None = None
