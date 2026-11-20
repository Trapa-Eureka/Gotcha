from dataclasses import dataclass, field
from typing import Any

TEXT_PRESERVED = "preserved"
TEXT_ALTERED = "altered"


@dataclass(frozen=True)
class Context:
    seed: bytes
    original_texts: tuple[str, ...]
    watermark_key: bytes | None = None


@dataclass(frozen=True)
class Outcome:
    pdf: bytes
    stats: dict[str, Any] = field(default_factory=dict)
    text: str = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoy_texts: tuple[str, ...] | None = None
    decoy_answers: tuple[int, ...] | None = None
    overlay_images: int = 0
    pixel_tolerance: int | None = None
    # 도트 무늬를 덧씌웠으면 그 칸 크기와 진하기. check가 무늬를 걷어 내고 비교한다.
    dots: dict[str, int] | None = None
