from dataclasses import dataclass, field
from typing import Any

TEXT_PRESERVED = "preserved"
TEXT_ALTERED = "altered"


@dataclass(frozen=True)
class Context:
    seed: bytes
    original_texts: tuple[str, ...]
    watermark_key: bytes | None = None
    # 쪽마다 미끼가 가리키면 안 되는 번호(그 쪽에 있는 문항들의 정답). 정답표가 없으면 None.
    avoid_answers: tuple[frozenset[int], ...] | None = None
    # 쪽마다 (문항 번호, 정답). 정답표에 문항 번호가 있을 때만 있다. 있으면 문항마다 미끼를 넣는다.
    questions: tuple[tuple[tuple[int, int], ...], ...] | None = None


@dataclass(frozen=True)
class Outcome:
    pdf: bytes
    stats: dict[str, Any] = field(default_factory=dict)
    text: str = TEXT_PRESERVED
    flattened_dpi: int | None = None
    decoy_texts: tuple[str, ...] | None = None
    decoy_answers: tuple[int, ...] | None = None
    # 미끼 하나하나: (쪽, 문항 번호 또는 None, 미끼 번호). 쪽은 1부터 센다.
    decoy_items: tuple[tuple[int, int | None, int], ...] | None = None
    overlay_images: int = 0
    pixel_tolerance: int | None = None
    # 도트 무늬를 덧씌웠으면 그 칸 크기와 진하기. check가 무늬를 걷어 내고 비교한다.
    dots: dict[str, int] | None = None
