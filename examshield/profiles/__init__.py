import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from examshield.errors import ExamShieldError

PROFILE_DIR = Path(__file__).resolve().parent
SCENARIOS = ("A", "B")
STATUSES = ("active", "experimental", "deprecated")
CHANNELS = ("text", "image", "file")


@dataclass(frozen=True)
class Technique:
    id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    name: str
    scenario: str
    status: str
    description: str
    techniques: tuple[Technique, ...]
    capture: dict[str, Any] | None
    sha256: str
    # 이 프로파일이 막겠다고 하는 공격 경로. 비어 있으면 잰 경로 전부를 막아야 통과다.
    blocks: tuple[str, ...] = ()


def list_profile_names() -> list[str]:
    return sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))


def load_profile(name: str) -> Profile:
    path = PROFILE_DIR / f"{name}.yaml"
    # 이름을 경로로 쓰기 전에 목록과 대조한다. '../x'나 절대 경로가 다른 파일을 가리키지 못한다.
    if name not in list_profile_names():
        known = ", ".join(list_profile_names())
        raise ExamShieldError(f"프로파일 '{name}'을(를) 찾지 못했다. 있는 프로파일: {known}")
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_profile(data)


def parse_profile(data: dict[str, Any]) -> Profile:
    scenario = str(data.get("scenario", ""))
    status = str(data.get("status", ""))
    if scenario not in SCENARIOS:
        raise ExamShieldError(f"프로파일의 scenario는 {SCENARIOS} 중 하나여야 한다: {scenario!r}")
    if status not in STATUSES:
        raise ExamShieldError(f"프로파일의 status는 {STATUSES} 중 하나여야 한다: {status!r}")
    techniques = tuple(
        Technique(id=str(entry["id"]), params=dict(entry.get("params") or {}))
        for entry in data.get("techniques") or []
    )
    capture = dict(data["capture"]) if data.get("capture") else None
    # 해시는 복사본을 바꾸는 값으로만 만든다. 이름, 상태, 설명은 넣지 않는다.
    # 넣으면 상태를 deprecated로 바꾸는 순간 그 근거가 된 검증 기록이 다른 프로파일의 것이 되고,
    # 같은 원본에서 나오는 복사본도 달라진다.
    behavior = {
        "scenario": scenario,
        "techniques": [{"id": entry.id, "params": entry.params} for entry in techniques],
        "capture": capture,
    }
    canonical = json.dumps(behavior, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    blocks = tuple(str(entry) for entry in data.get("blocks") or ())
    if any(entry not in CHANNELS for entry in blocks):
        raise ExamShieldError(f"프로파일의 blocks는 {CHANNELS} 중에서 고른다: {blocks!r}")
    return Profile(
        blocks=blocks,
        name=str(data["name"]),
        scenario=scenario,
        status=status,
        description=str(data.get("description", "")),
        techniques=techniques,
        capture=capture,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
