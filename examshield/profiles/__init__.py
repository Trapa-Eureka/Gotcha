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


def list_profile_names() -> list[str]:
    return sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))


def load_profile(name: str) -> Profile:
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.is_file():
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
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return Profile(
        name=str(data["name"]),
        scenario=scenario,
        status=status,
        description=str(data.get("description", "")),
        techniques=tuple(
            Technique(id=str(entry["id"]), params=dict(entry.get("params") or {}))
            for entry in data.get("techniques") or []
        ),
        capture=dict(data["capture"]) if data.get("capture") else None,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
