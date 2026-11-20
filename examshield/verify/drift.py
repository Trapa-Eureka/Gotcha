"""프로파일의 최신 검증 결과를 찾아, 아직 믿을 만한지 판정한다(명세 7절).

교란 효과는 특정 시점의 특정 모델을 상대로 잰 값이다. 그래서 오래됐거나, 도구나 문항 세트나
프로파일이 바뀐 뒤의 결과는 경고와 함께 보여준다.
"""

import json
import math
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from examshield.errors import ExamShieldError
from examshield.profiles import PROFILE_DIR, Profile, list_profile_names

RECORD_DIR = PROFILE_DIR / "verified"
DEFAULT_REPORT_DIR = Path("out/reports")
MAX_AGE_DAYS = 30
# 변환본의 정답률이 대조군의 이 비율 이하로 떨어져야 그 모델을 막았다고 본다.
HOLD_RATIO = 0.5
AI_KINDS = ("local", "relayed", "cloud")
# 유출 추적용 기법. AI 판독을 막는 기법이 아니라서 판독 저항 판정의 대상이 아니다.
# dots는 Claude로 재 보니 판독을 전혀 방해하지 못했다(2026-11-21, verified/b-dots.json).
TRACKING_ONLY = ("watermark", "dots")
MAX_REPORT_BYTES = 5_000_000
MAX_CLOCK_SKEW = timedelta(days=1)
VARIANTS = ("control", "protected")

UNVERIFIED = "unverified"
PASSED = "passed"
PARTIAL = "partial"
FAILED = "failed"
NOT_APPLICABLE = "not_applicable"
STATE_LABELS = {
    UNVERIFIED: "검증 안 됨",
    PASSED: "기준 통과",
    # 막겠다고 한 경로는 막았지만, 공격자가 쓸 수 있는 다른 경로가 열려 있다.
    PARTIAL: "일부 경로만 막음",
    FAILED: "기준 미달",
    NOT_APPLICABLE: "대상 아님",
}


@dataclass(frozen=True)
class ModelVerdict:
    """AI 모델 하나가 공격 경로 하나로 푼 결과."""

    model: str
    kind: str
    channel: str
    control: float | None
    protected: float | None
    # True 막음, False 못 막음, None 판정 못 함
    held: bool | None
    reason: str = ""


@dataclass(frozen=True)
class Assessment:
    state: str
    created_at: datetime | None
    age_days: int | None
    models: tuple[ModelVerdict, ...]
    warnings: tuple[str, ...]
    source: Path | None = None

    @property
    def label(self) -> str:
        text = STATE_LABELS[self.state]
        if self.created_at is None:
            return text
        day = self.created_at.date().isoformat()
        stale = ", 오래됨" if self.age_days is not None and self.age_days > MAX_AGE_DAYS else ""
        return f"{text} ({day}, {self.age_days}일 전{stale})"


@dataclass(frozen=True)
class Current:
    """지금 쓰는 도구와 문항 세트. 리포트가 만들어진 때와 달라졌는지 본다."""

    tool_version: str
    benchmark_version: str
    benchmark_sha256: str


@dataclass(frozen=True)
class Row:
    model: str
    channel: str
    variant: str
    items: int
    errors: int
    accuracy: float | None


@dataclass(frozen=True)
class Checked:
    """형식을 확인하고 필요한 값만 뽑은 리포트. 리포트 파일은 사람이 고칠 수 있어 그대로 믿지 않는다."""

    created: datetime
    profile_name: str
    profile_sha256: str
    tool_version: str
    benchmark_version: str
    benchmark_sha256: str
    kinds: dict[str, str]
    rows: tuple[Row, ...]
    skipped: tuple[str, ...]


def _clean(value: object) -> str:
    """화면에 그대로 찍을 문자열에서 제어 문자를 뺀다."""
    if not isinstance(value, str):
        raise ValueError("문자열이어야 하는 값이 문자열이 아니다.")
    return "".join(char if char.isprintable() else "?" for char in value)[:120]


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
        raise ValueError("문항 수나 오류 수가 올바른 정수가 아니다.")
    return value


def _rate(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("정답률이 숫자가 아니다.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("정답률이 범위를 벗어났다.") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("정답률이 0과 1 사이가 아니다.")
    return number


def check_report(report: object, now: datetime | None = None) -> Checked:
    """리포트의 형식을 확인한다. 판정에 쓸 수 없으면 ValueError를 낸다."""
    try:
        return _check(report, now)
    except ValueError:
        raise
    except Exception as error:  # 손으로 고친 파일은 어떤 모양이든 올 수 있다.
        raise ValueError(f"리포트의 형식이 맞지 않는다({type(error).__name__}).") from error


def _check(report: Any, now: datetime | None) -> Checked:
    if not isinstance(report, dict):
        raise ValueError("리포트가 JSON 객체가 아니다.")
    created = datetime.fromisoformat(_clean(report["created_at"]))
    if created.tzinfo is None:
        created = created.astimezone()
    if now is not None and created - now > MAX_CLOCK_SKEW:
        raise ValueError("리포트의 실행 날짜가 미래다. 검증한 PC의 시계를 확인한다.")
    kinds: dict[str, str] = {}
    for model in report["models"]:
        name = _clean(model["name"])
        if name in kinds:
            raise ValueError("같은 이름의 모델이 두 번 적혀 있다.")
        kinds[name] = _clean(model["kind"])
    rows = []
    for entry in report["results"]:
        row = Row(
            model=_clean(entry["model"]),
            channel=_clean(entry["channel"]),
            variant=_clean(entry["variant"]),
            items=_count(entry["items"]),
            errors=_count(entry["errors"]),
            accuracy=_rate(entry.get("accuracy")),
        )
        if row.variant not in VARIANTS or row.errors > row.items or row.model not in kinds:
            raise ValueError("결과 줄의 값이 맞지 않는다.")
        rows.append(row)
    skipped = report.get("skipped") or []
    return Checked(
        created=created,
        profile_name=_clean(report["profile"]["name"]),
        profile_sha256=_clean(report["profile"]["sha256"]),
        tool_version=_clean(report["tool_version"]),
        benchmark_version=_clean(report["benchmark"]["version"]),
        benchmark_sha256=_clean(report["benchmark"]["sha256"]),
        kinds=kinds,
        rows=tuple(rows),
        skipped=tuple(_clean(entry["model"]) for entry in skipped),
    )


def resists_reading(profile: Profile) -> bool:
    return any(technique.id not in TRACKING_ONLY for technique in profile.techniques)


CHANNEL_LABELS = {"text": "텍스트 복사", "image": "화면 캡처·촬영", "file": "파일 올리기"}


def channel_label(channel: str) -> str:
    return CHANNEL_LABELS.get(channel, channel)


def _verdict(name: str, kind: str, channel: str, rows: list[Row]) -> ModelVerdict:
    """한 모델이 한 경로로 푼 결과의 판정. 대조군과 변환본의 정답률을 비교한다.

    일부만 잰 결과로 '막음'이라고 하지 않는다. '못 막음'은 일부만 재도 알 수 있다.
    """
    scores: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    partial = False
    for row in rows:
        if row.accuracy is None or row.errors > 0:
            partial = True
        if row.accuracy is not None:
            scores[row.variant].append(row.accuracy)
    control = max(scores["control"], default=None)
    protected = max(scores["protected"], default=None)
    seen = {row.variant for row in rows}
    if control and protected is not None and protected > control * HOLD_RATIO:
        return ModelVerdict(name, kind, channel, control, protected, False)
    if partial or seen != set(VARIANTS) or protected is None:
        reason = "오류가 났거나 빠진 호출이 있어 일부만 쟀다"
        return ModelVerdict(name, kind, channel, control, protected, None, reason)
    if not control:
        # 대조군도 못 푼 모델로는 변환의 효과를 알 수 없다.
        return ModelVerdict(
            name, kind, channel, control, protected, None, "대조군 문항을 풀지 못했다"
        )
    return ModelVerdict(name, kind, channel, control, protected, True)


def model_verdicts(report: dict[str, Any] | Checked) -> tuple[ModelVerdict, ...]:
    """AI 모델마다, 공격 경로마다 판정 하나. 부정행위자는 이 가운데 가장 잘 풀리는 경로를 쓴다."""
    checked = report if isinstance(report, Checked) else check_report(report)
    grouped: dict[tuple[str, str], list[Row]] = {}
    for row in checked.rows:
        if checked.kinds[row.model] in AI_KINDS:
            grouped.setdefault((row.model, row.channel), []).append(row)
    order = {channel: index for index, channel in enumerate(CHANNEL_LABELS)}
    keys = sorted(grouped, key=lambda key: (key[0], order.get(key[1], 99), key[1]))
    return tuple(
        _verdict(model, checked.kinds[model], channel, grouped[(model, channel)])
        for model, channel in keys
    )


def measured_by_ai(checked: Checked) -> bool:
    """실제 AI 모델이 한 번이라도 답한 리포트인지 본다. 기준선과 모의 모델은 세지 않는다."""
    return any(
        checked.kinds[row.model] in AI_KINDS and row.errors < row.items for row in checked.rows
    )


def record_refusal(report: dict[str, Any], now: datetime | None = None) -> str | None:
    """이 리포트를 프로파일의 검증 기록으로 남길 수 없으면 그 이유를 돌려준다."""
    try:
        checked = check_report(report, now)
    except ValueError as error:
        return str(error)
    if isinstance(report, dict) and report.get("channels_limited"):
        return "경로를 골라서 잰 결과다. 재지 않은 경로가 있다."
    if checked.benchmark_version == "custom":
        return "내장 문항 세트로 잰 결과가 아니다."
    if not measured_by_ai(checked):
        return "실제 AI 모델이 답한 결과가 없다. 기준선과 모의 모델만으로는 기록하지 않는다."
    return None


def write_record(report: dict[str, Any], record_dir: Path) -> Path:
    name = str(report["profile"]["name"])
    if name not in list_profile_names():
        raise ExamShieldError(f"내장 프로파일이 아니라 검증 기록을 남기지 않는다: {name!r}")
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _read(path: Path, now: datetime) -> tuple[dict[str, Any], Checked] | None:
    """검증 기록으로 쓸 수 있는 리포트만 돌려준다. 깨진 파일은 조용히 건너뛴다."""
    try:
        status = path.stat()
        # 파이프나 장치 파일을 열면 멈출 수 있다.
        if not stat.S_ISREG(status.st_mode) or status.st_size > MAX_REPORT_BYTES:
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        if record_refusal(report, now) is not None:
            return None
        return report, check_report(report, now)
    except Exception:  # 리포트 폴더의 어떤 파일도 protect를 멈추면 안 된다.
        return None


def find_latest(
    profile: Profile, folders: list[Path], now: datetime
) -> tuple[dict[str, Any] | None, Path | None, bool]:
    """프로파일의 가장 최근 검증 리포트를 찾는다.

    세 번째 값은 이름은 같지만 프로파일 내용이 달랐던 때의 리포트만 있는지를 뜻한다.
    """
    newest: tuple[datetime, dict[str, Any], Path] | None = None
    changed = False
    for folder in folders:
        try:
            paths = sorted(folder.glob("*.json")) if folder.is_dir() else []
        except OSError:
            continue
        for path in paths:
            loaded = _read(path, now)
            if loaded is None or loaded[1].profile_name != profile.name:
                continue
            report, checked = loaded
            if checked.profile_sha256 != profile.sha256:
                changed = True
                continue
            if newest is None or checked.created > newest[0]:
                newest = (checked.created, report, path)
    if newest is None:
        return None, None, changed
    return newest[1], newest[2], False


def assess(
    profile: Profile,
    report: dict[str, Any] | None,
    now: datetime,
    current: Current,
    *,
    changed: bool = False,
    source: Path | None = None,
) -> Assessment:
    """리포트의 형식이 맞지 않으면 ValueError를 낸다."""
    name = profile.name
    warnings: list[str] = []
    if profile.status == "deprecated":
        warnings.append(
            f"프로파일 '{name}'은 deprecated다. 막겠다고 한 경로를 검증에서 막지 못해 "
            "더는 권하지 않는다."
        )
    elif profile.status == "experimental":
        warnings.append(f"프로파일 '{name}'은 experimental이다. 효과가 확인되지 않았다.")
    if not resists_reading(profile):
        if profile.techniques:
            warnings.append(
                f"프로파일 '{name}'은 유출 추적용이다. AI가 문서를 읽고 푸는 것은 막지 않는다."
            )
        return Assessment(NOT_APPLICABLE, None, None, (), tuple(warnings))
    rerun = f"`examshield verify --profile {name} --models all`"
    if report is None:
        if changed:
            warnings.append(
                f"프로파일 '{name}'의 내용이 마지막 검증 뒤에 바뀌었다. {rerun}로 다시 잰다."
            )
        else:
            warnings.append(
                f"프로파일 '{name}'을 AI 모델로 검증한 기록이 없다. {rerun}로 재고 쓴다."
            )
        return Assessment(UNVERIFIED, None, None, (), tuple(warnings))

    checked = check_report(report, now)
    created = checked.created
    age_days = max(0, (now - created).days)
    verdicts = model_verdicts(checked)
    # 프로파일이 막겠다고 한 경로. 적혀 있지 않으면 잰 경로를 전부 막아야 한다.
    claimed = set(profile.blocks) or {verdict.channel for verdict in verdicts}
    broken = [v for v in verdicts if v.held is False and v.channel in claimed]
    left_open = [v for v in verdicts if v.held is False and v.channel not in claimed]
    unknown = [verdict for verdict in verdicts if verdict.held is None]

    def scores(found: list[ModelVerdict]) -> str:
        return ", ".join(
            f"{channel_label(v.channel)}({v.model}, 정답률 {v.control:.0%} → {v.protected:.0%})"
            for v in found
        )

    if broken:
        state = FAILED
        warnings.append(
            f"프로파일 '{name}'이 기준에 못 미쳤다. 막겠다고 한 경로에서 변환본의 정답률이 "
            f"대조군의 {HOLD_RATIO:.0%} 이하로 떨어지지 않았다: {scores(broken)}"
        )
        if profile.status != "deprecated":
            warnings.append("이 프로파일은 deprecated로 표시해야 한다.")
    elif unknown or not verdicts:
        state = UNVERIFIED
    elif left_open:
        state = PARTIAL
    else:
        state = PASSED
    if left_open:
        warnings.append(
            f"프로파일 '{name}'은 다음 경로를 막지 못한다: {scores(left_open)}. "
            "그 경로로 문서를 읽는 AI에는 효과가 없다."
        )
    if unknown or not verdicts:
        names = ", ".join(
            f"{verdict.model} {channel_label(verdict.channel)}({verdict.reason})"
            for verdict in unknown
        )
        warnings.append(
            "마지막 검증에서 판정하지 못한 AI 모델이 있다"
            + (f": {names}. " if names else ". ")
            + f"{rerun}로 다시 잰다."
        )
    if checked.skipped:
        warnings.append(
            "마지막 검증에서 건너뛴 모델이 있다: "
            + ", ".join(checked.skipped)
            + ". 그 모델은 판정에 들어가지 않았다."
        )
    if age_days > MAX_AGE_DAYS:
        warnings.append(
            f"마지막 검증이 {age_days}일 전이다({created.date().isoformat()}). "
            f"{MAX_AGE_DAYS}일이 넘었으니 {rerun}로 다시 잰다."
        )
    if checked.tool_version != current.tool_version:
        warnings.append(
            f"마지막 검증은 examshield {checked.tool_version}로 했다. "
            f"지금은 {current.tool_version}이다."
        )
    if checked.benchmark_sha256 != current.benchmark_sha256:
        warnings.append(
            f"마지막 검증에 쓴 문항 세트({checked.benchmark_version})가 "
            f"지금 것({current.benchmark_version})과 다르다."
        )
    if verdicts and not any(verdict.kind == "cloud" for verdict in verdicts):
        warnings.append(
            "마지막 검증에 클라우드 모델이 없다. 로컬 모델을 막았다고 클라우드 모델도 막는 것은 아니다."
        )
    return Assessment(state, created, age_days, verdicts, tuple(warnings), source)


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def format_assessment(assessment: Assessment) -> str:
    lines = [f"검증 상태    {assessment.label}"]
    if assessment.source is not None:
        lines.append(f"검증 리포트  {assessment.source}")
    marks = {True: "막음", False: "못 막음", None: "판정 못 함"}
    for verdict in assessment.models:
        lines.append(
            f"  - {verdict.model} [{verdict.kind}] {channel_label(verdict.channel)}: 정답률 대조군 "
            f"{_percent(verdict.control)} → 변환본 {_percent(verdict.protected)}  "
            f"{marks[verdict.held]}"
        )
    lines += [f"경고: {warning}" for warning in assessment.warnings]
    return "\n".join(lines)
