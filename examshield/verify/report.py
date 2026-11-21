import json
from pathlib import Path
from typing import Any

from examshield.errors import ExamShieldError

REQUIRED_KEYS = ("created_at", "tool_version", "profile", "benchmark", "models", "results")
CHANNEL_LABELS = {"text": "텍스트 레이어", "image": "이미지", "file": "PDF 파일"}
VARIANT_LABELS = {"control": "대조군", "protected": "변환본"}


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(report["created_at"]).replace(":", "").replace("-", "")[:15]
    path = out_dir / f"verify-{report['profile']['name']}-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_replies(replies: list[dict[str, Any]], report_path: Path) -> Path:
    """리포트 옆에 응답 글을 따로 쓴다. 리포트에는 문항 텍스트를 넣지 않기 때문이다."""
    path = report_path.with_name(report_path.stem + ".replies.json")
    content = {"report": report_path.name, "replies": replies}
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def format_replies(replies: list[dict[str, Any]]) -> str:
    """문항마다 모델의 답이 정답, 미끼, 다른 번호, 답 없음 가운데 어디에 드는지 센다."""
    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for entry in replies:
        if not entry["solves"]:
            # 글자만 읽는 기준선은 답을 내지 않는다.
            continue
        bucket = counts.setdefault(
            (entry["model"], entry["channel"], entry["variant"]),
            {"correct": 0, "decoy": 0, "other": 0, "none": 0},
        )
        for question in entry["questions"]:
            given = question["given"]
            if given is None:
                bucket["none"] += 1
            elif given == question["answer"]:
                bucket["correct"] += 1
            elif given == question["decoy"]:
                bucket["decoy"] += 1
            else:
                bucket["other"] += 1
    lines = ["문항별 답 (정답 / 미끼 / 다른 번호 / 답 없음)"]
    for (model, channel, variant), bucket in counts.items():
        lines.append(
            f"  {model}  {channel}  {variant}:  {bucket['correct']} / {bucket['decoy']} / "
            f"{bucket['other']} / {bucket['none']}"
        )
    return "\n".join(lines)


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExamShieldError(f"리포트 파일을 찾지 못했다: {path}")
    try:
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExamShieldError(f"리포트 파일이 JSON이 아니다: {path}") from error
    missing = [key for key in REQUIRED_KEYS if key not in report]
    if missing:
        raise ExamShieldError(f"리포트에 필수 항목이 없다: {', '.join(missing)}")
    return report


def format_report(report: dict[str, Any]) -> str:
    profile = report["profile"]
    benchmark = report["benchmark"]
    lines = [
        f"실행 날짜    {report['created_at']}",
        f"도구 버전    examshield {report['tool_version']}",
        f"프로파일     {profile['name']} (시나리오 {profile['scenario']}, {profile['status']})",
        f"프로파일 해시 {profile['sha256'][:16]}",
        f"문항 세트    {benchmark['version']} ({benchmark['item_count']}문항, {benchmark['sha256'][:16]})",
        f"시뮬레이션   {_simulation(report.get('simulation'))}",
        "",
        "모델",
    ]
    for model in report["models"]:
        lines.append(
            f"  - {model['name']} [{model['kind']}] {model['provider']} {model['version']}"
        )
        if model.get("note"):
            lines.append(f"      {model['note']}")
    lines += ["", _table(report["results"])]
    for entry in report.get("skipped") or []:
        lines.append(f"건너뜀: {entry['model']} — {entry['reason']}")
    for note in report.get("notes") or []:
        lines.append(f"참고: {note}")
    return "\n".join(lines)


def _simulation(conditions: dict[str, Any] | None) -> str:
    if not conditions:
        return "없음 (파일을 직접 넣는 경로)"
    return ", ".join(f"{key}={value}" for key, value in conditions.items())


def _table(results: list[dict[str, Any]]) -> str:
    header = ("모델", "경로", "구분", "글자 판독률", "정답률", "미끼 적중률", "오류")
    rows = [header]
    for row in results:
        rows.append(
            (
                str(row["model"]),
                CHANNEL_LABELS.get(row["channel"], str(row["channel"])),
                VARIANT_LABELS.get(row["variant"], str(row["variant"])),
                _percent(row["legibility"]),
                _percent(row["accuracy"]),
                _percent(row["decoy_hit_rate"]),
                str(row["errors"]),
            )
        )
    widths = [max(display_width(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(
            cell + " " * (widths[i] - display_width(cell)) for i, cell in enumerate(row)
        ).rstrip()
        for row in rows
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 0x2E7F else 1 for char in text)
