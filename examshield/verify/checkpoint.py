"""verify가 받은 응답을 호출마다 파일에 적어 두고, 끊긴 실행을 이어 한다.

클라우드 호출은 돈이 든다. 중간에 끊기거나 일부 호출이 실패했을 때 받은 응답까지 버리지 않으려는 것이다.
응답에는 문항을 옮겨 적은 글이 들어 있으므로 내장 문항 세트로 잴 때만 쓴다.
"""

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from examshield.verify.models.base import (
    ModelClient,
    ModelInfo,
    ModelReply,
    PageInput,
    prompt_of,
    token_counts,
)

# 오래된 응답을 오늘 잰 것처럼 쓰지 않는다. 모델은 예고 없이 바뀐다.
MAX_AGE = timedelta(hours=24)


def checkpoint_path(out_dir: Path, profile: str) -> Path:
    return out_dir / f"verify-{profile}.checkpoint.jsonl"


class Checkpoint:
    def __init__(self, path: Path, now: datetime, resume: bool) -> None:
        self.path = path
        self.now = now
        self.reused = 0
        self._entries: dict[str, dict[str, Any]] = {}
        if resume:
            self._load()
        else:
            path.unlink(missing_ok=True)

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return
        for line in lines:
            try:
                entry = json.loads(line)
                taken = datetime.fromisoformat(str(entry["time"]))
                age = self.now - taken
                if timedelta(0) <= age <= MAX_AGE:
                    self._entries[str(entry["key"])] = entry
            except (ValueError, KeyError, TypeError):
                # 끊기면서 반쯤 쓰인 줄일 수 있다. 그 호출은 다시 한다.
                continue

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        entry = {"key": key, "time": self.now.isoformat(timespec="seconds"), **entry}
        self._entries[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def call_key(info: ModelInfo, page: PageInput, channel: str) -> str:
    """같은 모델에 같은 것을 같은 경로로 보낸 호출이면 같은 값."""
    parts = [
        info.provider,
        info.name,
        channel,
        prompt_of(page),
        str(page.choice_count),
        hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
        hashlib.sha256(page.image_png).hexdigest(),
        hashlib.sha256(page.pdf).hexdigest(),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class Tracked:
    """모델 호출을 감싸서 호출마다 진행을 알리고, 받은 응답을 적어 두거나 적어 둔 것을 돌려준다."""

    def __init__(
        self,
        model: ModelClient,
        checkpoint: Checkpoint | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self._checkpoint = checkpoint
        self._on_call = on_call

    @property
    def info(self) -> ModelInfo:
        return self.model.info

    def run(self, page: PageInput, channel: str) -> ModelReply:
        try:
            if self._checkpoint is None:
                return self.model.run(page, channel)
            key = call_key(self.model.info, page, channel)
            kept = self._checkpoint.get(key)
            if kept is not None:
                self._checkpoint.reused += 1
                seen = getattr(self.model, "saw_version", None)
                if callable(seen):
                    seen(kept.get("version"))
                return ModelReply(
                    transcription=str(kept["transcription"]),
                    answer=kept["answer"],
                    refused=bool(kept["refused"]),
                    answers={int(number): int(given) for number, given in kept["answers"].items()},
                    raw=str(kept["raw"]),
                    tokens=token_counts(*pair) if (pair := kept.get("tokens")) else None,
                    seconds=float(kept.get("seconds") or 0.0),
                )
            # 실패한 호출은 적지 않는다. 이어 할 때 다시 한다.
            began = time.perf_counter()
            reply = self.model.run(page, channel)
            taken = time.perf_counter() - began
            self._checkpoint.put(
                key,
                {
                    "model": self.model.info.name,
                    "channel": channel,
                    "version": self.model.info.version,
                    "transcription": reply.transcription,
                    "answer": reply.answer,
                    "refused": reply.refused,
                    "answers": {str(number): given for number, given in reply.answers.items()},
                    "raw": reply.raw,
                    "tokens": list(reply.tokens) if reply.tokens is not None else None,
                    "seconds": round(taken, 3),
                },
            )
            return reply
        finally:
            if self._on_call is not None:
                self._on_call()
