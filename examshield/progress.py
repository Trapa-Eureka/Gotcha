import sys
import time
from collections.abc import Callable
from typing import TextIO

LINE_EVERY = 0.1
MIN_TOTAL = 2


class Progress:
    """오래 걸리는 일의 진행을 표준 오류에 보여 준다. 결과 출력(표준 출력)과 섞이지 않는다.

    터미널에서는 한 줄을 고쳐 쓰고, 파일이나 다른 프로그램으로 넘길 때는 10%마다 한 줄을 쓴다.
    """

    def __init__(
        self,
        total: int,
        label: str,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        already: int = 0,
    ) -> None:
        self.total = total
        self.label = label
        self.done = already
        self._already = already
        self._stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self._started = clock()
        self._last_written = -1.0
        self._last_tenth = (already * 10 // total) if total else 0
        self._live = bool(getattr(self._stream, "isatty", lambda: False)())
        self._dirty = False

    def step(self, count: int = 1) -> None:
        self.done = min(self.total, self.done + count)
        if self.total < MIN_TOTAL:
            return
        now = self._clock()
        if self._live:
            if now - self._last_written >= LINE_EVERY or self.done == self.total:
                self._stream.write("\r" + self._line(now) + "\033[K")
                self._stream.flush()
                self._last_written = now
                self._dirty = True
            return
        tenth = self.done * 10 // self.total
        if tenth > self._last_tenth:
            self._last_tenth = tenth
            self._stream.write(self._line(now) + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._live and self._dirty:
            self._stream.write("\n")
            self._stream.flush()
            self._dirty = False

    def _line(self, now: float) -> str:
        made = self.done - self._already
        text = f"{self.label} {self.done}/{self.total} ({self.done * 100 // self.total}%)"
        if made > 0 and self.done < self.total:
            left = (now - self._started) / made * (self.total - self.done)
            text += f", 남은 시간 약 {_duration(left)}"
        return text


def _duration(seconds: float) -> str:
    whole = max(0, round(seconds))
    if whole < 60:
        return f"{whole}초"
    if whole < 3600:
        return f"{whole // 60}분 {whole % 60}초"
    return f"{whole // 3600}시간 {whole % 3600 // 60}분"
