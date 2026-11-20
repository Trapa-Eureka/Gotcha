import hashlib
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def derive_seed(input_sha256: str, profile_sha256: str, seed_material: bytes | None) -> bytes:
    digest = hashlib.sha256()
    for part in (input_sha256.encode(), profile_sha256.encode(), seed_material or b""):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


class SeedStream:
    """SHA-256 카운터로 만든 난수열. 파이썬 버전이 바뀌어도 같은 값을 낸다."""

    def __init__(self, seed: bytes, label: str) -> None:
        self._key = hashlib.sha256(seed + b"\x00" + label.encode("utf-8")).digest()
        self._counter = 0

    def _next(self) -> int:
        block = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        return int.from_bytes(block[:8], "big")

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound는 1 이상이어야 한다")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self._next()
            if value < limit:
                return value % bound

    def shuffled(self, items: Sequence[T]) -> list[T]:
        result = list(items)
        for index in range(len(result) - 1, 0, -1):
            other = self.below(index + 1)
            result[index], result[other] = result[other], result[index]
        return result
