import hashlib
from pathlib import Path

SPEC_DIR = Path(__file__).resolve().parent.parent / "spec"
ORIGINAL_SHA256 = "21fe8b9905bf09725dc4ca082bd773dddad6f793d48f4e9c4a23caf83c0ffdd2"
ORIGINAL_SIZE = 18384


def test_spec_parts_rejoin_to_the_original_document() -> None:
    parts = sorted(SPEC_DIR.glob("[0-9][0-9]-*.md"))
    joined = b"".join(part.read_bytes() for part in parts)

    assert len(parts) == 15
    assert len(joined) == ORIGINAL_SIZE
    assert hashlib.sha256(joined).hexdigest() == ORIGINAL_SHA256
