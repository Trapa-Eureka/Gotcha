#!/bin/bash
command -v uv >/dev/null 2>&1 || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
[ -d .venv ] || exit 0

out=$(uv run --no-sync pytest -q -x -m "not slow" 2>&1)
status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 5 ]; then
  echo "테스트 실패. 원인을 고친 뒤 끝낸다." >&2
  echo "$out" | tail -40 >&2
  exit 2
fi
exit 0
