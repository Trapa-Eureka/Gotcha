#!/bin/bash
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  "$CLAUDE_PROJECT_DIR"/*.py) ;;
  *) exit 0 ;;
esac
[ -f "$f" ] || exit 0
command -v uv >/dev/null 2>&1 || exit 0

cd "$CLAUDE_PROJECT_DIR" || exit 0
uv run --no-sync ruff check "$f" 1>&2 || exit 2
exit 0
