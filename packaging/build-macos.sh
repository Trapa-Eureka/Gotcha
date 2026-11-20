#!/bin/bash
# 폐쇄망용 단일 실행 파일을 만든다(macOS). 결과: dist/examshield
#
# 넣는 것: examshield, 프로파일, 내장 문항 세트, 웹 화면, PDF·이미지 처리 라이브러리.
# 넣지 않는 것:
#   - 클라우드 SDK(openai, anthropic, google-genai). 폐쇄망에서는 쓸 수 없다.
#   - Tesseract, LibreOffice. 따로 설치한 것을 찾아 쓴다. 없어도 PDF 처리는 된다.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run --no-sync pyinstaller packaging/entry.py \
  --name examshield \
  --onefile \
  --noconfirm \
  --clean \
  --distpath dist \
  --workpath build/pyinstaller \
  --specpath build \
  --collect-data examshield \
  --add-data "$PWD/LICENSE:." \
  --exclude-module openai \
  --exclude-module anthropic \
  --exclude-module google \
  --exclude-module pikepdf \
  --exclude-module hypothesis \
  --exclude-module pytest \
  --exclude-module mypy \
  --exclude-module tkinter

echo "만든 파일: dist/examshield ($(du -h dist/examshield | cut -f1))"
dist/examshield profile list >/dev/null
echo "실행 확인: dist/examshield profile list 통과"
