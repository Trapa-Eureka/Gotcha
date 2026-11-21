#!/bin/bash
# 폐쇄망용 실행 파일을 만든다(macOS).
#
#   packaging/build-macos.sh           둘 다 만든다
#   packaging/build-macos.sh onefile   파일 하나: dist/examshield
#   packaging/build-macos.sh folder    폴더: dist/folder/examshield/ 와 그것을 묶은 dist/examshield-macos-folder.zip
#
# 파일 하나는 옮기기 쉽지만 실행할 때마다 임시 폴더에 풀려서 시작이 느리다.
# 폴더는 풀려 있어서 시작이 빠르고, protect --copies가 여러 프로세스를 띄울 때도 다시 풀지 않는다.
#
# 넣는 것: examshield, 프로파일, 내장 문항 세트, 웹 화면, PDF·이미지 처리 라이브러리.
# 넣지 않는 것:
#   - 클라우드 SDK(openai, anthropic, google-genai). 폐쇄망에서는 쓸 수 없다.
#   - Tesseract, LibreOffice. 따로 설치한 것을 찾아 쓴다. 없어도 PDF 처리는 된다.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mode="${1:-all}"
case "$mode" in
  all | onefile | folder) ;;
  *)
    echo "쓰는 법: packaging/build-macos.sh [all|onefile|folder]" >&2
    exit 2
    ;;
esac

build() {
  # $1: --onefile 또는 --onedir, $2: 결과를 둘 폴더
  uv run --no-sync pyinstaller packaging/entry.py \
    --name examshield \
    "$1" \
    --noconfirm \
    --clean \
    --distpath "$2" \
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
}

if [ "$mode" = all ] || [ "$mode" = onefile ]; then
  build --onefile dist
  echo "만든 파일: dist/examshield ($(du -h dist/examshield | cut -f1))"
  dist/examshield profile list >/dev/null
  echo "실행 확인: dist/examshield profile list 통과"
fi

if [ "$mode" = all ] || [ "$mode" = folder ]; then
  build --onedir dist/folder
  program=dist/folder/examshield/examshield
  echo "만든 폴더: dist/folder/examshield ($(du -sh dist/folder/examshield | cut -f1))"
  "$program" profile list >/dev/null
  echo "실행 확인: $program profile list 통과"
  # 폴더 안에는 링크가 있다. 링크를 그대로 두는 ditto로 묶는다.
  rm -f dist/examshield-macos-folder.zip
  ditto -c -k --keepParent dist/folder/examshield dist/examshield-macos-folder.zip
  echo "묶은 파일: dist/examshield-macos-folder.zip ($(du -h dist/examshield-macos-folder.zip | cut -f1))"
fi
