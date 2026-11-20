#!/bin/bash
cmd=$(jq -r '.tool_input.command // empty')
[ -z "$cmd" ] && exit 0

deny() {
  echo "차단: $1. 우회하지 말고 사람에게 확인받는다." >&2
  exit 2
}

if echo "$cmd" | grep -Eq '(^|[;&|[:space:]])rm[[:space:]]+(-[a-zA-Z]*[rR][a-zA-Z]*[[:space:]]+)+(-[a-zA-Z]+[[:space:]]+)*(/|~|\$HOME|\.\.?)/?([[:space:];&|]|$)'; then
  deny "넓은 범위의 재귀 삭제"
fi
if echo "$cmd" | grep -Eq 'git[[:space:]].*push.*(--force|[[:space:]]-f([[:space:]]|$)|--force-with-lease)'; then
  deny "git 강제 푸시"
fi
if echo "$cmd" | grep -Eq 'git[[:space:]].*(reset[[:space:]]+--hard|clean[[:space:]]+-[a-zA-Z]*[fd])'; then
  deny "되돌릴 수 없는 git 초기화"
fi
if echo "$cmd" | grep -Eq 'git[[:space:]]+(.*[[:space:]])?add[[:space:]]' &&
  echo "$cmd" | grep -Eq '([[:space:]]-f([[:space:]]|$)|--force|[[:space:]/]\.env|secrets/|docs/|\.seed|watermark-map)'; then
  deny "비밀 파일이나 git 제외 폴더(docs/, secrets/) 스테이징"
fi
if echo "$cmd" | grep -Eq '(curl|wget)[^|;]*\|[[:space:]]*(sudo[[:space:]]+)?(ba|z)?sh'; then
  deny "내려받은 스크립트를 바로 실행"
fi
if echo "$cmd" | grep -Eq '(uv|twine)[[:space:]]+publish|twine[[:space:]]+upload'; then
  deny "패키지 공개 배포"
fi
exit 0
