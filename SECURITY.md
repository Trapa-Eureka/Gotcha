# 보안 문제 알리기

이 도구에서 보안 문제를 찾았다면 [GitHub 이슈](https://github.com/Trapa-Eureka/Gotcha/issues)에 남겨 주기 바란다.

이슈는 누구나 볼 수 있다. 그래서 아래는 이슈에 올리지 않는다.

- 실제 시드 파일, 수험번호 매핑표, 정답표, API 키
- 실제 시험 문서나 응시자의 수험번호
- 다른 사람의 시스템에서 실제로 통하는 공격 절차

문제를 재현할 때는 내장 문항(`examshield verify`가 쓰는 더미 문항)이나 직접 만든 문서를 쓴다. 공개된 곳에 적기 어려운 내용은 이슈에 "비공개로 알리고 싶다"고만 남기거나 1982gaya@gmail.com으로 보낸다.

## 보안 문제로 보는 것

- `protect`나 `check`가 네트워크로 무언가를 보내는 경우
- 시드, 매핑표, 정답, 수험번호, 문제의 글이 로그나 작업 기록, 웹 화면의 응답에 남는 경우
- 웹 화면(`examshield ui`)이 이 PC 밖에서 열리거나 토큰 없이 쓰이는 경우, 실행한 폴더 밖의 파일을 읽거나 쓰는 경우
- `check`가 내용이 바뀐 복사본을 통과시키는 경우
- `trace`가 엉뚱한 수험번호를 지목하는 경우

## 보안 문제가 아닌 것 (알려진 한계)

화면 캡처나 파일 올리기로 읽는 AI는 막지 못한다. 무늬는 그림을 심하게 손보면 지워진다. 이런 한계는 [README](README.md)의 "알려진 한계"에 적혀 있다. 거기에 없는 새 우회 방법은 이슈로 알려 주면 문서에 보탠다.

## 지원하는 버전

초기 개발 단계라 `main` 브랜치의 최신 상태만 본다. 혼자 관리하는 저장소여서 답이 늦을 수 있다.

---

**In English:** please report security problems in a [GitHub issue](https://github.com/Trapa-Eureka/Gotcha/issues). Issues are public, so do not post real seed files, mapping tables, answer keys, API keys, real exam documents or examinee numbers. For details that should not be public, say so in the issue or write to 1982gaya@gmail.com. Known limits listed in the README are not security problems.
