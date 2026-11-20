# examshield

시험 문서 AI 판독 저항 CLI. 무엇을 만드는지는 `spec/`에, 어디까지 만들었는지는 `README.md`의 로드맵에 있다.

## 명령

```bash
uv sync --extra cloud
uv run pytest
uv run ruff check .
uv run mypy
```

## 코드만 봐서는 모르는 것

- 작업은 Phase 하나씩 한다. 한 Phase가 끝나면 결과를 보고하고 멈춘다. 다음 Phase는 사람이 승인한 뒤에 시작한다. 절차는 `phase-work` 스킬에 있다.
- `docs/`는 git에서 제외돼 있다. 유료(계획중) 기능의 메모만 둔다. 공개 문서는 `README.md`, `USAGE.md`, `spec/`에 쓴다.
- `README.md`(한국어)를 고치면 `README.en.md`(영어)도 같이 고친다. 구조가 어긋나면 테스트가 실패한다.
- `spec/`의 번호 붙은 파일은 원본 명세를 나눈 것이라 내용을 고치지 않는다. 테스트가 해시로 검사한다.
- 유료(계획중) 범위인 시나리오 B 프로파일, 촬영 사진 복원, 단일 실행 파일 배포는 구현하지 않는다.
- PyMuPDF는 AGPL이다. 라이선스는 아직 정하지 않았다(`spec/13-open-decisions.md`).
- 시드 파일과 수험번호 매핑표는 `secrets/`에 둔다. 이 폴더는 읽기가 막혀 있다.
- 에이전트를 같은 문제로 두 번 고쳐 줬다면 이 파일에 줄을 보태기 전에 테스트나 린트 규칙으로 만들 수 있는지 본다.
