# Gotcha (잡았다 요놈) — examshield

로컬에 있는 시험 문서(PDF, Word)를 내용은 그대로 둔 채 AI가 읽고 풀기 어렵게 재가공한 복사본으로 만들고, 그 효과를 날짜와 모델 버전이 찍힌 수치로 증명하는 오프라인 CLI.

> **상태: 착수 단계 (Phase 0 완료).**
> 프로젝트 골격만 있다. `examshield` 명령은 아직 동작하지 않는다.
> 무료 범위(Phase 1~5)를 순서대로 만드는 중이고, 유료 플랜은 계획중이다.

**바로 가기:** [빠른 시작](#빠른-시작) · [**사용법**](USAGE.md) · [설치](#설치) · [로드맵](#로드맵) · [명세](spec/README.md)

## 개요

스마트안경과 생성형 AI를 쓴 시험 부정행위가 국내에서 잇따라 적발됐다. 지금 시행 중인 대책은 모두 기기 단속이고, 시험지 자체를 다루는 대책은 찾지 못했다. 이 도구는 그 빈자리를 겨냥한다. 배경은 [spec/02-background.md](spec/02-background.md)에 있다.

하는 일은 세 가지다.

1. 시험 문서를 내용이 같은 복사본으로 바꾼다. 텍스트 레이어 교란, 숨은 미끼, 수험번호별 추적 워터마크를 넣는다.
2. 원본과 복사본의 내용이 같은지 기계로 검사한다.
3. 복사본이 AI 판독과 풀이에 얼마나 저항하는지 모델별로 측정해 리포트로 남긴다.

**이 도구는 "AI가 못 읽는다"를 약속하지 않는다. "이 날짜, 이 모델들 기준으로 판독률이 몇 %였다"를 약속한다.** 교란 기법의 효과는 전부 가설이고, 측정해서 살아남은 것만 쓴다.

대상은 온라인 시험이나 과제 파일을 배포하는 교사, 출제자, 평가 기관이다.

## 기능

| 기능 | 상태 |
|---|---|
| 프로젝트 골격: uv, ruff, mypy, pytest, CI | 완료 |
| 개발용 안전 훅과 비밀 파일 읽기 차단 ([안전 훅](#안전-훅)) | 완료 |
| 명세 분할(`spec/`)과 원본 동일성 검사 | 완료 |
| `examshield verify` — 내장 문항 세트로 프로파일의 저항 효과 측정, 리포트 | 예정 (Phase 1) |
| 촬영 시뮬레이션: 해상도, 기울기, 원근, 조명, 노이즈, JPEG | 예정 (Phase 1) |
| `examshield protect` — 시나리오 A(디지털 배포) 프로파일로 복사본 생성 | 예정 (Phase 2) |
| `examshield check` — 원본과 복사본의 내용 동일성 검사 | 예정 (Phase 2) |
| `examshield auth` — API 키를 OS 키체인에 보관 | 예정 (Phase 3) |
| 클라우드 모델 검증 (본인 키 사용, 기본값 꺼짐) | 예정 (Phase 3) |
| 수험번호별 추적 워터마크, `--copies`, `examshield trace` | 예정 (Phase 4) |
| 모델 드리프트 경고, `examshield report`, `examshield profile` | 예정 (Phase 5) |
| 시나리오 B(인쇄 시험) 프로파일, 촬영 사진에서 워터마크 복원 | 계획중 (유료) |
| 폐쇄망용 단일 실행 파일, 상용 라이선스, 검증된 프로파일 구독 | 계획중 (유료) |

### 동작 방식

- **변환은 네트워크 없이**: `protect`와 `check`는 네트워크 호출이 0건이다. 테스트에서 소켓 생성을 막고 통과하는지 본다.
- **원본은 그대로**: 원본은 읽기 전용으로 열고, 결과는 항상 다른 폴더에 쓴다.
- **같은 입력이면 같은 출력**: 입력, 프로파일, 시드가 같으면 복사본이 바이트 단위로 같다.
- **검증 대상은 문서가 아니라 프로파일**: `verify`는 내장된 공개 문항 세트에 프로파일을 적용해 측정한다. 실제 시험지는 클라우드로 보내지 않는다.
- **API 키는 사용자 것**: 클라우드 검증은 선택 사항이다. 키는 환경변수나 OS 키체인에서 읽고, 평문 파일에 쓰지 않는다.

보안 요구사항 전체는 [spec/05-security.md](spec/05-security.md)에 있다.

## 요구 사항

- Python **3.11 이상** (개발은 3.12)
- [uv](https://docs.astral.sh/uv/)
- Git
- 선택: [Tesseract](https://github.com/tesseract-ocr/tesseract)와 `kor`, `eng` 데이터 (OCR 기준선), [Ollama](https://ollama.com)와 비전 모델 (로컬 모델 측정), LibreOffice (Word 입력)

## 설치

아직 패키지로 배포하지 않았다. 소스에서 설치한다.

```bash
git clone https://github.com/Trapa-Eureka/Gotcha.git
cd Gotcha
uv sync
```

명령이 구현되면 다음처럼 설치할 수 있게 할 예정이다.

```bash
uv tool install examshield
```

## 빠른 시작

아직 쓸 수 있는 명령이 없다. Phase 2가 끝나면 다음 흐름이 된다.

```bash
examshield protect 시험지.pdf --out out/ --profile <이름>   # 복사본 만들기
examshield check 시험지.pdf out/시험지.pdf                  # 내용이 같은지 검사
examshield verify --profile <이름> --models local           # 저항 효과 측정
```

## 사용법

명령별 설명은 **[USAGE.md](USAGE.md)**에 쓴다. 지금은 명령 설계만 있다: [spec/06-commands.md](spec/06-commands.md).

### 이 저장소 개발하기

```bash
uv sync
uv run pytest            # 테스트 (명세 동일성, 안전 훅)
uv run ruff check .      # 린트
uv run mypy              # 타입 검사
```

## 알려진 한계

- 구현된 명령이 없다.
- 개발 PC에 Tesseract 한국어 데이터, Ollama, LibreOffice가 없어서, 이들이 필요한 기능은 설치 전까지 실제 검증을 못 한다.
- 명세의 0단계(타당성 실험)는 하지 않았다. 실제 AI 모델과 API 키가 필요하다. `verify`가 생기면 그 도구로 돌린다.
- 최신 멀티모달 AI에는 교란 폰트, 색 대비 왜곡, 방해선이 거의 효과가 없다는 연구가 있다. 어떤 기법도 통하지 않으면 방향을 "미끼 + 추적"으로 좁힌다 ([spec/12-risks.md](spec/12-risks.md)).
- 안전 훅은 명령 문자열을 대조하는 안전망이다. 감싸거나 돌려 쓴 명령은 빠져나갈 수 있다.

## 유료 플랜 (계획중)

사용자 PC에서 혼자 돌리는 것은 모두 무료다. 기관이 시험을 운영할 때 필요한 기능(인쇄 시험용 프로파일, 촬영 사진 포렌식, 폐쇄망 설치본, 상용 라이선스, 검증된 프로파일 구독)은 유료 플랜으로 계획중이다. 판매하는 것은 아직 없고, 가격과 세부 사항은 정해지지 않았다.

## 안전 훅

이 저장소에는 Claude Code 훅이 들어 있다(`.claude/settings.json`, `.claude/hooks/`). 이 저장소에서 Claude Code로 개발하는 동안 다음이 적용된다.

- **실행 전 차단** (`block-dangerous.sh`): 넓은 범위의 재귀 삭제, git 강제 푸시와 `reset --hard`, `clean -fd`, 비밀 파일이나 `docs/`·`secrets/` 스테이징, `curl | sh`, 패키지 공개 배포.
- **파일을 고칠 때마다 린트** (`lint-file.sh`): 고친 Python 파일에 ruff를 돌리고, 실패하면 내용을 에이전트에게 돌려보낸다.
- **끝내기 전에 테스트** (`run-tests.sh`): 테스트가 실패하면 작업을 끝내지 못한다.
- **읽기 거부 규칙**: `.env`, `.env.*`, `secrets/`.

훅은 실패 시 종료 코드 2로 끝난다. 그래야 실패 내용이 에이전트에게 전달된다. 훅 자체의 테스트는 `tests/test_hooks.py`에 있다.

## 데이터와 개인정보

- 시드 파일과 수험번호 매핑표는 `secrets/`에 둔다. 이 폴더와 `*.seed`, `.env`는 `.gitignore`로 제외돼 있다. 기관이 직접 보관한다.
- 출력 폴더(`out/`)와 로그도 커밋하지 않는다.
- 로그에는 파일 해시, 프로파일 이름, 시각만 남긴다. 문제 텍스트는 남기지 않는다.

## 폴더 구조

```
Gotcha/
├── examshield/        # CLI 패키지 (지금은 버전 정보만)
├── tests/             # 명세 동일성 검사, 훅 테스트
├── spec/              # 작업 명세서 (원본을 절 단위로 나눔)
├── .claude/
│   ├── settings.json  # 거부 규칙, 훅 등록
│   ├── hooks/         # 차단, 린트, 테스트 훅
│   ├── skills/        # phase-work: Phase 하나를 진행하는 절차
│   └── agents/        # fresh-reviewer: 처음 보는 눈으로 하는 검토
├── .github/workflows/ # CI: ruff, mypy, pytest
├── CLAUDE.md          # AI 에이전트용 짧은 안내
├── USAGE.md           # 사용법
└── pyproject.toml
```

`docs/`는 git에서 제외된 폴더로, 유료 플랜 메모만 둔다.

## 로드맵

Phase 하나를 끝낼 때마다 결과를 확인받고 다음으로 넘어간다.

- [x] **Phase 0** — 프로젝트 골격: 명세 분할, 훅과 거부 규칙, 린트·타입 검사·CI, 문서 초안
- [ ] **Phase 1** — `verify`: 문항 세트, 촬영 시뮬레이션, 기준선(텍스트 추출, Tesseract), 로컬 모델, 리포트. 완료 기준: 변환 없는 대조군 점수가 나온다
- [ ] **Phase 2** — `protect` + `check`, 시나리오 A 프로파일(이미지화, 글리프 매핑 변조, 숨은 미끼). 완료 기준: 결정적 출력, 네트워크 0건 테스트 통과
- [ ] **Phase 3** — `auth`, 클라우드 모델 검증, `--offline`. 완료 기준: 키 없이 실행하면 안내 후 종료, 호출 전 확인 절차 동작
- [ ] **Phase 4** — 추적 워터마크, `--copies`, `trace`. 완료 기준: PDF와 화면 캡처 수준 이미지에서 수험번호 복원
- [ ] **Phase 5** — 모델 드리프트 경고, `report`, `profile`, 문서 마무리. 완료 기준: 전체 테스트 통과
- [ ] **Phase 6** — 시나리오 B 인쇄용 프로파일, 실제 인쇄·촬영 보정, 촬영 사진에서 워터마크 복원 (계획중, 유료)
- [ ] **Phase 7** — 폐쇄망용 단일 실행 파일, 상용 라이선스, 프로파일 구독, 웹 UI, HWP 입력 (계획중, 유료)

명세의 단계별 계획([spec/11-plan.md](spec/11-plan.md))과의 대응: 명세 1~4단계가 Phase 1~4, 명세 5단계가 Phase 6, 명세 6단계가 Phase 7이다. `examshield trace`는 명세의 명령 목록에 없지만 워터마크 복원에 필요해서 추가한다.

### 기술 스택

| 영역 | 선택 |
|---|---|
| 언어, 패키지 관리 | Python 3.12, uv |
| CLI | Typer |
| PDF 읽기·쓰기·렌더링 | PyMuPDF, 보조로 pikepdf |
| 폰트 | fontTools |
| 이미지, 촬영 시뮬레이션 | OpenCV, Pillow, NumPy |
| OCR 기준선 | Tesseract |
| 로컬 모델 | Ollama |
| 클라우드 모델 | 각 사 공식 Python SDK (선택 설치 `examshield[cloud]`) |
| 키 저장 | keyring |
| 품질 | pytest, hypothesis, ruff, mypy strict, GitHub Actions |

## 기여

초기 단계다. 변경을 보내기 전에 이슈로 먼저 논의해 주기 바란다.

## 라이선스

정하지 않았다. PyMuPDF가 AGPL이라 그와 맞물려 결정해야 한다 ([spec/13-open-decisions.md](spec/13-open-decisions.md)). 정해지기 전까지는 재배포 조건이 없는 상태다.
