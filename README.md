# Gotcha (잡았다 요놈) — examshield

로컬에 있는 시험 문서(PDF, Word)를 내용은 그대로 둔 채 AI가 읽고 풀기 어렵게 재가공한 복사본으로 만들고, 그 효과를 날짜와 모델 버전이 찍힌 수치로 증명하는 오프라인 CLI.

> **상태: 초기 개발 (Phase 1 완료).**
> 측정 도구인 `examshield verify`와 `examshield report`가 동작한다. 지금은 변환 없는 대조군 점수만 잴 수 있다. 복사본을 만드는 `protect`는 아직 없다.
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
| 프로젝트 골격: uv, ruff, mypy, pytest, 푸시 전 검사(git pre-push 훅) | 완료 |
| 개발용 안전 훅과 비밀 파일 읽기 차단 ([안전 훅](#안전-훅)) | 완료 |
| 명세 분할(`spec/`)과 원본 동일성 검사 | 완료 |
| 내장 문항 세트 v1 (직접 쓴 더미 12문항, 기출문제 아님) | 완료 |
| `examshield verify` — 내장 문항 세트로 프로파일을 측정하고 리포트를 쓴다 | 완료 (대조군 프로파일 `none`, `none-print`만) |
| 촬영 시뮬레이션: 해상도, 기울기, 원근, 조명, 흐림, 노이즈, JPEG | 완료 (실제 촬영과 비교한 적 없음) |
| 기준선: 텍스트 레이어 추출, Tesseract OCR | 완료 (Tesseract는 영어 데이터로만 실행해 봄) |
| 로컬 모델(Ollama) 연결 | 완료 (가짜 서버로만 테스트, 실제 Ollama로는 실행해 보지 않음) |
| `examshield report` — 리포트 파일을 표로 본다 | 완료 |
| `examshield protect` — 시나리오 A(디지털 배포) 프로파일로 복사본 생성 | 예정 (Phase 2) |
| `examshield check` — 원본과 복사본의 내용 동일성 검사 | 예정 (Phase 2) |
| `examshield auth` — API 키를 OS 키체인에 보관 | 예정 (Phase 3) |
| 클라우드 모델 검증 (본인 키 사용, 기본값 꺼짐) | 예정 (Phase 3) |
| 수험번호별 추적 워터마크, `--copies`, `examshield trace` | 예정 (Phase 4) |
| 모델 드리프트 경고, `examshield profile` | 예정 (Phase 5) |
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

[설치](#설치)를 마친 뒤, 저장소 폴더에서 실행한다.

```bash
uv run examshield verify --profile none --models local        # 디지털 배포 경로의 대조군 점수
uv run examshield verify --profile none-print --models local  # 인쇄·촬영 경로(시뮬레이션)의 대조군 점수
uv run examshield report out/reports/<리포트파일>.json        # 저장된 리포트 다시 보기
```

`--models local`은 텍스트 레이어 추출, Tesseract(설치돼 있으면), Ollama(떠 있으면)를 쓴다. 없는 것은 건너뛰고 리포트에 이유를 적는다.

출력 예 (2026-11-20, Tesseract 영어 데이터만 있고 Ollama는 없는 PC):

```
모델           경로           구분    글자 판독률  정답률  미끼 적중률  오류
text-layer     텍스트 레이어  대조군  100.0%       -       -            0
tesseract-eng  이미지         대조군  40.4%        -       -            0
건너뜀: ollama — Ollama(http://127.0.0.1:11434)에 연결하지 못했다. `ollama serve`가 떠 있는지 확인한다.
```

Tesseract 수치가 낮은 것은 문항이 한국어인데 한국어 데이터가 없어서다. 교란 효과가 아니다.

복사본을 만드는 `protect`와 `check`는 Phase 2에서 생긴다.

## 사용법

명령별 설명은 **[USAGE.md](USAGE.md)**에 있다. 아직 없는 명령의 설계는 [spec/06-commands.md](spec/06-commands.md)에 있다.

### 이 저장소 개발하기

```bash
uv sync
uv run pytest            # 테스트
uv run ruff check .      # 린트
uv run mypy              # 타입 검사
```

푸시 전 검사를 켜려면 저장소를 받은 뒤 한 번 실행한다.

```bash
git config core.hooksPath .githooks
```

그러면 `git push` 직전에 ruff, mypy, pytest가 돌고, 하나라도 실패하면 푸시가 막힌다. 서버에서 도는 CI는 쓰지 않는다. 각자 PC에서 도는 검사라 `git push --no-verify`로 건너뛸 수 있다.

## 알려진 한계

- `verify`는 아직 대조군만 잰다. 교란 기법이 든 프로파일은 `protect`가 생기는 Phase 2부터 측정한다.
- 실제 AI 모델로 잰 점수가 아직 없다. Ollama 연결은 응답 형식을 흉내 낸 가짜 서버로만 테스트했다. 실제 Ollama에서 응답 형식이나 비전 기능 판별이 다르면 고쳐야 한다.
- 개발 PC의 Tesseract에 한국어 데이터가 없어서 한국어 OCR 기준선은 실제 값이 아니다.
- 촬영 시뮬레이션은 실제 인쇄·촬영과 비교해 보정한 적이 없다.
- 문항 세트 v1은 쉬운 상식 문항 12개다. 정답률의 통계적 의미는 작다.
- `--models mock`의 `mock-reader`는 실제 AI가 아니다. 측정 파이프라인을 점검하는 용도다.
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
├── examshield/
│   ├── cli.py         # Typer 진입점: verify, report
│   ├── benchmark/     # 문항 세트 (v1)와 PDF 렌더링
│   ├── profiles/      # 프로파일 YAML: none, none-print
│   ├── verify/        # runner, capture_sim, metrics, report, models/
│   ├── netguard.py    # 오프라인 설정일 때 네트워크 거부
│   └── pdfio.py       # PDF 읽기, 렌더링, 결정적 저장
├── tests/
├── spec/              # 작업 명세서 (원본을 절 단위로 나눔)
├── .claude/
│   ├── settings.json  # 거부 규칙, 훅 등록
│   ├── hooks/         # 차단, 린트, 테스트 훅
│   ├── skills/        # phase-work: Phase 하나를 진행하는 절차
│   └── agents/        # fresh-reviewer: 처음 보는 눈으로 하는 검토
├── .githooks/         # pre-push: 푸시 전에 ruff, mypy, pytest
├── CLAUDE.md          # AI 에이전트용 짧은 안내
├── USAGE.md           # 사용법
└── pyproject.toml
```

`docs/`는 git에서 제외된 폴더로, 유료 플랜 메모만 둔다.

## 로드맵

Phase 하나를 끝낼 때마다 결과를 확인받고 다음으로 넘어간다.

- [x] **Phase 0** — 프로젝트 골격: 명세 분할, 훅과 거부 규칙, 린트·타입 검사·푸시 전 검사, 문서 초안
- [x] **Phase 1** — `verify`: 문항 세트, 촬영 시뮬레이션, 기준선(텍스트 추출, Tesseract), 로컬 모델, 리포트. 완료 기준: 변환 없는 대조군 점수가 나온다
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
| 품질 | pytest, hypothesis, ruff, mypy strict, git pre-push 훅 |

## 기여

초기 단계다. 변경을 보내기 전에 이슈로 먼저 논의해 주기 바란다.

## 라이선스

정하지 않았다. PyMuPDF가 AGPL이라 그와 맞물려 결정해야 한다 ([spec/13-open-decisions.md](spec/13-open-decisions.md)). 정해지기 전까지는 재배포 조건이 없는 상태다.
