# 사용법

아래 예시는 저장소 폴더에서 `uv run examshield ...`으로 실행한다.

| 하고 싶은 일 | 명령 | 상태 |
|---|---|---|
| 프로파일의 저항 효과를 측정한다 | [`examshield verify`](#프로파일-측정-verify) | 완료 (대조군만) |
| 저장된 리포트를 다시 본다 | [`examshield report`](#리포트-보기-report) | 완료 |
| 시험 문서의 복사본을 만든다 | `examshield protect` | 예정 (Phase 2) |
| 원본과 복사본의 내용이 같은지 검사한다 | `examshield check` | 예정 (Phase 2) |
| 클라우드 모델용 API 키를 등록한다 | `examshield auth` | 예정 (Phase 3) |
| 수험번호별 복사본을 만들고 유출본을 추적한다 | `examshield protect --copies`, `examshield trace` | 예정 (Phase 4) |
| 프로파일 목록을 본다 | `examshield profile` | 예정 (Phase 5) |

아직 없는 명령의 설계는 [spec/06-commands.md](spec/06-commands.md)에 있다.

## 프로파일 측정 (verify)

```bash
uv run examshield verify --profile none --models local
```

검증 대상은 사용자의 문서가 아니라 프로파일이다. 내장 문항 세트를 PDF로 만들어 모델에 넣고, 얼마나 읽히고 풀리는지 잰다. 사용자의 파일은 쓰지 않는다.

| 옵션 | 뜻 |
|---|---|
| `--profile <이름>` | 측정할 프로파일. 지금은 `none`(디지털 배포 대조군), `none-print`(인쇄·촬영 대조군) |
| `--models local` | 기본값. 텍스트 레이어 추출, Tesseract, Ollama. 없는 것은 건너뛴다 |
| `--models mock` | 실제 AI가 아닌 모의 모델. 설치 없이 파이프라인이 도는지 볼 때 쓴다 |
| `--models cloud`, `all` | 아직 없다 (Phase 3). 실행하면 안내하고 종료 코드 2로 끝난다 |
| `--out <폴더>` | 리포트를 쓸 폴더. 기본값 `out/reports` |
| `--local-model <이름>` | 쓸 Ollama 모델. 없으면 환경변수 `EXAMSHIELD_OLLAMA_MODEL`, 그것도 없으면 설치된 첫 모델 |

### 경로 두 가지

- **텍스트 레이어**: PDF에서 추출한 글자를 모델에 준다. 파일을 올리거나 복사해서 붙여넣는 경우다.
- **이미지**: 쪽을 그림으로 만들어 모델에 준다. 비전 모델이 화면이나 사진을 보는 경우다.

시나리오 A 프로파일은 두 경로를 다 재고, 시나리오 B 프로파일은 촬영 시뮬레이션을 거친 이미지만 잰다.

### 측정값

| 값 | 뜻 |
|---|---|
| 글자 판독률 | 모델이 옮겨 적은 글자와 원문의 일치율. 공백과 전각·반각 차이는 무시한다 |
| 정답률 | 객관식 정답 비율. 문제를 풀지 않는 기준선(텍스트 레이어, Tesseract)은 `-` |
| 미끼 적중률 | 미끼가 유도한 답을 낸 비율. 미끼 기법이 생기는 Phase 2부터 채워진다 |
| 오류 | 모델 호출이 실패한 문항 수 |

### 모델별 준비

- **Tesseract**: `brew install tesseract tesseract-lang` (macOS). `kor` 데이터가 없으면 리포트에 그 사실이 적히고 한글 판독률이 낮게 나온다.
- **Ollama**: `ollama serve`를 띄우고 비전 모델을 하나 받는다. 주소는 환경변수 `OLLAMA_HOST`로 바꿀 수 있다.

### 오프라인 설정

환경변수 `EXAMSHIELD_OFFLINE=1`을 주면 이 PC 밖으로 나가는 접속을 모두 거부한다. 같은 PC의 Ollama(127.0.0.1)는 밖으로 나가지 않으므로 허용한다. 다른 기계의 Ollama 주소를 지정했다면 거부된다.

### 리포트 파일

`out/reports/verify-<프로파일>-<날짜T시각>.json`에 쓴다. 실행 날짜, 도구 버전, 프로파일 이름과 해시, 모델 이름과 버전, 문항 세트 버전과 해시, 시뮬레이션 조건이 들어간다. 문항 텍스트는 들어가지 않는다.

## 리포트 보기 (report)

```bash
uv run examshield report out/reports/verify-none-20261120T140750.json
```

저장된 리포트를 `verify`가 끝에 보여준 것과 같은 표로 다시 보여준다. 필수 항목이 없는 파일은 거부하고 종료 코드 2로 끝난다.
