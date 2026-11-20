## 6. 명령어 설계

```
examshield protect <입력> --out <출력폴더> --profile <이름> [--copies 수험번호.csv] [--seed-file 시드파일]
examshield check   <원본> <복사본>
examshield verify  --profile <이름> --models local|cloud|all [--yes] [--allow-custom <파일>]
examshield auth    add|list|remove <제공사>
examshield profile list|show <이름>
examshield report  <리포트파일>
```

### 6.1 protect

- 입력: PDF 파일 또는 폴더. Word(.docx)는 LibreOffice headless로 PDF로 바꾼 뒤 처리한다.
- 출력: PDF 복사본, 작업 기록(JSON).
- 결정적 동작: 같은 입력, 프로파일, 시드면 바이트 단위로 같은 결과.
- `--copies`: CSV의 수험번호마다 고유 워터마크가 든 복사본을 만든다.

### 6.2 check

"내용 100% 그대로"의 기계적 증명.

- 비교 항목: 쪽수, 쪽별 텍스트(정규화 후), 그림 개수, 쪽 크기.
- 시각 비교: 원본과 복사본을 렌더링해 사람이 읽는 영역의 차이를 수치로 낸다.
- 차이가 있으면 0이 아닌 종료 코드. `protect` 끝에 자동으로 한 번 실행한다.
- 주의: 텍스트 레이어를 일부러 교란하는 프로파일은 추출 텍스트가 다르게 나온다. 이 경우 변환 전 원본 텍스트를 기준값으로 두고, 렌더링 결과를 OCR해 비교한다.

### 6.3 verify

검증 대상은 문서가 아니라 프로파일이다.

1. 내장 문항 세트(공개 기출, 더미 문항)에 프로파일을 적용한다.
2. 시나리오 B면 촬영 시뮬레이션을 거친다: 인쇄 해상도, 기울기, 원근 왜곡, 조명 불균일, 노이즈, JPEG 압축.
3. 모델에 넣고 측정한다.
4. 리포트를 쓴다.

측정값(모델별):

- 글자 판독률: 추출 텍스트와 원문의 일치율
- 정답률: 객관식 문항 정답 비율. 변환 전 대조군과 나란히 표시
- 미끼 적중률: 미끼에 유도된 답을 낸 비율
- 기준선: Tesseract OCR 판독률

리포트 필수 항목: 실행 날짜, 도구 버전, 프로파일 이름과 해시, 모델 이름과 버전, 문항 세트 버전, 시뮬레이션 조건.

### 6.4 auth

- 설치 온보딩에서 키를 묻지 않는다.
- 우선순위: 환경변수(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` 등 각 사 표준 이름) → OS 키체인.
- `auth add`는 키를 OS 키체인에 저장한다. 평문 설정 파일에 쓰지 않는다.
- `auth list`는 끝 4자리만 보여준다.
- 키 없이 `verify --models cloud`를 실행하면 설정 방법을 안내하고 끝낸다.
- 클라우드 호출 직전에 보낼 파일 수, 대상 회사, 예상 호출 수를 보여주고 확인받는다. `--yes`로 생략.

