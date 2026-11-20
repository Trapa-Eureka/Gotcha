## 10. 폴더 구조 (제안)

```
examshield/
  cli.py              # Typer 진입점
  protect/
    pipeline.py       # 입력 → 변환 → 출력
    techniques/       # 기법별 모듈 (glyph_remap.py, decoy.py, dots.py, watermark.py)
  check/
    compare.py        # 텍스트, 구조, 시각 비교
  verify/
    runner.py         # 문항 세트 × 모델 실행
    capture_sim.py    # 인쇄·촬영 시뮬레이션
    models/           # local.py(Ollama), openai.py, anthropic.py, google.py
    metrics.py
    report.py
  auth/
    keys.py           # 환경변수, keyring
  profiles/           # 프로파일 YAML
  benchmark/          # 공개 문항 세트 (버전 관리)
tests/
  test_no_network.py  # S1 강제
  test_deterministic.py
  test_check.py
```

