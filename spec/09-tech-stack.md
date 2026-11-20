## 9. 기술 스택

언어는 Python 하나.

| 영역                    | 선택                                                   |
| ----------------------- | ------------------------------------------------------ |
| CLI                     | Typer                                                  |
| PDF 읽기·쓰기·렌더링    | PyMuPDF, 보조로 pikepdf                                |
| Word → PDF              | LibreOffice headless                                   |
| 폰트                    | fontTools                                              |
| 이미지, 촬영 시뮬레이션 | OpenCV, Pillow, NumPy                                  |
| OCR 기준선              | Tesseract (kor, eng 데이터)                            |
| 로컬 모델               | Ollama + 오픈 가중치 비전 모델                         |
| 클라우드 모델           | 각 사 공식 Python SDK. 기본값 꺼짐                     |
| 키 저장                 | keyring                                                |
| 동일성 검사             | PyMuPDF 텍스트 추출 + difflib                          |
| 테스트                  | pytest                                                 |
| 배포                    | PyInstaller 단일 실행 파일, 개발자용 `uv tool install` |

착수 전 확인할 것:

- PyMuPDF의 폰트 교체, ToUnicode 수정 가능 범위 (공식 문서)
- Ollama의 최신 비전 모델 목록과 한국어 성능
- PyMuPDF는 AGPL이다. 오픈소스 공개면 문제없고, 비공개 상용이면 유료 라이선스가 필요하다
- LibreOffice를 번들할지, 사용자가 따로 설치하게 할지

대안: npm 배포를 유지하려면 TypeScript(mupdf.js, pdf-lib, sharp, fontkit)로도 가능하다. 촬영 시뮬레이션과 폰트 조작 일부를 직접 구현해야 한다.

