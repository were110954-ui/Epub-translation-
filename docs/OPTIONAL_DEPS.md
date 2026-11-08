# 선택 의존성

소설 EPUB 번역은 `requirements.txt`만으로 동작합니다. 아래는 PDF 렌더와 만화 OCR에만 필요합니다. API 키는 여기 적지 않습니다.

## PDF 만화 번역에서 페이지를 이미지로 만들 때

번역 없이 PDF를 EPUB으로 바꾸는 기능은 없습니다. 아래는 **만화 AI 번역**으로 올린 PDF에만 쓰입니다. 임베디드 JPEG 한 장이 페이지 전체이면 `pypdf`로 원본 바이트를 꺼냅니다. 글자 레이어가 있거나 이미지가 여러 장이면 렌더가 필요하고, 그때 결과가 새 PNG라는 점이 보고서에 남습니다.

하나만 설치하면 됩니다.

```bash
pip install pymupdf
```

또는 Poppler:

- Windows: poppler 빌드의 `pdftoppm.exe`를 PATH에 넣습니다.
- Debian/Ubuntu: `sudo apt install poppler-utils`

DPI는 300–600만 허용합니다. 결과 픽셀이 내부 상한(긴 변 20000, 4000만 픽셀)을 넘으면 그 페이지는 실패하고 나머지 페이지만 이어갑니다.

## 일본어 OCR

자동 선택은 다음 순서입니다.

1. GPU(CUDA)가 보이고 RapidOCR 또는 EasyOCR이 설치되어 있으면 그 백엔드
2. 아니면 Tesseract `jpn` (CPU)
3. 없으면 RapidOCR, 그다음 EasyOCR
4. 아무것도 없으면 만화 작업은 오류로 멈추고, 가짜 대사를 만들지 않습니다

GPU가 없으면 CPU로 실행합니다. `EPUB_OCR_GPU=cpu`면 GPU가 있어도 CPU만 씁니다.

말풍선·글자 위치는 이 저장소의 CPU 휴리스틱입니다. 학습된 검출기(RT-DETR, YOLO)와 LaMa 같은 인페인트 가중치는 넣지 않습니다. 그 모델은 용량이 크고 라이선스가 따로라서, 참조한 다른 프로그램의 파일도 복사하지 않습니다. Tesseract `jpn`/`jpn_vert`가 CPU 기준선입니다.

### Tesseract (권장, 추가 Python 패키지 없음)

- Windows: https://github.com/UB-Mannheim/tesseract/wiki 설치 후 일본어 언어 데이터(`jpn`, 세로쓰기 `jpn_vert`)를 추가합니다. `tesseract`가 PATH에 있어야 합니다.
- Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert`

### RapidOCR (선택)

```bash
pip install rapidocr-onnxruntime
```

기본 휠은 CPU용 onnxruntime입니다. CUDA 공급자가 있는 빌드를 직접 설치한 경우에만 GPU로 표시됩니다. 이 저장소는 GPU 휠을 강제 설치하지 않습니다.

### EasyOCR (선택, 용량이 큼)

```bash
pip install easyocr
```

첫 실행에 일본어 모델을 받습니다. `torch`가 함께 설치되므로 소설만 번역할 때는 설치하지 마세요. GPU는 `torch.cuda`가 가능할 때만 사용합니다.

## EPUBCheck

Java와 `epubcheck.jar`가 있으면 결과 검사에 사용합니다. `tools/epubcheck/epubcheck.jar` 또는 `EPUBCHECK_JAR`.

## 안전 한도

CBZ/PDF 이미지는 경로 탈출, 암호화 ZIP, 비정상 압축률, 파일당 80MB, 총 해제 4GB, 페이지 5000장, 픽셀 4000만을 넘으면 거부하거나 그 페이지만 실패시킵니다. 원본 파일은 복사만 하고 수정하지 않습니다.
