# EPUB AI Translator — 일본어 전자책 → 한국어 자동 번역기

사용자가 **번역할 권한을 가진** 일본어 EPUB을 업로드하면 LLM(OpenAI / Gemini / Anthropic / OpenAI 호환)으로
전체를 한국어로 번역하고, 원본의 구조·서식(목차, 표지, 삽화, CSS, 링크, 각주, 루비)을 유지한 **새 EPUB**을 만듭니다.

- 원본 파일은 절대 수정하지 않습니다 (작업 폴더에 복사본을 만들어 읽기 전용으로 사용).
- **DRM(암호화) EPUB은 거부합니다.** 우회/제거 기능은 없습니다. (`META-INF/encryption.xml` 의 글꼴 난독화만 허용)

## 빠른 시작

### Windows (더블클릭)
1. Python 3.12 이상 설치 (설치 시 "Add python.exe to PATH" 체크)
2. 폴더의 **`start.bat`** 더블클릭 → 가상환경/패키지 자동 설치 → 브라우저에서 `http://127.0.0.1:8000` 이 열립니다.
3. 화면에서 API 키 입력(또는 `.env`에 저장) → EPUB 업로드 → "분석 + 비용 추정" → "시작".

### 명령줄 (Windows PowerShell / macOS / Linux)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # macOS/Linux: cp .env.example .env  → 키 입력

# 웹 UI
python -m epub_translator serve --port 8000

# CLI 번역 (중단 후 같은 명령을 다시 실행하면 이어서 진행)
python -m epub_translator translate 책.epub --engine gemini -o 책.ko.epub --report-dir 보고서
python -m epub_translator translate 책.epub --engine openai --model gpt-4o-mini --concurrency 4 --chunk-size 2500
python -m epub_translator translate 책.epub --engine anthropic --model claude-sonnet-4-5
python -m epub_translator translate 책.epub --engine openai_compatible --base-url http://localhost:11434/v1 --model qwen2.5:14b
python -m epub_translator translate 책.epub --engine mock -o test.epub     # 오프라인 동작 확인용(가짜 번역)

# 토큰/비용 추정만
python -m epub_translator estimate 책.epub --engine gemini --chunk-size 2500

# 테스트
python -m pytest -v          # 또는 run_tests.bat
```

소설 번역본의 기본 서체는 명조 계열 폴백, 줄간격 1.65, 문단 간격 0, 첫 줄 들여쓰기 1em이다. `!important`와 px 글자 크기는 쓰지 않아서 리더의 글꼴·크기·간격 설정이 덮어쓸 수 있다. 드롭캡과 첫 글자 굵기는 없애고, `em`/`strong`과 제목·장면 구분·삽화는 남긴다. 이미 번역된 파일은 API 없이 `python -m epub_translator restyle 책.epub -o 책.restyle.epub` 로 같은 기본값을 다시 입힌다. 고정 레이아웃 만화에는 적용하지 않는다.

## CBZ 변환 / 만화 번역

PDF를 번역 없이 EPUB으로 바꾸지는 않습니다. PDF는 리더에서 그대로 열 수 있습니다. PDF는 **만화 AI 번역**을 켠 경우에만 받고, 켜기 전에는 작업을 만들거나 번역 API를 호출하지 않습니다. 예전에 만들어 둔 PDF 변환 결과와 DB 기록은 지우지 않으며, 완료된 파일은 받을 수 있습니다. 그 작업의 시작·재개는 거부됩니다.

| 파일 | 모드 | 하는 일 |
| --- | --- | --- |
| `.epub` | 소설 번역 | 기존 파이프라인. DRM EPUB은 거부 |
| `.cbz`/`.zip` | CBZ → EPUB3 고정 레이아웃 | JPG/PNG/WebP, 자연 정렬, 원본 바이트 유지. RTL 기본 |
| PDF 또는 CBZ + 만화 번역 | 만화 | 말풍선 휴리스틱, 일본어 OCR, 기존 LLM으로 한국어 번역, 단색 배경 복원, HTML 식자, 편집기. PDF 페이지가 이미지 한 장이면 그 바이트를 재인코딩하지 않음 |

시작 전에 **적법한 번역·변환 권한**을 확인해야 합니다. CBZ와 만화는 이 확인이 없으면 작업을 만들지 않습니다.

```bash
python -m epub_translator translate 책.cbz --mode cbz --rights-ack --ltr -o 책.fxl.epub
python -m epub_translator translate 책.cbz --mode comic --engine gemini --rights-ack -o 책.ko.epub
python -m epub_translator translate 책.pdf --mode comic --engine gemini --rights-ack -o 책.ko.epub
```

OCR·PDF 렌더의 선택 설치는 `docs/OPTIONAL_DEPS.md`. 소설만 쓸 때는 `requirements.txt`만 설치하면 됩니다.

만화 편집기는 완료된 만화 작업에서 원본/번역을 나란히 보여주고, 대사·OCR·글꼴·위치를 저장한 뒤 EPUB만 다시 만듭니다. 검출은 가로 한 줄 또는 세로 한 열이고, 지우는 것도 그 줄의 획만입니다. 확신이 낮으면 원본을 그대로 두고 검토로 남깁니다. 이웃한 줄은 번역 문맥으로 묶고, 한국어는 그 묶음의 영역에 가로로 넣습니다. 학습된 검출 모델은 넣지 않았습니다.

만화 품질의 기본값은 **기본**입니다. CPU로 줄을 잡고, 선택한 엔진에는 글자만 보냅니다. **고품질**은 같은 엔진·모델·키에 페이지 이미지를 함께 보내 OCR과 번역을 고치고, 식자 뒤 이미지 검수를 한 번 더 물을 수 있습니다. 화면에서 고품질을 고르고, 이미지가 그 제공자에게 간다는 확인을 켜야 합니다. Mock이거나 모델이 이미지를 거절하면 기본 번역으로 남고, 다른 유료 모델로 바꾸지 않습니다. 모델 응답과 검수 의견은 정확하다는 뜻이 아닙니다. 용어집에 없는 이름·말장난은 검토로 남기고, 모델이 제안한 상자는 지우지 않습니다. 대사를 고치면 그 페이지 검수는 오래된 것으로 표시하고 다시 호출하지 않습니다.

```bash
python -m epub_translator translate 책.cbz --mode comic --engine gemini --rights-ack --manga-hq --image-input-ack -o 책.ko.epub
```

주요 옵션: `--model`, `--temperature`(기본 0.3), `--concurrency`(동시 요청, 기본 4), `--chunk-size`(문자, 기본 2500),
`--max-output-tokens`, `--max-retries`, `--keep-furigana`(후리가나 모두 유지), `--keep-vertical`(세로쓰기 유지), `--no-epubcheck`.

## API 키
- 환경변수 또는 `.env`: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`
- 웹 UI의 "API 키 저장" → 로컬 `data/secrets.json`(권한 600)에 저장, 화면에는 마스킹되어 표시.
- 화면의 "API 키 (이번 작업에만)" 칸은 메모리에만 보관되고 DB에 저장되지 않습니다.

## 구조

```
epub_translator/
  epub_model.py   EPUB 컨테이너/OPF/spine/NCX/nav 파싱, DRM 검사, 블록 추출(인라인 태그 → <tN> 토큰), 재조립
  chunker.py      의미 단위 청크 분할 (장면 전환·제목·서술/대화 경계 우선, 대화 연속 구간은 가능한 한 분리 안 함)
  context.py      번역 프롬프트: 이전 청크 요약, 직전 원문/번역 쌍, 용어집, 인물 말투·호칭 DB, 후리가나 읽기 힌트
  engines.py      OpenAI / Gemini / Anthropic / OpenAI 호환 / Mock 엔진, 재시도 가능한 오류 분류, 429 할당량 처리
  validator.py    블록 단위 검증: 누락, 일본어 잔존, 태그 불일치, 비정상 길이(+의미 신호로 교차확인), 대화·숫자·문장 수, 중복
  pipeline.py     작업 관리: SQLite 상태, 캐시, 체크포인트, 일시정지/재개/취소, 선택적 재번역, 최종 생성/검수
  rebuild.py      번역 텍스트만 교체한 새 EPUB 생성, 세로쓰기→가로쓰기 CSS/OPF 변환, 언어 메타데이터 ko
  qa.py           1차 구조검사, EPUBCheck 실행, 2차 번역검사, JSON/HTML 보고서
  server.py       FastAPI REST API + 웹 UI (static/index.html)
  cli.py          명령줄 인터페이스
tests/            pytest (파싱, 토큰 왕복, 검증기, 청크, DRM 거부, 종단간, 장애 주입, 이어하기, 할당량, 웹 API)
scripts/make_sample_epub.py   직접 작성한 일본어 샘플 EPUB 생성기 (EPUB3 세로쓰기 + EPUB2)
samples/          샘플 입력/출력 EPUB, 검수 보고서 예제
tools/epubcheck/  EPUBCheck 5.1.0 (Java 필요)
```

### 처리 흐름
1. **분석**: 원본 SHA-256 해시 + 엔진/모델/언어 설정으로 작업 ID 생성 → 같은 파일·설정이면 기존 작업을 이어서 진행.
2. **블록 추출**: spine 순서의 XHTML을 lxml로 파싱, 하위 블록이 없는 블록 요소(p, h1~6, li, td, div…)마다
   고유 ID(`파일#번호`) 부여. 인라인 요소(em, span, a, ruby, br, img…)는 `<t3>…</t3>`, `<t5/>` 토큰으로 바꿔 LLM에 전달하고,
   돌려받은 뒤 **원래 요소(속성·id·href 포함)를 복제하여 재구성**합니다. HTML 전체를 평문으로 만들었다 다시 조립하지 않습니다.
   `<title>`, NCX 라벨, nav 목차, OPF 제목도 블록으로 처리합니다.
3. **루비**: 한자 위 히라가나 읽기(후리가나)는 한국어에서 불필요하므로 제거하고 읽기는 이름 음역 힌트로 사용.
   가타카나/한자 등 의미가 있는 루비(예: 蜃気楼→ミラージュ)는 `<ruby>`를 유지하고 루비 텍스트도 번역.
4. **번역**: 청크별로 JSON 요청/응답(`{"translations":[{"id","text"}], "summary", "new_terms", "characters"}`).
   장(파일) 단위로 병렬, 장 내부는 순차 → 이전 청크 요약/직전 문단이 다음 청크로 전달됩니다.
5. **검증/복구**: 응답 잘림·JSON 오류 → 청크를 반으로 나눠 재번역. API 오류 → 지수 백오프 재시도(429는 retry-after 준수,
   일일 할당량 초과 시 작업을 멈추고 '재개'로 이어하기). 블록 검증 실패 → 선택적 재번역 라운드(기본 2회).
   짧은 번역은 길이만으로 실패 처리하지 않고 대화부호 소실·숫자 불일치·문장 수 급감 등 신호가 겹칠 때만 오류로 봅니다.
6. **생성**: 원본 ZIP의 모든 항목을 복사하고 번역된 XHTML/NCX/OPF/CSS만 교체. `mimetype`은 첫 항목·무압축.
7. **검수**: 구조검사 → EPUBCheck(설치 시) → 번역검사(미번역/중복/대화/고유명사/말투/길이) → `report.json`, `report.html`.

### 세로쓰기 → 가로쓰기
CSS와 인라인 style의 `writing-mode`/`-epub-writing-mode`/`-webkit-writing-mode: vertical-rl` → `horizontal-tb`,
`text-combine-upright`(종중횡) → `none`, `text-orientation: upright` → `mixed`, OPF `page-progression-direction="rtl"` → `ltr`,
`primary-writing-mode` → `horizontal-lr`, `dc:language`/`xml:lang` → `ko`.

### 진행률
실제로 검증을 통과한(done/warn) 텍스트 블록 수 ÷ 전체 블록 수. 웹 UI는 1초마다 갱신하며 현재 장, 실패/재시도, 토큰, 로그를 표시합니다.

## 알려진 제한
- 장(파일) 사이를 병렬 처리하므로 **다른 장의 요약**은 그 장이 먼저 끝났을 때만 전달됩니다 (용어집/인물 DB는 공유). `--concurrency 1`이면 완전 순차.
- 인물 말투 불일치 검사는 휴리스틱(존댓말/반말 비율)이며 화자 판별은 하지 않습니다. 결과는 참고용.
- 비용은 추정치이며 모델 가격표(`engines.py`의 `PRICES`)는 직접 갱신해야 할 수 있습니다.
- 소설 EPUB 안의 그림 글자는 이 모드에서 번역하지 않습니다. 만화는 CBZ 또는 PDF를 만화 모드로 올리며, 말풍선 검출과 배경 복원은 휴리스틱입니다. 번역 없는 PDF→EPUB 변환은 없습니다.
- WebP는 바이트를 유지하지만 EPUBCheck는 WebP를 핵심 미디어 타입으로 보지 않을 수 있습니다.
- 무료 API 키는 일일 요청 수 제한이 있어 장편은 여러 날에 걸쳐 이어하기가 필요할 수 있습니다 (큰 `--chunk-size`로 요청 수 감소).
