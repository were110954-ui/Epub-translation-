@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo [1/3] 가상환경 생성 중...
  %PY% -m venv .venv || (echo Python 3.12 이상을 설치하세요: https://www.python.org/downloads/ & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
echo [2/3] 패키지 설치/확인 중...
python -m pip install -q --upgrade pip >nul
python -m pip install -q -r requirements.txt || (echo 패키지 설치 실패 & pause & exit /b 1)
if not exist .env if exist .env.example copy .env.example .env >nul
echo 선택 기능(PDF 렌더, OCR)은 docs\OPTIONAL_DEPS.md 를 참고하세요. 소설 번역에는 필요 없습니다.
echo [3/3] 서버 시작: http://127.0.0.1:8000
start "" http://127.0.0.1:8000
python -m epub_translator serve --host 127.0.0.1 --port 8000
pause
