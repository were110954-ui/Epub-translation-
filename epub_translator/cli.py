"""Command line: python -m epub_translator translate book.epub --engine gemini"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import shutil
import sys
import time

from .config import DATA_DIR, load_dotenv, load_secrets
from .epub_model import DRMError, EpubError
from .pipeline import JobManager
from .safety import RightsError, SafetyError


def _create(jm, path, settings):
    try:
        return jm.create_job(path, settings)
    except (RightsError, DRMError, SafetyError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return None


def main(argv=None):
    load_dotenv()
    load_secrets()
    ap = argparse.ArgumentParser(prog="epub_translator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate", help="EPUB 번역")
    t.add_argument("epub")
    t.add_argument("-o", "--output", help="결과 EPUB 경로")
    t.add_argument("--engine", default="mock", choices=["mock", "openai", "gemini", "anthropic", "openai_compatible"])
    t.add_argument("--model", default="")
    t.add_argument("--base-url", default="")
    t.add_argument("--temperature", type=float, default=0.3)
    t.add_argument("--concurrency", type=int, default=4)
    t.add_argument("--chunk-size", type=int, default=2500)
    t.add_argument("--keep-furigana", action="store_true", help="루비(후리가나)를 모두 유지")
    t.add_argument("--keep-vertical", action="store_true", help="세로쓰기 유지")
    t.add_argument("--no-epubcheck", action="store_true")
    t.add_argument("--max-output-tokens", type=int, default=16384)
    t.add_argument("--max-retries", type=int, default=4)
    t.add_argument("--report-dir", help="보고서 복사 위치")
    t.add_argument("--data-dir", default=DATA_DIR)
    t.add_argument("--src-lang", default="ja", help="원문 언어 (ja/en/zh)")
    t.add_argument("--mode", default="auto", choices=["auto", "epub", "pdf", "cbz", "comic"],
                   help="auto는 확장자/매직바이트. comic은 PDF/CBZ 만화 번역")
    t.add_argument("--dpi", type=int, default=300, help="PDF 렌더 DPI (300-600). 임베디드 이미지와 같으면 사용하지 않음")
    t.add_argument("--page-parallel", type=int, default=2, help="페이지 병렬 (1-4)")
    t.add_argument("--ocr-backend", default="auto", choices=["auto", "tesseract", "rapidocr", "easyocr", "fixture"])
    t.add_argument("--ltr", action="store_true", help="좌→우 읽기 (기본은 만화 RTL)")
    t.add_argument("--rights-ack", action="store_true", help="PDF/CBZ/만화: 적법한 번역·변환 권한이 있음")
    e = sub.add_parser("estimate", help="토큰/비용 추정만")
    e.add_argument("epub")
    e.add_argument("--engine", default="gemini")
    e.add_argument("--model", default="")
    e.add_argument("--chunk-size", type=int, default=2500)
    e.add_argument("--data-dir", default=DATA_DIR)
    r = sub.add_parser("restyle", help="번역된 소설 EPUB에 한국어 본문 서체만 다시 적용 (API 없음)")
    r.add_argument("epub")
    r.add_argument("-o", "--output", required=True)
    s = sub.add_parser("serve", help="웹 UI 실행")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "restyle":
        from .rebuild import restyle_epub
        try:
            restyle_epub(a.epub, a.output)
        except (DRMError, EpubError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
        print("restyled:", a.output)
        return 0
    if a.cmd == "serve":
        import uvicorn
        print(f"EPUB AI Translator: http://{a.host}:{a.port}")
        uvicorn.run("epub_translator.server:app", host=a.host, port=a.port)
        return 0
    jm = JobManager(a.data_dir)
    settings = {"engine": a.engine, "model": a.model, "chunk_size": a.chunk_size}
    if a.cmd == "translate":
        settings.update(mode=a.mode, dpi=a.dpi, page_parallel=a.page_parallel, ocr_backend=a.ocr_backend,
                        rtl=not a.ltr, rights_ack=bool(a.rights_ack))
    if a.cmd == "estimate":
        job = _create(jm, a.epub, settings)
        if not job:
            return 2
        print(json.dumps(jm.job_estimate(job["job_id"]), ensure_ascii=False, indent=2))
        return 0
    settings.update(src_lang=a.src_lang, base_url=a.base_url, temperature=a.temperature, concurrency=a.concurrency,
                    drop_furigana=not a.keep_furigana, max_output_tokens=a.max_output_tokens, max_retries=a.max_retries, horizontal=not a.keep_vertical, run_epubcheck=not a.no_epubcheck)
    job = _create(jm, a.epub, settings)
    if not job:
        return 2
    jid = job["job_id"]
    print(f"job {jid}: {job['title']}  blocks={job['total_blocks']}")
    print("estimate:", json.dumps(jm.job_estimate(jid), ensure_ascii=False))

    async def go():
        task = jm.start(jid)
        last = -1
        while not task.done():
            await asyncio.sleep(2)
            st = jm.db.get_job(jid)
            if st["done_blocks"] != last:
                last = st["done_blocks"]
                print(f"[{time.strftime('%H:%M:%S')}] {st['done_blocks']}/{st['total_blocks']} blocks "
                      f"failed={st['failed_blocks']} tokens={st['prompt_tokens']}/{st['completion_tokens']} "
                      f"chapter={st['current_chapter']}", flush=True)
        await task
    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        print("중단됨 — 같은 명령을 다시 실행하면 이어서 번역합니다.")
        return 130
    st = jm.db.get_job(jid)
    print("status:", st["status"], st["error"] or "")
    if a.output and os.path.exists(st["out_path"]):
        shutil.copyfile(st["out_path"], a.output)
        print("output:", a.output)
    d = os.path.dirname(st["out_path"])
    if a.report_dir:
        os.makedirs(a.report_dir, exist_ok=True)
        for f in ("report.json", "report.html"):
            if os.path.exists(os.path.join(d, f)):
                shutil.copyfile(os.path.join(d, f), os.path.join(a.report_dir, f))
    print("report:", os.path.join(d, "report.html"))
    return 0 if st["status"] in ("completed", "completed_with_issues") else 1


if __name__ == "__main__":
    sys.exit(main())
