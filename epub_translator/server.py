"""FastAPI web server + REST API used by the web UI."""
from __future__ import annotations
import json
import os
import re
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import DATA_DIR, load_dotenv, load_secrets, masked, save_secret
from .engines import DEFAULT_MODELS, PRICES
from .epub_model import DRMError, EpubError
from .pipeline import JobManager

load_dotenv()
load_secrets()
app = FastAPI(title="EPUB AI Translator")
JM: JobManager | None = None
STATIC = os.path.join(os.path.dirname(__file__), "static")


def jm() -> JobManager:
    global JM
    if JM is None:
        JM = JobManager(os.environ.get("EPUB_TRANSLATOR_DATA", DATA_DIR))
    return JM


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()


@app.get("/api/config")
def config():
    return {"keys": masked(), "default_models": DEFAULT_MODELS, "prices": PRICES}


@app.post("/api/keys")
def set_key(name: str = Form(...), value: str = Form("")):
    try:
        save_secret(name, value.strip())
    except ValueError:
        raise HTTPException(400, "unknown key")
    return {"keys": masked()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), engine: str = Form("mock"), model: str = Form(""),
                     api_key: str = Form(""), base_url: str = Form(""), temperature: float = Form(0.3),
                     concurrency: int = Form(4), chunk_size: int = Form(2500), src_lang: str = Form("ja"),
                     tgt_lang: str = Form("ko"), drop_furigana: bool = Form(True), horizontal: bool = Form(True),
                     autostart: bool = Form(False)):
    if engine not in DEFAULT_MODELS:
        raise HTTPException(400, "unknown engine")
    name = re.sub(r"[^\w.\-가-힣ぁ-んァ-ン一-龥 ()]", "_", os.path.basename(file.filename or "book.epub"))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        while chunk := await file.read(1 << 20):
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        job = jm().create_job(tmp_path, dict(engine=engine, model=model, api_key=api_key, base_url=base_url,
                                             temperature=temperature, concurrency=max(1, min(32, concurrency)),
                                             chunk_size=max(300, min(20000, chunk_size)), src_lang=src_lang,
                                             tgt_lang=tgt_lang, drop_furigana=drop_furigana, horizontal=horizontal),
                              file_name=name)
    except DRMError as e:
        raise HTTPException(422, str(e))
    except (EpubError, Exception) as e:
        raise HTTPException(400, f"EPUB 분석 실패: {e}")
    finally:
        os.unlink(tmp_path)
    if autostart:
        jm().start(job["job_id"])
    return {"job_id": job["job_id"], "estimate": jm().job_estimate(job["job_id"]), "status": jm().status(job["job_id"])}


@app.get("/api/jobs")
def list_jobs():
    return jm().db.q("SELECT job_id,file_name,title,status,total_blocks,done_blocks,failed_blocks,updated FROM jobs ORDER BY updated DESC")


def _job(job_id):
    st = jm().status(job_id)
    if not st:
        raise HTTPException(404, "job not found")
    return st


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job(job_id)


@app.post("/api/jobs/{job_id}/start")
def job_start(job_id: str, api_key: str = Form("")):
    _job(job_id)
    if api_key:
        jm().secrets[job_id] = api_key
    jm().start(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/pause")
def job_pause(job_id: str):
    _job(job_id)
    jm().pause(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
def job_resume(job_id: str):
    _job(job_id)
    jm().resume(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    _job(job_id)
    jm().cancel_job(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/estimate")
def job_estimate(job_id: str):
    _job(job_id)
    return jm().job_estimate(job_id)


@app.get("/api/jobs/{job_id}/glossary")
def glossary(job_id: str):
    return {"glossary": jm().db.q("SELECT source,target,type,reading,locked FROM glossary WHERE job_id=? ORDER BY source", (job_id,)),
            "characters": jm().db.q("SELECT name,korean,speech,honorifics FROM characters WHERE job_id=? ORDER BY name", (job_id,))}


@app.post("/api/jobs/{job_id}/glossary")
def glossary_set(job_id: str, source: str = Form(...), target: str = Form(""), type: str = Form("term")):
    if target:
        jm().db.x("INSERT OR REPLACE INTO glossary(job_id,source,target,type,locked) VALUES(?,?,?,?,1)", (job_id, source, target, type))
    else:
        jm().db.x("DELETE FROM glossary WHERE job_id=? AND source=?", (job_id, source))
    return glossary(job_id)


@app.get("/api/jobs/{job_id}/blocks")
def blocks(job_id: str, status: str = "", limit: int = 200):
    if status:
        return jm().db.q("SELECT block_id,chapter,status,flags,error,source,target FROM blocks WHERE job_id=? AND status=? ORDER BY idx LIMIT ?", (job_id, status, limit))
    return jm().db.q("SELECT block_id,chapter,status,flags,error,source,target FROM blocks WHERE job_id=? ORDER BY idx LIMIT ?", (job_id, limit))


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    j = _job(job_id)
    if not os.path.exists(j["out_path"]):
        raise HTTPException(404, "아직 결과 파일이 없습니다")
    return FileResponse(j["out_path"], media_type="application/epub+zip", filename=os.path.basename(j["out_path"]))


@app.get("/api/jobs/{job_id}/report.{fmt}")
def report(job_id: str, fmt: str):
    j = _job(job_id)
    p = os.path.join(os.path.dirname(j["out_path"]), f"report.{fmt}")
    if fmt not in ("json", "html") or not os.path.exists(p):
        raise HTTPException(404, "보고서 없음")
    return FileResponse(p, media_type="application/json" if fmt == "json" else "text/html")
