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
from .image_jobs import editor_state, page_image_path, rebuild_edited, retranslate_region, update_region
from .pipeline import JobManager
from .ingest import PDF_CONVERSION_REMOVED
from .safety import MAX_UPLOAD, RightsError, SafetyError

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
                     autostart: bool = Form(False), mode: str = Form("auto"), dpi: int = Form(300),
                     page_parallel: int = Form(2), rtl: bool = Form(True), rights_ack: bool = Form(False),
                     ocr_backend: str = Form("auto")):
    if engine not in DEFAULT_MODELS:
        raise HTTPException(400, "unknown engine")
    if mode == "pdf":
        raise HTTPException(400, PDF_CONVERSION_REMOVED)
    if mode not in ("auto", "epub", "cbz", "comic"):
        raise HTTPException(400, "unknown mode")
    if ocr_backend not in ("auto", "tesseract", "rapidocr", "easyocr", "fixture"):
        raise HTTPException(400, "unknown ocr backend")
    name = re.sub(r"[^\w.\-가-힣ぁ-んァ-ン一-龥 ()]", "_", os.path.basename(file.filename or "book.epub"))
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".epub", ".pdf", ".cbz", ".zip"):
        ext = ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, "파일이 너무 큽니다")
                tmp.write(chunk)
        job = jm().create_job(tmp_path, dict(
            engine=engine, model=model, api_key=api_key, base_url=base_url,
            temperature=temperature, concurrency=max(1, min(32, concurrency)),
            chunk_size=max(300, min(20000, chunk_size)), src_lang=src_lang,
            tgt_lang=tgt_lang, drop_furigana=drop_furigana, horizontal=horizontal,
            mode=mode, dpi=dpi, page_parallel=max(1, min(4, page_parallel)), rtl=rtl,
            rights_ack=rights_ack, ocr_backend=ocr_backend,
        ), file_name=name)
    except RightsError as e:
        raise HTTPException(422, str(e))
    except DRMError as e:
        raise HTTPException(422, str(e))
    except (SafetyError, EpubError, ValueError) as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"파일 분석 실패: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
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
async def job_start(job_id: str, api_key: str = Form("")):
    _job(job_id)
    if api_key:
        jm().secrets[job_id] = api_key
    try:
        jm().start(job_id)
    except SafetyError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/pause")
def job_pause(job_id: str):
    _job(job_id)
    jm().pause(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
async def job_resume(job_id: str):
    _job(job_id)
    try:
        jm().resume(job_id)
    except SafetyError as e:
        raise HTTPException(400, str(e))
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


@app.get("/api/jobs/{job_id}/editor")
def job_editor(job_id: str):
    _job(job_id)
    return editor_state(jm(), job_id)


@app.get("/api/jobs/{job_id}/pages/{page_id}/image")
def page_image(job_id: str, page_id: str, variant: str = "original"):
    _job(job_id)
    try:
        path = page_image_path(jm(), job_id, page_id, variant)
    except SafetyError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "이미지 없음")
    media = "image/png"
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        media = "image/jpeg"
    elif path.endswith(".webp"):
        media = "image/webp"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/regions/{region_id}")
def region_update(job_id: str, region_id: str, ocr_text: str | None = Form(None), translation: str | None = Form(None),
                  font: str | None = Form(None), font_size: float | None = Form(None),
                  x: int | None = Form(None), y: int | None = Form(None), w: int | None = Form(None), h: int | None = Form(None)):
    _job(job_id)
    fields = {"ocr_text": ocr_text, "translation": translation, "font": font, "font_size": font_size,
              "x": x, "y": y, "w": w, "h": h}
    try:
        row = update_region(jm(), job_id, region_id, fields)
    except SafetyError as e:
        raise HTTPException(400, str(e))
    except KeyError:
        raise HTTPException(404, "영역 없음")
    return {"region_id": row["region_id"], "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"],
            "translation": row["translation"], "ocr_text": row["ocr_text"], "font": row["font"], "font_size": row["font_size"]}


@app.post("/api/jobs/{job_id}/regions/{region_id}/retranslate")
async def region_retranslate(job_id: str, region_id: str):
    _job(job_id)
    try:
        row = await retranslate_region(jm(), job_id, region_id)
    except KeyError:
        raise HTTPException(404, "영역 없음")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, str(e)[:300])
    return {"region_id": region_id, "translation": row.get("translation"), "status": row.get("status")}


@app.post("/api/jobs/{job_id}/rebuild")
async def job_rebuild(job_id: str):
    _job(job_id)
    try:
        await rebuild_edited(jm(), job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True, "status": jm().status(job_id)}
