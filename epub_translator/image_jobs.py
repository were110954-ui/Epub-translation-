"""PDF, CBZ, and comic jobs on the existing JobManager queue.

Novel EPUB translation does not enter this module. Page images are written to disk one at a
time; parallelism is capped by settings['page_parallel'] (1–4). Checkpoints are per page.
The caller's original file is only copied.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import re
import shutil
import threading
import time

from .cbz_convert import extract_cbz_member, scan_cbz
from .epub_model import DRMError, sha256_file
from .fxl import compare_output, layout_korean_text, write_fxl_epub
from .inpaint import restore_regions
from .ocr import gpu_info, make_backend, needs_review, recognize_page
from .pdf_convert import PdfEncrypted, extract_pdf_page, inspect_pdf
from .qa import run_epubcheck, structure_check, translation_check, write_reports
from .safety import RightsError, SafetyError, clamp_dpi, clamp_parallel, configure_pillow

_ERASE_BLOCK = {
    "low-confidence", "effect-uncertain", "placement-uncertain", "ocr-empty",
    "ocr-failed", "ocr-backend-error", "direction-uncertain",
}

LIMITS = [
    "글자 검출은 가로 한 줄 또는 세로 한 열의 CPU 휴리스틱입니다. 칸이나 말풍선 전체를 글자 영역으로 쓰지 않습니다. 손글씨·후리가나·효과음은 불확실하면 검토입니다.",
    "일본어 제거는 생성형 인페인트가 아닙니다. 확신 있는 줄의 어두운 획만 지우고, 확신이 낮으면 원본 픽셀을 유지합니다.",
    "한국어는 줄 묶음의 문단 영역에 가로로 넣습니다. 영역에 안 들어가면 typeset-overflow로 검토가 되고, 편집기에서 고칩니다.",
    "고품질 모드는 선택한 제공자에게만 페이지 이미지를 보냅니다. 응답은 검수 자료이고 정확성 증명이 아닙니다. 모델이 제안한 영역은 지우지 않습니다. 기본 모드는 이미지를 보내지 않습니다.",
    "OCR은 설치된 Tesseract, RapidOCR, EasyOCR의 품질에 따릅니다. GPU는 그 백엔드가 CUDA를 제공할 때만 씁니다.",
    "원본 파일과 orig/ 아래 추출 이미지는 다시 인코딩하지 않습니다. 만화 모드의 work PNG는 파생 이미지입니다.",
]

FONTS = {"gothic", "serif", "sans"}


class JobCancelled(Exception):
    pass


class Slots:
    def __init__(self):
        self.cur = 0
        self.peak = 0
        self.lock = threading.Lock()

    def enter(self):
        with self.lock:
            self.cur += 1
            self.peak = max(self.peak, self.cur)

    def exit(self):
        with self.lock:
            self.cur = max(0, self.cur - 1)

    def reset(self):
        with self.lock:
            self.cur = 0
            self.peak = 0


SLOTS = Slots()
_PAGE_ID = re.compile(r"p\d{4}")
_REGION_ID = re.compile(r"p\d{4}r\d{2}")


def image_job_key(file_hash: str, s: dict) -> str:
    dpi = str(int(s.get("dpi") or 0)) if s.get("input_kind") == "pdf" else "-"
    parts = [
        s.get("mode") or "", s.get("input_kind") or "", s.get("engine") or "", s.get("model") or "",
        s.get("src_lang") or "", s.get("tgt_lang") or "", dpi,
        "rtl" if s.get("rtl", True) else "ltr", s.get("ocr_backend") or "auto",
    ]
    if s.get("manga_quality") == "hq":
        parts.append("hq")
    sig = "|".join(parts)
    return file_hash[:16] + "-" + hashlib.sha1(sig.encode()).hexdigest()[:8]


def _dirs(jm, job_id):
    root = jm.job_dir(job_id)
    orig = os.path.join(root, "orig")
    work = os.path.join(root, "work")
    os.makedirs(orig, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    return root, orig, work


def _orig_file(orig_dir: str, page_id: str) -> str | None:
    if not os.path.isdir(orig_dir):
        return None
    for name in os.listdir(orig_dir):
        if name.startswith(page_id + "."):
            return os.path.join(orig_dir, name)
    return None


def _page_block_id(page_id: str) -> str:
    return "page:" + page_id


def create_image_job(jm, src_path: str, settings: dict, file_name: str, api_key: str = "") -> dict:
    if settings.get("mode") == "pdf":
        from .ingest import PDF_CONVERSION_REMOVED
        raise SafetyError(PDF_CONVERSION_REMOVED)
    if not settings.get("rights_ack"):
        raise RightsError("이 파일을 번역·변환할 법적 권한이 있는지 확인해야 합니다.")
    configure_pillow()
    kind = settings["input_kind"]
    if kind == "pdf":
        settings["dpi"] = clamp_dpi(settings.get("dpi") or 300)
        try:
            plan = inspect_pdf(src_path)
        except PdfEncrypted as e:
            raise DRMError(str(e)) from e
    elif kind == "cbz":
        plan = scan_cbz(src_path)
    else:
        raise SafetyError("이미지 작업은 PDF 또는 CBZ만 지원합니다.")
    settings["page_parallel"] = clamp_parallel(settings.get("page_parallel") or 2)
    settings["rtl"] = bool(settings.get("rtl", True))
    file_hash = sha256_file(src_path)
    job_id = image_job_key(file_hash, settings)
    if api_key:
        jm.secrets[job_id] = api_key
    existing = jm.db.get_job(job_id)
    if existing:
        stored = json.loads(existing["settings"])
        for key in ("concurrency", "temperature", "max_retries", "page_parallel", "base_url", "run_epubcheck", "max_output_tokens"):
            if key in settings:
                stored[key] = settings[key]
        jm.db.update_job(job_id, settings=json.dumps(stored, ensure_ascii=False))
        jm.db.event(job_id, "info", "동일 파일/설정의 기존 작업을 재사용합니다 (이어하기).")
        return jm.db.get_job(job_id)
    root, _orig, _work = _dirs(jm, job_id)
    ext = ".pdf" if kind == "pdf" else ".cbz"
    src_copy = os.path.join(root, "source" + ext)
    shutil.copyfile(src_path, src_copy)
    stem = os.path.splitext(file_name)[0]
    out_name = stem + (".ko.epub" if settings["mode"] == "comic" else ".fxl.epub")
    now = time.time()
    jm.db.x(
        "INSERT INTO jobs(job_id,file_hash,file_name,src_path,out_path,status,settings,title,created,updated,total_blocks,mode,input_kind)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, file_hash, file_name, src_copy, os.path.join(root, out_name), "ready",
         json.dumps(settings, ensure_ascii=False), stem, now, now, len(plan), settings["mode"], kind),
    )
    page_rows = []
    block_rows = []
    for item in plan:
        page_id = f"p{item['index'] + 1:04d}"
        page_rows.append((
            job_id, page_id, item["index"], item.get("src_name") or "", int(item.get("width") or 0),
            int(item.get("height") or 0), item.get("media_type") or "", "", "", int(item.get("byte_size") or 0),
            0, "planned", 0, None, now,
        ))
        block_rows.append({
            "job_id": job_id, "block_id": _page_block_id(page_id), "idx": item["index"],
            "file": "page:" + page_id, "chapter": f"p.{item['index'] + 1}", "kind": "page",
            "xpath": page_id, "source": f"페이지 {page_id}", "furigana": "{}",
        })
    jm.db.x(
        "INSERT INTO pages(job_id,page_id,idx,src_name,width,height,media_type,method,sha256,byte_size,spread,stage,attempts,error,updated)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        page_rows, many=True,
    )
    jm.db.x(
        "INSERT INTO blocks(job_id,block_id,idx,file,chapter,kind,xpath,source,furigana,status,updated)"
        " VALUES(:job_id,:block_id,:idx,:file,:chapter,:kind,:xpath,:source,:furigana,'pending',0)",
        block_rows, many=True,
    )
    jm.db.event(job_id, "info", f"분석 완료: 모드 {settings['mode']} ({kind}), 페이지 {len(plan)}장. 권리 확인이 기록되었습니다.")
    jm.db.event(job_id, "info", LIMITS[0])
    return jm.db.get_job(job_id)


def materialize_page(jm, job_id: str, page: dict, settings: dict):
    """Write one original page image. Raises on failure; does not mark the page failed itself."""
    _root, orig, _work = _dirs(jm, job_id)
    job = jm.db.get_job(job_id)
    page_id = page["page_id"]
    if settings["input_kind"] == "pdf":
        result = extract_pdf_page(
            job["src_path"], int(page["idx"]), orig, int(settings.get("dpi") or 300), page_id,
            float(page.get("width_pt") or 0), float(page.get("height_pt") or 0),
        )
        digest = hashlib.sha256(open(result["path"], "rb").read()).hexdigest()
        result["sha256"] = digest
    else:
        result = extract_cbz_member(job["src_path"], page["src_name"], orig, page_id)
    now = time.time()
    jm.db.x(
        "UPDATE pages SET width=?,height=?,media_type=?,method=?,sha256=?,byte_size=?,stage='extracted',error=NULL,"
        "attempts=attempts+1,updated=? WHERE job_id=? AND page_id=?",
        (result["width"], result["height"], result["media_type"], result["method"], result["sha256"],
         result["byte_size"], now, job_id, page_id),
    )
    jm.db.x(
        "UPDATE blocks SET status='done', target=?, error=NULL WHERE job_id=? AND block_id=?",
        (f"추출됨 {result['method']}", job_id, _page_block_id(page_id)),
    )
    jm.db.update_job(job_id, current_chapter=f"추출 {page_id}")


def _fail_page(jm, job_id, page, message: str):
    text = str(message)[:500]
    now = time.time()
    jm.db.x(
        "UPDATE pages SET stage='failed', error=?, attempts=attempts+1, updated=? WHERE job_id=? AND page_id=?",
        (text, now, job_id, page["page_id"]),
    )
    jm.db.x(
        "UPDATE blocks SET status='failed', error=? WHERE job_id=? AND block_id=?",
        (text, job_id, _page_block_id(page["page_id"])),
    )
    jm.db.event(job_id, "warn", f"{page['page_id']} 실패: {text[:200]}")


def _can_erase(region, flags: list[str]) -> bool:
    if float(region.get("confidence") or 0) < 0.55:
        return False
    return not any(flag in _ERASE_BLOCK for flag in flags)


def _save_regions(jm, job_id, page, regions: list[dict]):
    page_id = page["page_id"]
    jm.db.x("DELETE FROM regions WHERE job_id=? AND page_id=?", (job_id, page_id))
    jm.db.x("DELETE FROM blocks WHERE job_id=? AND kind='balloon' AND file=?", (job_id, "page:" + page_id))
    if not regions:
        return
    now = time.time()
    region_rows = []
    block_rows = []
    for local, region in enumerate(regions):
        rid = f"{page_id}r{local + 1:02d}"
        flags = list(region.get("flags") or [])
        review = 1 if region.get("review") or needs_review(region.get("confidence"), flags) else 0
        direction = region.get("direction") or "horizontal"
        gid = region.get("group_id") or rid
        region_rows.append((
            job_id, rid, page_id, int(page["idx"]) * 10000 + local,
            int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"]),
            direction, region.get("ocr_text") or "", None, float(region.get("confidence") or 0),
            "gothic", None, "pending", json.dumps(flags, ensure_ascii=False), review, 0, now, gid,
        ))
    grouped: dict[str, list] = {}
    group_order = []
    for row in region_rows:
        gid = row[-1]
        if gid not in grouped:
            group_order.append(gid)
            grouped[gid] = []
        grouped[gid].append(row)
    for gid in group_order:
        texts = [row[9].strip() for row in grouped[gid] if (row[9] or "").strip()]
        if not texts:
            continue
        anchor = grouped[gid][0]
        block_rows.append({
            "job_id": job_id, "block_id": anchor[1], "idx": anchor[3],
            "file": "page:" + page_id, "chapter": f"p.{int(page['idx']) + 1}", "kind": "balloon",
            "xpath": f"group:{gid}:{anchor[4]},{anchor[5]},{anchor[6]},{anchor[7]}",
            "source": "\n".join(texts), "furigana": "{}",
        })
    jm.db.x(
        "INSERT INTO regions(job_id,region_id,page_id,idx,x,y,w,h,direction,ocr_text,translation,confidence,font,font_size,"
        "status,flags,review,edited,updated,group_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        region_rows, many=True,
    )
    if block_rows:
        jm.db.x(
            "INSERT INTO blocks(job_id,block_id,idx,file,chapter,kind,xpath,source,furigana,status,updated)"
            " VALUES(:job_id,:block_id,:idx,:file,:chapter,:kind,:xpath,:source,:furigana,'pending',0)",
            block_rows, many=True,
        )


def _ocr_page(jm, job_id, page, settings, backend):
    _root, orig, _work = _dirs(jm, job_id)
    path = _orig_file(orig, page["page_id"])
    if not path:
        raise SafetyError("추출된 페이지 이미지가 없습니다.")
    regions, info = recognize_page(path, backend, rtl=bool(settings.get("rtl", True)))
    _save_regions(jm, job_id, page, regions)
    now = time.time()
    jm.db.x(
        "UPDATE pages SET stage='ocr_done', error=NULL, attempts=attempts+1, updated=? WHERE job_id=? AND page_id=?",
        (now, job_id, page["page_id"]),
    )
    label = "대사 없음" if not regions else f"OCR {len(regions)}"
    jm.db.x(
        "UPDATE blocks SET status='done', target=?, error=NULL WHERE job_id=? AND block_id=?",
        (label, job_id, _page_block_id(page["page_id"])),
    )
    jm.db.update_job(job_id, current_chapter=f"OCR {page['page_id']}")
    return info


def _make_chunks(jm, job_id, settings):
    from .chunker import make_chunks
    rows = jm.db.q(
        "SELECT * FROM blocks WHERE job_id=? AND kind='balloon' AND (chunk_id IS NULL OR chunk_id='') ORDER BY idx",
        (job_id,),
    )
    if not rows:
        return 0
    chunks = make_chunks(rows, int(settings.get("chunk_size") or 2500))
    crow = []
    for seq, chunk in enumerate(chunks):
        # Keep existing chunk ids stable if we already inserted some; seq offset by current count.
        crow.append((job_id, chunk["chunk_id"], seq, chunk["chapter"], chunk["file"], json.dumps(chunk["block_ids"])))
        jm.db.x(
            "UPDATE blocks SET chunk_id=? WHERE job_id=? AND block_id IN (%s)" % ",".join("?" * len(chunk["block_ids"])),
            (chunk["chunk_id"], job_id, *chunk["block_ids"]),
        )
    base = jm.db.q1("SELECT COUNT(*) n FROM chunks WHERE job_id=?", (job_id,))["n"]
    crow = [(a, b, base + i, c, d, e) for i, (a, b, _seq, c, d, e) in enumerate(crow)]
    jm.db.x("INSERT INTO chunks(job_id,chunk_id,seq,chapter,file,block_ids) VALUES(?,?,?,?,?,?)", crow, many=True)
    return len(crow)


async def _parallel(jm, job_id, settings, pages, worker):
    if not pages:
        return
    sem = asyncio.Semaphore(int(settings.get("page_parallel") or 2))

    async def one(page):
        await jm._gate(job_id)
        async with sem:
            SLOTS.enter()
            try:
                await jm._gate(job_id)
                await asyncio.to_thread(worker, page)
            except JobCancelled:
                raise asyncio.CancelledError()
            finally:
                SLOTS.exit()

    await asyncio.gather(*(one(p) for p in pages))


def _guard(jm, job_id):
    delay = float(os.environ.get("EPUB_PAGE_DELAY") or 0)
    if delay:
        time.sleep(delay)
    if job_id in jm.cancel:
        raise JobCancelled()


async def _extract_all(jm, job_id, settings):
    _root, orig, _work = _dirs(jm, job_id)
    pages = jm.db.q("SELECT * FROM pages WHERE job_id=? ORDER BY idx", (job_id,))
    todo = [p for p in pages if not _orig_file(orig, p["page_id"])]

    def worker(page):
        _guard(jm, job_id)
        try:
            materialize_page(jm, job_id, page, settings)
        except JobCancelled:
            raise
        except Exception as e:
            path = _orig_file(orig, page["page_id"])
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            _fail_page(jm, job_id, page, e)

    await _parallel(jm, job_id, settings, todo, worker)
    jm.db.refresh_counts(job_id)


async def _ocr_all(jm, job_id, settings, backend):
    _root, orig, _work = _dirs(jm, job_id)
    pages = jm.db.q("SELECT * FROM pages WHERE job_id=? ORDER BY idx", (job_id,))
    todo = []
    for page in pages:
        if page["stage"] in ("ocr_done", "done"):
            continue
        if not _orig_file(orig, page["page_id"]):
            continue
        n = jm.db.q1("SELECT COUNT(*) n FROM regions WHERE job_id=? AND page_id=?", (job_id, page["page_id"]))["n"]
        if n:
            continue
        todo.append(page)
    infos = []

    def worker(page):
        _guard(jm, job_id)
        try:
            infos.append(_ocr_page(jm, job_id, page, settings, backend))
        except JobCancelled:
            raise
        except Exception as e:
            _fail_page(jm, job_id, page, e)

    await _parallel(jm, job_id, settings, todo, worker)
    jm.db.refresh_counts(job_id)
    return infos


async def _translate_all(jm, job_id, engine, settings):
    db = jm.db
    chunks = db.q("SELECT * FROM chunks WHERE job_id=? AND status!='done' ORDER BY seq", (job_id,))
    db.event(job_id, "info", f"번역 시작: 남은 청크 {len(chunks)}개 (엔진 {engine.name}/{engine.model})")
    by_file: dict[str, list] = {}
    for chunk in chunks:
        by_file.setdefault(chunk["file"], []).append(chunk)
    sem = asyncio.Semaphore(int(settings.get("concurrency", 4)))

    async def do_file(group):
        for chunk in group:
            await jm._gate(job_id)
            async with sem:
                await jm._gate(job_id)
                await jm.translate_chunk(job_id, engine, settings, chunk)

    await asyncio.gather(*(do_file(group) for group in by_file.values()))
    jm.local_repair(job_id, settings)
    rounds = 0
    retried = 0
    for rnd in range(int(settings.get("retranslate_rounds", 2))):
        await jm._gate(job_id)
        targets = [row for row in jm.blocks_needing_retranslation(job_id, include_warn=(rnd == 0)) if row.get("kind") == "balloon"]
        if not targets:
            break
        rounds += 1
        retried += len(targets)
        db.event(job_id, "info", f"선택적 재번역 라운드 {rnd + 1}: {len(targets)}개 대사")
        groups, cur = [], []
        for row in targets:
            if cur and (len(cur) >= 12 or cur[-1]["file"] != row["file"]):
                groups.append(cur)
                cur = []
            cur.append(row)
        if cur:
            groups.append(cur)

        async def do_group(group):
            async with sem:
                await jm._gate(job_id)
                await jm.translate_blocks(job_id, engine, settings, group, chapter=group[0]["chapter"], retry_mode=True)

        await asyncio.gather(*(do_group(group) for group in groups))
    db.x("UPDATE blocks SET status='failed' WHERE job_id=? AND kind='balloon' AND status IN ('retry','pending')", (job_id,))
    db.refresh_counts(job_id)
    _sync_regions_from_blocks(jm, job_id)
    return {"rounds": rounds, "blocks": retried}


def _sync_regions_from_blocks(jm, job_id):
    blocks = {b["block_id"]: b for b in jm.db.q("SELECT * FROM blocks WHERE job_id=? AND kind='balloon'", (job_id,))}
    for region in jm.db.q("SELECT * FROM regions WHERE job_id=?", (job_id,)):
        block = blocks.get(region["region_id"])
        flags = json.loads(region.get("flags") or "[]")
        status = region.get("status") or "pending"
        translation = region.get("translation")
        if block:
            translation = block.get("target")
            status = block.get("status") or status
            for flag in json.loads(block.get("flags") or "[]"):
                if flag not in flags:
                    flags.append(flag)
        review = 1 if region.get("review") or needs_review(region.get("confidence"), flags) or status == "failed" else 0
        if block and status == "done" and review:
            status = "warn"
            jm.db.x(
                "UPDATE blocks SET status='warn', flags=? WHERE job_id=? AND block_id=?",
                (json.dumps(flags, ensure_ascii=False), job_id, block["block_id"]),
            )
        jm.db.x(
            "UPDATE regions SET translation=?, status=?, flags=?, review=? WHERE job_id=? AND region_id=?",
            (translation, status, json.dumps(flags, ensure_ascii=False), review, job_id, region["region_id"]),
        )


def _typeset_page(jm, job_id, page, settings) -> dict:
    _root, orig, work = _dirs(jm, job_id)
    path = _orig_file(orig, page["page_id"])
    regions = jm.db.q("SELECT * FROM regions WHERE job_id=? AND page_id=? ORDER BY idx", (job_id, page["page_id"]))
    width = int(page["width"] or 0) or 600
    height = int(page["height"] or 0) or 800
    if not path:
        return {"page_id": page["page_id"], "idx": page["idx"], "width": width, "height": height,
                "image_path": None, "placeholder": True, "error": page.get("error") or "페이지를 만들지 못했습니다.",
                "title": str(page["idx"] + 1), "overlays": [], "method": page.get("method") or ""}
    overlays = []
    image_path = path
    media = page.get("media_type") or "image/png"
    method = page.get("method") or "copy"
    if regions:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            boxes = []
            for r in regions:
                flags = json.loads(r.get("flags") or "[]")
                boxes.append({
                    "x": int(r["x"]), "y": int(r["y"]), "w": int(r["w"]), "h": int(r["h"]),
                    "erase": _can_erase(r, flags),
                })
            restored, notes = restore_regions(im, boxes)
        for region, note in zip(regions, notes):
            flags = json.loads(region.get("flags") or "[]")
            flag = "inpaint-" + note["method"]
            if flag not in flags:
                flags.append(flag)
            review = region.get("review") or 0
            if not note.get("confident"):
                review = 1
            jm.db.x(
                "UPDATE regions SET flags=?, review=? WHERE job_id=? AND region_id=?",
                (json.dumps(flags, ensure_ascii=False), review, job_id, region["region_id"]),
            )
            region["flags"] = json.dumps(flags, ensure_ascii=False)
            region["review"] = review
        work_path = os.path.join(work, page["page_id"] + ".png")
        restored.save(work_path, format="PNG")
        restored.close()
        image_path = work_path
        media = "image/png"
        method = "inpaint-png"
        grouped: dict[str, list] = {}
        order = []
        for region in regions:
            gid = region.get("group_id") or region["region_id"]
            if gid not in grouped:
                order.append(gid)
                grouped[gid] = []
            grouped[gid].append(region)
        for gid in order:
            members = grouped[gid]
            text = ""
            anchor = members[0]
            for member in members:
                if (member.get("translation") or "").strip():
                    text = member["translation"].strip()
                    anchor = member
                    break
            if not text:
                continue
            x0 = min(int(m["x"]) for m in members)
            y0 = min(int(m["y"]) for m in members)
            x1 = max(int(m["x"]) + int(m["w"]) for m in members)
            y1 = max(int(m["y"]) + int(m["h"]) for m in members)
            laid = layout_korean_text(text, x1 - x0, y1 - y0, False)
            size = float(anchor["font_size"]) if anchor.get("font_size") else laid["font_size"]
            shown = laid["text"] if not anchor.get("font_size") else text
            if not anchor.get("font_size"):
                jm.db.x("UPDATE regions SET font_size=? WHERE job_id=? AND region_id=?", (size, job_id, anchor["region_id"]))
            flags = json.loads(anchor.get("flags") or "[]")
            if laid["overflow"] or size <= 10:
                if "typeset-overflow" not in flags:
                    flags.append("typeset-overflow")
                jm.db.x(
                    "UPDATE regions SET flags=?, review=1 WHERE job_id=? AND region_id=?",
                    (json.dumps(flags, ensure_ascii=False), job_id, anchor["region_id"]),
                )
            overlays.append({
                "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                "text": shown, "font": anchor.get("font") or "gothic", "font_size": float(size),
                "direction": "horizontal",
            })
    return {"page_id": page["page_id"], "idx": page["idx"], "width": width, "height": height,
            "image_path": image_path, "media_type": media, "method": method, "overlays": overlays,
            "title": str(page["idx"] + 1), "placeholder": False, "error": None}


async def _finalize_image(jm, job_id, settings, retranslation=None):
    db = jm.db
    job = db.get_job(job_id)
    db.update_job(job_id, status="building", current_chapter="EPUB 생성/검수")
    pages = db.q("SELECT * FROM pages WHERE job_id=? ORDER BY idx", (job_id,))
    built: dict[str, dict] = {}
    lock = threading.Lock()

    def worker(page):
        _guard(jm, job_id)
        try:
            item = _typeset_page(jm, job_id, page, settings)
        except JobCancelled:
            raise
        except Exception as e:
            _fail_page(jm, job_id, page, e)
            item = {"page_id": page["page_id"], "idx": page["idx"], "width": int(page["width"] or 600),
                    "height": int(page["height"] or 800), "image_path": None, "placeholder": True,
                    "error": str(e), "title": str(page["idx"] + 1), "overlays": [], "method": page.get("method") or ""}
        with lock:
            built[page["page_id"]] = item

    await _parallel(jm, job_id, settings, pages, worker)
    hq_info = settings.get("_hq") or {}
    if hq_info.get("applied_result"):
        from .engines import make_engine
        from .hq import post_typeset
        engine = make_engine({**settings, "api_key": jm.secrets.get(job_id, "")})
        await post_typeset(jm, job_id, engine, settings, built, hq_info)
        settings["_hq"] = hq_info
    ordered = [built[p["page_id"]] for p in pages if p["page_id"] in built]
    real = [p for p in ordered if p.get("image_path")]
    if not real:
        db.update_job(job_id, status="error", error="모든 페이지를 변환하지 못했습니다. 결과 EPUB을 만들지 않았습니다.", current_chapter="실패")
        db.event(job_id, "error", "모든 페이지 실패")
        _write_failure_report(jm, job_id, settings, ordered, retranslation)
        return
    if settings.get("mode") == "comic":
        lang = settings.get("tgt_lang") or "ko"
    else:
        lang = settings.get("src_lang") or "ja"
    import uuid
    meta = {"title": job["title"] or job["file_name"], "language": lang, "rtl": bool(settings.get("rtl", True)),
            "uid": str(uuid.uuid5(uuid.NAMESPACE_URL, "epub-ai-translator:" + job_id))}
    for item in ordered:
        item["lang"] = lang
    await asyncio.to_thread(write_fxl_epub, job["out_path"], ordered, meta)
    for page in pages:
        item = built.get(page["page_id"])
        if item and item.get("image_path") and page["stage"] != "failed":
            db.x("UPDATE pages SET stage='done', updated=? WHERE job_id=? AND page_id=?", (time.time(), job_id, page["page_id"]))
    quality = await asyncio.to_thread(compare_output, job["out_path"], ordered)
    try:
        quality["source_copy_matches_upload"] = sha256_file(job["src_path"]) == job["file_hash"]
    except OSError:
        quality["source_copy_matches_upload"] = False
    struct = await asyncio.to_thread(structure_check, job["out_path"])
    if settings.get("run_epubcheck", True):
        ec = await asyncio.to_thread(run_epubcheck, job["out_path"])
    else:
        ec = {"available": False, "message": "disabled"}
    regions = db.q("SELECT * FROM regions WHERE job_id=? ORDER BY idx", (job_id,))
    balloons = db.q("SELECT * FROM blocks WHERE job_id=? AND kind='balloon' ORDER BY idx", (job_id,))
    if settings.get("mode") == "comic":
        glossary = db.q("SELECT source,target,type,reading FROM glossary WHERE job_id=? ORDER BY source", (job_id,))
        chars = db.q("SELECT name,korean,speech,honorifics FROM characters WHERE job_id=? ORDER BY name", (job_id,))
        trans = translation_check(balloons, glossary, chars, settings.get("tgt_lang") or "ko")
    else:
        glossary, chars = [], []
        trans = {"blocks": 0, "ok": 0, "warn": 0, "error": 0, "untranslated": 0, "duplicates": 0,
                 "dialogue_source": 0, "dialogue_preserved": 0, "glossary_inconsistencies": [],
                 "speech_inconsistencies": [], "issues": [],
                 "note": "PDF/CBZ 변환 전용입니다. 텍스트 번역은 하지 않았습니다."}
    omission = {
        "detected": len(regions),
        "ocr_empty": [r["region_id"] for r in regions if not (r.get("ocr_text") or "").strip()],
        "translation_missing": [r["region_id"] for r in regions if (r.get("ocr_text") or "").strip() and not (r.get("translation") or "").strip()],
        "review": [r["region_id"] for r in regions if r.get("review")],
    }
    db.refresh_counts(job_id)
    job = db.get_job(job_id)
    failed_pages = [p["page_id"] for p in db.q("SELECT page_id FROM pages WHERE job_id=? AND stage='failed' ORDER BY idx", (job_id,))]
    review_n = len(omission["review"])
    image_info = {
        "mode": settings.get("mode"), "input_kind": settings.get("input_kind"),
        "dpi": settings.get("dpi") if settings.get("input_kind") == "pdf" else None,
        "rtl": bool(settings.get("rtl", True)),
        "page_parallel": settings.get("page_parallel"),
        "ocr_backend": settings.get("_ocr_label") or settings.get("ocr_backend"),
        "ocr_real": bool(settings.get("_ocr_real")),
        "gpu": settings.get("_gpu"),
        "limits": LIMITS,
        "quality": quality,
        "omission": omission,
        "failed_pages": failed_pages,
        "inpaint": "text-mask-or-skip",
        "hq": settings.get("_hq") or {"requested": False, "reason": "basic", "high_quality_verified": False,
                                      "live_call": False, "ai_review_is_proof": False},
    }
    from .engines import price_for
    report = {
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "job": {"job_id": job_id, "title": job["title"], "file_name": job["file_name"], "file_hash": job["file_hash"],
                "engine": settings.get("engine"), "model": settings.get("model"), "mode": settings.get("mode"),
                "total_blocks": job["total_blocks"], "done_blocks": job["done_blocks"], "failed_blocks": job["failed_blocks"],
                "prompt_tokens": job["prompt_tokens"], "completion_tokens": job["completion_tokens"],
                "api_calls": job["api_calls"], "retries": job["retries"]},
        "retranslation": retranslation or {},
        "structure": struct, "epubcheck": ec, "translation": trans,
        "image_pipeline": image_info, "glossary": glossary, "characters": chars,
        "failed_blocks": [{"block_id": r["block_id"], "error": r["error"], "source": (r["source"] or "")[:200]}
                          for r in balloons if r["status"] == "failed"],
    }
    root = jm.job_dir(job_id)
    write_reports(report, os.path.join(root, "report.json"), os.path.join(root, "report.html"))
    issues = (not struct["ok"]) or bool(failed_pages) or (job["failed_blocks"] or 0) > 0 or review_n > 0
    if settings.get("mode") == "comic" and not regions:
        issues = True
    if settings.get("mode") != "comic":
        issues = (not struct["ok"]) or bool(failed_pages)
    status = "completed" if not issues else "completed_with_issues"
    db.update_job(job_id, status=status, report_json=json.dumps(report, ensure_ascii=False)[:5_000_000],
                  report_html=os.path.join(root, "report.html"), current_chapter="완료")
    db.event(job_id, "info", f"완료: 페이지 {len(real)}/{len(pages)} 이미지, 구조 {'통과' if struct['ok'] else '문제'}, "
                             f"검토 {review_n}, EPUBCheck fatal {ec.get('fatal', '-')}")
    return report


def _write_failure_report(jm, job_id, settings, ordered, retranslation):
    root = jm.job_dir(job_id)
    job = jm.db.get_job(job_id)
    report = {
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "job": {"job_id": job_id, "title": job["title"], "file_name": job["file_name"], "mode": settings.get("mode"),
                "engine": settings.get("engine"), "model": settings.get("model"), "total_blocks": job["total_blocks"],
                "done_blocks": job["done_blocks"], "failed_blocks": job["failed_blocks"],
                "prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0, "retries": 0},
        "retranslation": retranslation or {},
        "structure": {"ok": False, "errors": ["결과 EPUB 없음"], "warnings": [], "checked": {}},
        "epubcheck": {"available": False, "message": "결과를 만들지 않음"},
        "translation": {"blocks": 0, "ok": 0, "warn": 0, "error": 0, "untranslated": 0, "duplicates": 0,
                        "dialogue_source": 0, "dialogue_preserved": 0, "glossary_inconsistencies": [],
                        "speech_inconsistencies": [], "issues": []},
        "image_pipeline": {"limits": LIMITS, "failed_pages": [p.get("page_id") for p in ordered], "quality": None},
        "glossary": [], "characters": [],
    }
    write_reports(report, os.path.join(root, "report.json"), os.path.join(root, "report.html"))
    jm.db.update_job(job_id, report_html=os.path.join(root, "report.html"),
                     report_json=json.dumps(report, ensure_ascii=False)[:5_000_000])


async def run_image_job(jm, job_id: str):
    from .engines import EngineError, make_engine
    db = jm.db
    job = db.get_job(job_id)
    settings = json.loads(job["settings"])
    retranslation = {"rounds": 0, "blocks": 0}
    if settings.get("mode") == "pdf":
        from .ingest import PDF_CONVERSION_REMOVED
        db.event(job_id, "warn", PDF_CONVERSION_REMOVED)
        if not str(job.get("status") or "").startswith("completed"):
            db.update_job(job_id, status="cancelled", error=PDF_CONVERSION_REMOVED[:500])
        return
    try:
        if not settings.get("rights_ack"):
            raise RightsError("법적 권한 확인이 없습니다.")
        configure_pillow()
        db.update_job(job_id, status="running", error=None)
        backend = None
        if settings.get("mode") == "comic":
            gpu = gpu_info()
            backend = make_backend(settings.get("ocr_backend") or "auto", gpu)
            settings["_ocr_label"] = backend.label
            settings["_ocr_real"] = bool(backend.real)
            settings["_gpu"] = gpu
            db.event(job_id, "info", f"OCR {backend.label}. {gpu.get('note')}")
            if not backend.real:
                db.event(job_id, "warn", "fixture OCR은 테스트 전용입니다. 실제 글자 인식이 아닙니다.")
        await _extract_all(jm, job_id, settings)
        if job_id in jm.cancel:
            raise asyncio.CancelledError()
        if settings.get("mode") == "comic":
            await _ocr_all(jm, job_id, settings, backend)
            if job_id in jm.cancel:
                raise asyncio.CancelledError()
            _make_chunks(jm, job_id, settings)
            hq_info = {"requested": False, "applied_result": False, "request_accepted": False, "live_call": False,
                       "high_quality_verified": False, "image_use_proven": False, "ai_review_is_proof": False,
                       "reason": "basic"}
            engine = None
            if settings.get("manga_quality") == "hq":
                engine = make_engine({**settings, "api_key": jm.secrets.get(job_id, "")})
                from .hq import run_hq
                hq_info = await run_hq(jm, job_id, engine, settings)
            settings["_hq"] = hq_info
            pending = db.q1("SELECT COUNT(*) n FROM blocks WHERE job_id=? AND kind='balloon' AND status NOT IN ('done','warn')", (job_id,))
            if pending and pending["n"]:
                if engine is None:
                    engine = make_engine({**settings, "api_key": jm.secrets.get(job_id, "")})
                retranslation = await _translate_all(jm, job_id, engine, settings)
            else:
                if not db.q1("SELECT COUNT(*) n FROM regions WHERE job_id=?", (job_id,))["n"]:
                    db.event(job_id, "warn", "번역할 대사가 없습니다. 이미지 EPUB만 만들고 검토가 필요합니다.")
                _sync_regions_from_blocks(jm, job_id)
        await _finalize_image(jm, job_id, settings, retranslation)
    except RightsError as e:
        db.update_job(job_id, status="error", error=str(e)[:500])
        db.event(job_id, "error", str(e)[:300])
    except EngineError as e:
        db.x("UPDATE blocks SET status='pending' WHERE job_id=? AND status='running'", (job_id,))
        db.refresh_counts(job_id)
        db.update_job(job_id, status="error", error=str(e)[:500])
        db.event(job_id, "error", f"중단: {str(e)[:300]}")
    except asyncio.CancelledError:
        db.x("UPDATE blocks SET status='pending' WHERE job_id=? AND status='running'", (job_id,))
        db.refresh_counts(job_id)
        if db.get_job(job_id)["status"] != "error":
            db.update_job(job_id, status="cancelled")
    except Exception as e:
        db.update_job(job_id, status="error", error=repr(e)[:500])
        db.event(job_id, "error", f"작업 오류: {e!r}"[:300])


def editor_state(jm, job_id: str) -> dict:
    job = jm.db.get_job(job_id)
    if not job:
        return {}
    settings = json.loads(job["settings"])
    pages = jm.db.q(
        "SELECT page_id, idx, stage, width, height, method, error FROM pages WHERE job_id=? ORDER BY idx",
        (job_id,),
    )
    regions = jm.db.q(
        "SELECT region_id, page_id, idx, x, y, w, h, direction, ocr_text, translation, confidence, font, font_size, status, flags, review, edited "
        "FROM regions WHERE job_id=? ORDER BY idx",
        (job_id,),
    )
    for region in regions:
        region["flags"] = json.loads(region.get("flags") or "[]")
    hq = {"requested": settings.get("manga_quality") == "hq",
          "note": "고품질을 켜면 선택한 모델에만 페이지 이미지를 보냅니다. 편집하면 그 페이지의 이미지 검수는 오래되어 다시 호출하지 않습니다."}
    return {"mode": settings.get("mode"), "pages": pages, "regions": regions,
            "limits": LIMITS if settings.get("mode") == "comic" else [], "hq": hq}


def page_image_path(jm, job_id: str, page_id: str, variant: str) -> str:
    if not _PAGE_ID.fullmatch(page_id or ""):
        raise SafetyError("페이지 ID가 올바르지 않습니다.")
    if variant not in ("original", "work"):
        raise SafetyError("image variant 는 original 또는 work 입니다.")
    _root, orig, work = _dirs(jm, job_id)
    folder = orig if variant == "original" else work
    path = _orig_file(folder, page_id) if variant == "original" else (
        os.path.join(work, page_id + ".png") if os.path.isfile(os.path.join(work, page_id + ".png")) else None
    )
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(page_id)
    folder_real = os.path.realpath(folder)
    if not os.path.realpath(path).startswith(folder_real + os.sep):
        raise SafetyError("이미지 경로가 작업 폴더 밖입니다.")
    return path


def update_region(jm, job_id: str, region_id: str, fields: dict) -> dict:
    if not _REGION_ID.fullmatch(region_id or ""):
        raise SafetyError("영역 ID가 올바르지 않습니다.")
    region = jm.db.q1("SELECT * FROM regions WHERE job_id=? AND region_id=?", (job_id, region_id))
    if not region:
        raise KeyError(region_id)
    page = jm.db.q1("SELECT * FROM pages WHERE job_id=? AND page_id=?", (job_id, region["page_id"]))
    x = int(fields["x"]) if fields.get("x") is not None else int(region["x"])
    y = int(fields["y"]) if fields.get("y") is not None else int(region["y"])
    w = int(fields["w"]) if fields.get("w") is not None else int(region["w"])
    h = int(fields["h"]) if fields.get("h") is not None else int(region["h"])
    max_w = int(page["width"] or 0) or w
    max_h = int(page["height"] or 0) or h
    x = max(0, min(x, max(0, max_w - 8)))
    y = max(0, min(y, max(0, max_h - 8)))
    w = max(8, min(w, max_w - x))
    h = max(8, min(h, max_h - y))
    font = fields.get("font") or region.get("font") or "gothic"
    if font not in FONTS:
        font = "gothic"
    size = fields.get("font_size")
    size = float(region["font_size"] or 16) if size is None else float(size)
    size = max(8.0, min(72.0, size))
    ocr_text = region.get("ocr_text") if fields.get("ocr_text") is None else str(fields.get("ocr_text"))[:4000]
    translation = region.get("translation") if fields.get("translation") is None else str(fields.get("translation"))[:4000]
    flags = json.loads(region.get("flags") or "[]")
    if "user-edited" not in flags:
        flags.append("user-edited")
    jm.db.x(
        "UPDATE regions SET x=?,y=?,w=?,h=?,font=?,font_size=?,ocr_text=?,translation=?,edited=1,flags=?,updated=? "
        "WHERE job_id=? AND region_id=?",
        (x, y, w, h, font, size, ocr_text, translation, json.dumps(flags, ensure_ascii=False), time.time(), job_id, region_id),
    )
    prefix = "v" if region.get("direction") == "vertical" else "h"
    block = jm.db.q1("SELECT block_id FROM blocks WHERE job_id=? AND block_id=?", (job_id, region_id))
    if block:
        status = "done" if (translation or "").strip() else "pending"
        jm.db.x(
            "UPDATE blocks SET source=?, target=?, xpath=?, status=?, flags=? WHERE job_id=? AND block_id=?",
            (ocr_text, translation, f"{prefix}:{x},{y},{w},{h}", status, json.dumps(flags, ensure_ascii=False), job_id, region_id),
        )
    elif (ocr_text or "").strip():
        jm.db.x(
            "INSERT INTO blocks(job_id,block_id,idx,file,chapter,kind,xpath,source,target,furigana,status,flags,updated)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, region_id, region["idx"], "page:" + region["page_id"], region["page_id"], "balloon",
             f"{prefix}:{x},{y},{w},{h}", ocr_text, translation, "{}",
             "done" if (translation or "").strip() else "pending", json.dumps(flags, ensure_ascii=False), time.time()),
        )
    from .hq import invalidate_page_review
    invalidate_page_review(jm, job_id, region["page_id"])
    jm.db.event(job_id, "info", f"{region_id} 편집 저장. 이 페이지의 이미지 검수는 오래되어 다시 호출하지 않습니다.")
    return jm.db.q1("SELECT * FROM regions WHERE job_id=? AND region_id=?", (job_id, region_id))


async def retranslate_region(jm, job_id: str, region_id: str):
    from .engines import make_engine
    region = jm.db.q1("SELECT * FROM regions WHERE job_id=? AND region_id=?", (job_id, region_id))
    if not region:
        raise KeyError(region_id)
    if not (region.get("ocr_text") or "").strip():
        raise ValueError("OCR 텍스트가 비어 있어 재번역할 수 없습니다.")
    job = jm.db.get_job(job_id)
    settings = json.loads(job["settings"])
    block = jm.db.q1("SELECT * FROM blocks WHERE job_id=? AND block_id=?", (job_id, region_id))
    if not block:
        update_region(jm, job_id, region_id, {"ocr_text": region["ocr_text"]})
        block = jm.db.q1("SELECT * FROM blocks WHERE job_id=? AND block_id=?", (job_id, region_id))
    jm.db.x("UPDATE blocks SET source=?, target=NULL, status='pending' WHERE job_id=? AND block_id=?",
            (region["ocr_text"], job_id, region_id))
    block = jm.db.q1("SELECT * FROM blocks WHERE job_id=? AND block_id=?", (job_id, region_id))
    engine = make_engine({**settings, "api_key": jm.secrets.get(job_id, "")})
    await jm.translate_blocks(job_id, engine, settings, [block], chapter=block["chapter"], retry_mode=True)
    _sync_regions_from_blocks(jm, job_id)
    return jm.db.q1("SELECT * FROM regions WHERE job_id=? AND region_id=?", (job_id, region_id))


async def rebuild_edited(jm, job_id: str):
    if job_id in jm.tasks and not jm.tasks[job_id].done():
        raise RuntimeError("작업이 실행 중이라 다시 내보낼 수 없습니다.")
    job = jm.db.get_job(job_id)
    settings = json.loads(job["settings"])
    if settings.get("mode") != "comic":
        raise ValueError("편집 후 다시 내보내기는 만화 작업만 지원합니다.")
    return await _finalize_image(jm, job_id, settings, {"rounds": 0, "blocks": 0, "note": "편집 내용으로 다시 내보내기"})
