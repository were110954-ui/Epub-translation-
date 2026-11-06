"""Job orchestration: analysis -> chunking -> translation (async, resumable) -> validation ->
selective retranslation -> EPUB rebuild -> QA reports."""
from __future__ import annotations
import asyncio
import hashlib
import json
from .epub_model import TOKEN_RE
from .validator import flatten_tokens
import logging
import os
import random
import re
import shutil
import time
from datetime import datetime

from .chunker import make_chunks
from .context import build_prompt, parse_response
from .db import DB
from .engines import EngineError, make_engine, price_for, DEFAULT_MODELS
from .epub_model import EpubBook, sha256_file
from .qa import structure_check, run_epubcheck, translation_check, write_reports
from .rebuild import apply_translations, write_epub
from .validator import check_block, find_duplicates, repair_tokens

log = logging.getLogger("epub_translator")

DEFAULT_SETTINGS = {
    "engine": "mock", "model": "", "api_key": "", "base_url": "", "temperature": 0.3,
    "concurrency": 4, "chunk_size": 2500, "max_retries": 4, "src_lang": "ja", "tgt_lang": "ko",
    "drop_furigana": True, "horizontal": True, "retranslate_rounds": 2, "max_output_tokens": 16384,
    "run_epubcheck": True, "dpi": 300, "page_parallel": 2, "rtl": True, "rights_ack": False,
    "ocr_backend": "auto",
}


def job_key(file_hash: str, s: dict) -> str:
    sig = f"{s['engine']}|{s.get('model') or DEFAULT_MODELS[s['engine']]}|{s['src_lang']}|{s['tgt_lang']}|{int(bool(s['drop_furigana']))}"
    return file_hash[:16] + "-" + hashlib.sha1(sig.encode()).hexdigest()[:8]


def estimate(blocks: list[dict], chunks: list[dict], model: str) -> dict:
    src_chars = sum(len(b["source"]) for b in blocks)
    n = max(1, len(chunks))
    prompt = int(n * 1300 + src_chars * 1.15 + n * 400)      # system + context + source (~1 token/JA char)
    completion = int(src_chars * 1.35 + n * 250)              # Korean output + JSON/summary overhead
    pin, pout = price_for(model)
    return {"blocks": len(blocks), "chunks": len(chunks), "source_chars": src_chars,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "cost_usd": round(prompt / 1e6 * pin + completion / 1e6 * pout, 4),
            "price_per_1m": {"input": pin, "output": pout},
            "note": "추정치 (재시도/재번역 제외, 모델별 토크나이저 차이 있음)"}


class JobManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "jobs"), exist_ok=True)
        self.db = DB(os.path.join(data_dir, "state.sqlite3"))
        self.tasks: dict[str, asyncio.Task] = {}
        self.pause_ev: dict[str, asyncio.Event] = {}
        self.cancel: set[str] = set()
        self.secrets: dict[str, str] = {}   # job_id -> api key (memory only, never stored in DB)

    def job_dir(self, job_id):
        d = os.path.join(self.data_dir, "jobs", job_id)
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------ create/analyze
    def create_job(self, src_path: str, settings: dict, file_name: str | None = None) -> dict:
        s = {**DEFAULT_SETTINGS, **{k: v for k, v in settings.items() if v not in (None, "")}}
        s["model"] = s.get("model") or DEFAULT_MODELS[s["engine"]]
        api_key = s.pop("api_key", "") or ""
        name = file_name or os.path.basename(src_path)
        from .ingest import resolve_mode
        from .safety import RightsError
        mode, kind = resolve_mode(src_path, name, s.get("mode"))
        s["mode"] = mode
        s["input_kind"] = kind
        if mode != "epub":
            if not s.get("rights_ack"):
                raise RightsError("이 파일을 번역·변환할 법적 권한이 있는지 확인해야 합니다. 권리 확인 후에만 진행할 수 있습니다.")
            from .image_jobs import create_image_job
            return create_image_job(self, src_path, s, name, api_key)
        fh = sha256_file(src_path)
        job_id = job_key(fh, s)
        if api_key:
            self.secrets[job_id] = api_key
        existing = self.db.get_job(job_id)
        if existing:
            # resume: keep blocks/cache; update tunables (concurrency, temperature, chunk size untouched for existing chunks)
            st = json.loads(existing["settings"])
            st.update({k: s[k] for k in ("concurrency", "temperature", "max_retries", "retranslate_rounds", "base_url", "run_epubcheck", "max_output_tokens")})
            self.db.update_job(job_id, settings=json.dumps(st))
            self.db.event(job_id, "info", "동일 파일/설정의 기존 작업을 재사용합니다 (이어하기).")
            self.resync_blocks(job_id, existing["src_path"], st)
            return self.db.get_job(job_id)
        d = self.job_dir(job_id)
        src_copy = os.path.join(d, "source.epub")
        shutil.copyfile(src_path, src_copy)          # never touch the user's original
        book = EpubBook(src_copy)                    # raises DRMError
        blocks, _ = book.extract_blocks(drop_furigana=s["drop_furigana"])
        rows = [dict(job_id=job_id, block_id=b.block_id, idx=b.index, file=b.file, chapter=b.chapter, kind=b.kind,
                     xpath=b.xpath, source=b.source, furigana=json.dumps(b.furigana, ensure_ascii=False)) for b in blocks]
        name = file_name or os.path.basename(src_path)
        out_name = os.path.splitext(name)[0] + f".{s['tgt_lang']}.epub"
        self.db.x("INSERT INTO jobs(job_id,file_hash,file_name,src_path,out_path,status,settings,title,created,updated,total_blocks)"
                  " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (job_id, fh, name, src_copy, os.path.join(d, out_name), "ready", json.dumps(s), book.title,
                   time.time(), time.time(), len(rows)))
        self.db.x("INSERT OR IGNORE INTO blocks(job_id,block_id,idx,file,chapter,kind,xpath,source,furigana,status,updated)"
                  " VALUES(:job_id,:block_id,:idx,:file,:chapter,:kind,:xpath,:source,:furigana,'pending',0)", rows, many=True)
        chunks = make_chunks([dict(r) for r in rows], int(s["chunk_size"]))
        crow = []
        for seq, c in enumerate(chunks):
            crow.append((job_id, c["chunk_id"], seq, c["chapter"], c["file"], json.dumps(c["block_ids"])))
            self.db.x("UPDATE blocks SET chunk_id=? WHERE job_id=? AND block_id IN (%s)" % ",".join("?" * len(c["block_ids"])),
                      (c["chunk_id"], job_id, *c["block_ids"]))
        self.db.x("INSERT INTO chunks(job_id,chunk_id,seq,chapter,file,block_ids) VALUES(?,?,?,?,?,?)", crow, many=True)
        # seed glossary readings from furigana (names with reading hints; target filled by LLM)
        self.db.event(job_id, "info", f"분석 완료: 블록 {len(rows)}개, 청크 {len(chunks)}개, EPUB {book.version}")
        est = estimate(rows, chunks, s["model"])
        self.db.update_job(job_id, report_json=json.dumps({"estimate": est}, ensure_ascii=False))
        return self.db.get_job(job_id)

    def resync_blocks(self, job_id, src_path, s):
        """After an extractor upgrade (e.g. drop-cap lead-in merging) re-extract and reset only the blocks
        whose source changed; everything else keeps its cached translation."""
        try:
            blocks, _ = EpubBook(src_path).extract_blocks(drop_furigana=s.get("drop_furigana", True))
        except Exception:  # noqa
            return 0
        cur = {r["block_id"]: r for r in self.db.q("SELECT block_id, source, xpath FROM blocks WHERE job_id=?", (job_id,))}
        n = 0
        for b in blocks:
            r = cur.get(b.block_id)
            if r and (r["source"] != b.source or r["xpath"] != b.xpath):
                self.db.x("UPDATE blocks SET source=?, xpath=?, target=NULL, status='pending', flags=NULL, error=NULL WHERE job_id=? AND block_id=?",
                          (b.source, b.xpath, job_id, b.block_id))
                n += 1
        if n:
            self.db.event(job_id, "info", f"원문 구조 재분석: {n}개 블록의 원문이 바뀌어 재번역 대상으로 표시")
            self.db.refresh_counts(job_id)
        return n

    def job_estimate(self, job_id):
        j = self.db.get_job(job_id)
        s = json.loads(j["settings"])
        mode = s.get("mode") or "epub"
        if mode in ("pdf", "cbz"):
            n = self.db.q1("SELECT COUNT(*) n FROM pages WHERE job_id=?", (job_id,))["n"]
            return {"blocks": n, "chunks": 0, "source_chars": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "cost_usd": 0.0, "price_per_1m": {"input": 0, "output": 0},
                    "note": "PDF/CBZ를 EPUB으로 변환만 합니다. 번역 API를 호출하지 않습니다."}
        if mode == "comic":
            rows = self.db.q("SELECT source FROM blocks WHERE job_id=? AND kind='balloon'", (job_id,))
            if rows:
                chunks = self.db.q("SELECT chunk_id FROM chunks WHERE job_id=?", (job_id,))
                est = estimate(rows, chunks or [{"chunk_id": "pending"}], s["model"])
                est["note"] = "검출된 대사 기준 추정치입니다 (재시도 제외)."
                return est
            n = self.db.q1("SELECT COUNT(*) n FROM pages WHERE job_id=?", (job_id,))["n"] or 1
            fake = [{"source": "あ" * 20} for _ in range(max(1, n * 4))]
            chunks = [{"chunk_id": str(i)} for i in range(max(1, n))]
            est = estimate(fake, chunks, s["model"])
            est["note"] = "OCR 이전 가정치입니다 (페이지당 대사 4개). 실제 비용은 검출 결과에 따릅니다."
            est["provisional"] = True
            return est
        rows = self.db.q("SELECT source FROM blocks WHERE job_id=?", (job_id,))
        chunks = self.db.q("SELECT chunk_id FROM chunks WHERE job_id=?", (job_id,))
        return estimate(rows, chunks, s["model"])

    # ------------------------------------------------------------ control
    def start(self, job_id) -> asyncio.Task:
        if job_id in self.tasks and not self.tasks[job_id].done():
            return self.tasks[job_id]
        self.cancel.discard(job_id)
        ev = asyncio.Event()
        ev.set()
        self.pause_ev[job_id] = ev
        t = asyncio.create_task(self.run(job_id))
        self.tasks[job_id] = t
        return t

    def pause(self, job_id):
        if job_id in self.pause_ev:
            self.pause_ev[job_id].clear()
            self.db.update_job(job_id, status="paused")
            self.db.event(job_id, "info", "일시정지")

    def resume(self, job_id):
        if job_id in self.pause_ev and job_id in self.tasks and not self.tasks[job_id].done():
            self.pause_ev[job_id].set()
            self.db.update_job(job_id, status="running")
            self.db.event(job_id, "info", "재개")
        else:
            self.start(job_id)

    def cancel_job(self, job_id):
        self.cancel.add(job_id)
        if job_id in self.pause_ev:
            self.pause_ev[job_id].set()
        self.db.update_job(job_id, status="cancelled")
        self.db.event(job_id, "warn", "취소 요청 (완료된 블록은 저장되어 있어 다시 시작하면 이어서 진행)")

    async def _gate(self, job_id):
        ev = self.pause_ev.get(job_id)
        if ev:
            await ev.wait()
        if job_id in self.cancel:
            raise asyncio.CancelledError()

    # ------------------------------------------------------------ main run
    async def run(self, job_id):
        db = self.db
        job = db.get_job(job_id)
        s = json.loads(job["settings"])
        if (s.get("mode") or "epub") != "epub":
            from .image_jobs import run_image_job
            await run_image_job(self, job_id)
            return
        try:
            engine = make_engine({**s, "api_key": self.secrets.get(job_id, "")})
        except EngineError as e:
            db.update_job(job_id, status="error", error=str(e))
            db.event(job_id, "error", str(e))
            return
        db.update_job(job_id, status="running", error=None)
        db.x("UPDATE blocks SET status='pending' WHERE job_id=? AND status='running'", (job_id,))
        db.refresh_counts(job_id)
        try:
            chunks = db.q("SELECT * FROM chunks WHERE job_id=? AND status!='done' ORDER BY seq", (job_id,))
            db.event(job_id, "info", f"번역 시작: 남은 청크 {len(chunks)}개 (엔진 {engine.name}/{engine.model})")
            by_file: dict[str, list] = {}
            for c in chunks:
                by_file.setdefault(c["file"], []).append(c)
            sem = asyncio.Semaphore(int(s.get("concurrency", 4)))

            async def do_file(cl):
                for c in cl:
                    await self._gate(job_id)
                    async with sem:
                        await self._gate(job_id)
                        await self.translate_chunk(job_id, engine, s, c)

            await asyncio.gather(*(do_file(cl) for cl in by_file.values()))
            self.local_repair(job_id, s)
            # ---- selective retranslation rounds
            rt_blocks = 0
            rounds = 0
            for rnd in range(int(s.get("retranslate_rounds", 2))):
                await self._gate(job_id)
                targets = self.blocks_needing_retranslation(job_id, include_warn=(rnd == 0))
                if not targets:
                    break
                rounds += 1
                rt_blocks += len(targets)
                db.event(job_id, "info", f"선택적 재번역 라운드 {rnd + 1}: {len(targets)}개 블록")
                groups, cur = [], []
                for t in targets:
                    if cur and (len(cur) >= 12 or cur[-1]["file"] != t["file"]):
                        groups.append(cur)
                        cur = []
                    cur.append(t)
                if cur:
                    groups.append(cur)

                async def do_group(g):
                    async with sem:
                        await self._gate(job_id)
                        await self.translate_blocks(job_id, engine, s, g, chapter=g[0]["chapter"], retry_mode=True)
                await asyncio.gather(*(do_group(g) for g in groups))
            db.x("UPDATE blocks SET status='failed' WHERE job_id=? AND status IN ('retry','pending')", (job_id,))
            db.refresh_counts(job_id)
            await self.finalize(job_id, {"rounds": rounds, "blocks": rt_blocks})
        except EngineError as e:
            self.cancel.discard(job_id)
            db.x("UPDATE blocks SET status='pending' WHERE job_id=? AND status='running'", (job_id,))
            db.refresh_counts(job_id)
            db.update_job(job_id, status="error", error=str(e)[:500])
            db.event(job_id, "error", f"중단: {str(e)[:300]}")
        except asyncio.CancelledError:
            if job_id in self.cancel and db.get_job(job_id)["status"] == "error":
                return
            db.x("UPDATE blocks SET status='pending' WHERE job_id=? AND status='running'", (job_id,))
            db.refresh_counts(job_id)
            db.update_job(job_id, status="cancelled")
        except Exception as e:  # noqa
            log.exception("job failed")
            db.update_job(job_id, status="error", error=repr(e))
            db.event(job_id, "error", f"작업 오류: {e!r} — 다시 시작하면 미완료 부분부터 이어집니다.")

    def local_repair(self, job_id, s):
        """Re-validate stored failed targets after local token repair (no API call)."""
        n = 0
        for r in self.db.q("SELECT * FROM blocks WHERE job_id=? AND status IN ('retry','failed') AND target IS NOT NULL", (job_id,)):
            dst = repair_tokens(r["source"], r["target"])
            sev, flags = check_block(r["source"], dst, r["kind"], s["tgt_lang"])
            if sev != "error":
                self.db.x("UPDATE blocks SET target=?, status=?, flags=?, error=NULL WHERE job_id=? AND block_id=?",
                          (dst, "warn" if sev == "warn" else "done", json.dumps(flags, ensure_ascii=False), job_id, r["block_id"]))
                n += 1
                continue
            flat = flatten_tokens(r["source"], dst)
            if flat is not None:
                sev2, flags2 = check_block(TOKEN_RE.sub("", r["source"]), TOKEN_RE.sub("", flat), r["kind"], s["tgt_lang"])
                if sev2 != "error":
                    self.db.x("UPDATE blocks SET target=?, status='warn', flags=?, error=NULL WHERE job_id=? AND block_id=?",
                              (flat, json.dumps(flags2 + ["tags-flattened"], ensure_ascii=False), job_id, r["block_id"]))
                    n += 1
        if n:
            self.db.event(job_id, "info", f"로컬 토큰 복구로 {n}개 블록 통과")
        self.db.refresh_counts(job_id)

    def blocks_needing_retranslation(self, job_id, include_warn=False):
        db = self.db
        rows = db.q("SELECT * FROM blocks WHERE job_id=? ORDER BY idx", (job_id,))
        need = {r["block_id"] for r in rows if r["status"] in ("retry", "pending", "failed")}
        if include_warn:
            for r in rows:
                if r["status"] == "warn" and any(f in json.loads(r["flags"] or "[]") for f in ("too-short", "dialogue-lost", "sentences-missing", "numbers-mismatch", "leadin-old-translation")):
                    need.add(r["block_id"])
        for bid in find_duplicates([r for r in rows if r["target"]]):
            need.add(bid)
        # glossary consistency -> retranslate offending blocks (bounded)
        gl = db.q("SELECT * FROM glossary WHERE job_id=? AND target!=''", (job_id,))
        cnt = 0
        for g in gl:
            if len(g["source"]) < 2:
                continue
            occ = [r for r in rows if r["target"] and g["source"] in r["source"]]
            if len(occ) >= 3:
                miss = [r for r in occ if g["target"] not in r["target"]]
                if 0 < len(miss) <= len(occ) * 0.5:
                    for r in miss[:20]:
                        if cnt < 200 and r["block_id"] not in need:
                            need.add(r["block_id"])
                            cnt += 1
        return [r for r in rows if r["block_id"] in need]

    async def translate_chunk(self, job_id, engine, s, chunk):
        db = self.db
        ids = json.loads(chunk["block_ids"])
        blocks = db.q("SELECT * FROM blocks WHERE job_id=? AND block_id IN (%s) ORDER BY idx" % ",".join("?" * len(ids)),
                      (job_id, *ids))
        pending = [b for b in blocks if b["status"] not in ("done", "warn")]
        db.update_job(job_id, current_chapter=chunk["chapter"])
        if pending:
            summary = await self.translate_blocks(job_id, engine, s, pending, chunk["chapter"])
            db.x("UPDATE chunks SET summary=? WHERE job_id=? AND chunk_id=?", (summary or "", job_id, chunk["chunk_id"]))
        st = db.q1("SELECT SUM(status IN ('done','warn')) d, COUNT(*) n FROM blocks WHERE job_id=? AND chunk_id=?", (job_id, chunk["chunk_id"]))
        db.x("UPDATE chunks SET status=?, attempts=attempts+1 WHERE job_id=? AND chunk_id=?",
             ("done" if st["d"] == st["n"] else "partial", job_id, chunk["chunk_id"]))   # checkpoint
        db.refresh_counts(job_id)

    def _context(self, job_id, first_idx):
        db = self.db
        glossary = db.q("SELECT * FROM glossary WHERE job_id=? AND target!=''", (job_id,))
        chars = db.q("SELECT * FROM characters WHERE job_id=?", (job_id,))
        prev_chunks = db.q("SELECT c.summary FROM chunks c JOIN blocks b ON b.job_id=c.job_id AND b.chunk_id=c.chunk_id "
                           "WHERE c.job_id=? AND c.summary IS NOT NULL AND c.summary!='' GROUP BY c.chunk_id "
                           "HAVING MIN(b.idx) < ? ORDER BY MIN(b.idx)", (job_id, first_idx))
        prev = db.q("SELECT source,target FROM blocks WHERE job_id=? AND idx<? AND target IS NOT NULL AND kind IN ('block','balloon') "
                    "ORDER BY idx DESC LIMIT 4", (job_id, first_idx))
        return glossary, chars, [p["summary"] for p in prev_chunks], [(p["source"], p["target"]) for p in reversed(prev)]

    async def translate_blocks(self, job_id, engine, s, blocks, chapter, retry_mode=False, depth=0):
        """Translate a group of blocks with retries, truncation splitting, validation and caching."""
        db = self.db
        for b in blocks:
            b["furigana"] = json.loads(b["furigana"] or "{}") if isinstance(b.get("furigana"), str) else (b.get("furigana") or {})
        key_src = json.dumps([[b["block_id"], b["source"]] for b in blocks], ensure_ascii=False)
        ckey = hashlib.sha256(f"{engine.name}|{engine.model}|{s['tgt_lang']}|{retry_mode}|{key_src}".encode()).hexdigest()
        data = None if retry_mode else db.cache_get(ckey)
        if data is None:
            glossary, chars, sums, prev = self._context(job_id, blocks[0]["idx"])
            system, user = build_prompt(s["src_lang"], s["tgt_lang"], blocks, glossary, chars, sums, prev, chapter)
            if retry_mode:
                user += ("\n\nIMPORTANT: a previous attempt for these blocks was rejected by QA (missing/short/untranslated/"
                         "duplicated/inconsistent terms). Translate each block fully and independently, follow the glossary exactly, "
                         "keep every <tN> token.")
            attempt = 0
            while True:
                attempt += 1
                db.x("UPDATE blocks SET status='running' WHERE job_id=? AND block_id IN (%s) AND status NOT IN ('done','warn')"
                     % ",".join("?" * len(blocks)), (job_id, *[b["block_id"] for b in blocks]))
                try:
                    res = await engine.complete(system, user)
                    db.add_usage(job_id, res.prompt_tokens, res.completion_tokens, 1, 1 if attempt > 1 else 0)
                    if res.truncated:
                        raise ValueError("truncated")
                    data = parse_response(res.text)
                    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
                        raise ValueError("bad-json-shape")
                    break
                except (ValueError, json.JSONDecodeError) as e:
                    # truncated/invalid output: split the group and retranslate halves
                    db.event(job_id, "warn", f"[{chapter}] 응답 잘림/파싱 실패 ({e}); 블록 {len(blocks)}개 → 분할 재번역")
                    if len(blocks) > 1 and depth < 6:
                        mid = len(blocks) // 2
                        a = await self.translate_blocks(job_id, engine, s, blocks[:mid], chapter, retry_mode, depth + 1)
                        b2 = await self.translate_blocks(job_id, engine, s, blocks[mid:], chapter, retry_mode, depth + 1)
                        return " ".join(x for x in (a, b2) if x)
                    if attempt >= int(s.get("max_retries", 4)):
                        self._mark(job_id, blocks, "retry", "truncated/invalid output")
                        return ""
                    db.add_usage(job_id, 0, 0, 0, 0)
                except EngineError as e:
                    db.event(job_id, "warn", f"[{chapter}] API 오류 (시도 {attempt}): {str(e)[:200]}")
                    if not e.retryable or attempt >= int(s.get("max_retries", 4)):
                        self._mark(job_id, blocks, "retry", str(e)[:300])
                        if not e.retryable:
                            self.cancel.add(job_id)      # stop sibling workers; job can be resumed later
                            raise
                        return ""
                    delay = e.retry_after if e.retry_after else (2 ** attempt) + random.random()
                    await asyncio.sleep(min(120, delay))
                await self._gate(job_id)
            db.cache_put(ckey, data)
        out = {}
        for it in data.get("translations", []):
            if isinstance(it, dict) and "id" in it:
                out.setdefault(str(it["id"]), str(it.get("text", "")))
        now = time.time()
        for b in blocks:
            dst = out.get(b["block_id"])
            if dst:
                # keep the original paragraph indentation (full-width space) that models often drop
                lead = re.match(r"^[\u3000 ]*", b["source"]).group(0)
                if lead and not dst.startswith(lead):
                    dst = lead + dst.lstrip("\u3000 ")
            dst = repair_tokens(b["source"], dst)
            sev, flags = check_block(b["source"], dst, b["kind"], s["tgt_lang"])
            if sev == "error":
                status = "retry"
            else:
                status = "warn" if sev == "warn" else "done"
            # in retry mode keep an earlier acceptable translation if the new one is worse
            if status == "retry" and b.get("target") and b["status"] in ("done", "warn"):
                continue
            db.x("UPDATE blocks SET target=?, status=?, attempts=attempts+1, flags=?, error=?, updated=? WHERE job_id=? AND block_id=?",
                 (dst if status != "retry" else (dst or None), status, json.dumps(flags, ensure_ascii=False),
                  None if status != "retry" else ",".join(flags), now, job_id, b["block_id"]))
            if status == "retry":
                db.event(job_id, "warn", f"블록 {b['block_id']} 검증 실패: {', '.join(flags)}")
        # glossary/characters
        for t in data.get("new_terms") or []:
            if isinstance(t, dict) and t.get("source") and t.get("target"):
                db.x("INSERT OR IGNORE INTO glossary(job_id,source,target,type,reading) VALUES(?,?,?,?,?)",
                     (job_id, str(t["source"])[:100], str(t["target"])[:100], str(t.get("type", ""))[:30],
                      next((b["furigana"].get(t["source"]) for b in blocks if t["source"] in b["furigana"]), "")))
        for c in data.get("characters") or []:
            if isinstance(c, dict) and c.get("name"):
                db.x("INSERT OR IGNORE INTO characters(job_id,name,korean,speech,honorifics) VALUES(?,?,?,?,?)",
                     (job_id, str(c["name"])[:60], str(c.get("korean", ""))[:60], str(c.get("speech", ""))[:300],
                      str(c.get("honorifics", ""))[:300]))
                if c.get("korean"):
                    db.x("INSERT OR IGNORE INTO glossary(job_id,source,target,type,reading) VALUES(?,?,?,?,?)",
                         (job_id, str(c["name"])[:100], str(c["korean"])[:100], "character", ""))
        db.refresh_counts(job_id)
        return str(data.get("summary") or "")

    def _mark(self, job_id, blocks, status, err):
        self.db.x("UPDATE blocks SET status=?, error=?, attempts=attempts+1 WHERE job_id=? AND block_id IN (%s) AND status NOT IN ('done','warn')"
                  % ",".join("?" * len(blocks)), (status, err, job_id, *[b["block_id"] for b in blocks]))

    # ------------------------------------------------------------ finalize
    async def finalize(self, job_id, retrans: dict | None = None):
        db = self.db
        job = db.get_job(job_id)
        s = json.loads(job["settings"])
        db.update_job(job_id, status="building", current_chapter="EPUB 생성/검수")
        rows = db.q("SELECT * FROM blocks WHERE job_id=? ORDER BY idx", (job_id,))
        book = EpubBook(job["src_path"])
        good = [r for r in rows if r["status"] in ("done", "warn") and r["target"]]
        repl = await asyncio.to_thread(apply_translations, book, good, s["tgt_lang"], s["drop_furigana"], s["horizontal"])
        await asyncio.to_thread(write_epub, book, repl, job["out_path"])
        struct = await asyncio.to_thread(structure_check, job["out_path"])
        ec = await asyncio.to_thread(run_epubcheck, job["out_path"]) if s.get("run_epubcheck", True) else {"available": False, "message": "disabled"}
        glossary = db.q("SELECT source,target,type,reading FROM glossary WHERE job_id=? ORDER BY source", (job_id,))
        chars = db.q("SELECT name,korean,speech,honorifics FROM characters WHERE job_id=? ORDER BY name", (job_id,))
        tq = translation_check([dict(r, target=r["target"] if r["status"] in ("done", "warn") else None) for r in rows],
                               glossary, chars, s["tgt_lang"])
        job = db.get_job(job_id)
        prev = json.loads(job["report_json"] or "{}") if (job["report_json"] or "").startswith("{") else {}
        report = {
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "job": {"job_id": job_id, "title": job["title"], "file_name": job["file_name"], "file_hash": job["file_hash"],
                    "engine": s["engine"], "model": s["model"], "total_blocks": job["total_blocks"],
                    "done_blocks": job["done_blocks"], "failed_blocks": job["failed_blocks"],
                    "prompt_tokens": job["prompt_tokens"], "completion_tokens": job["completion_tokens"],
                    "api_calls": job["api_calls"], "retries": job["retries"],
                    "actual_cost_usd": round(job["prompt_tokens"] / 1e6 * price_for(s["model"])[0] +
                                             job["completion_tokens"] / 1e6 * price_for(s["model"])[1], 4)},
            "estimate": prev.get("estimate"),
            "retranslation": retrans or prev.get("retranslation") or {},
            "structure": struct, "epubcheck": ec, "translation": tq,
            "failed_blocks": [{"block_id": r["block_id"], "error": r["error"], "source": r["source"][:200]}
                              for r in rows if r["status"] == "failed"],
            "glossary": glossary, "characters": chars,
        }
        d = self.job_dir(job_id)
        jp, hp = os.path.join(d, "report.json"), os.path.join(d, "report.html")
        write_reports(report, jp, hp)
        ok = struct["ok"] and job["failed_blocks"] == 0
        db.update_job(job_id, status="completed" if ok else "completed_with_issues", report_json=json.dumps(report, ensure_ascii=False)[:5_000_000],
                      report_html=hp, current_chapter="완료")
        db.event(job_id, "info", f"완료: 구조검사 {'통과' if struct['ok'] else '문제'}, 실패 블록 {job['failed_blocks']}, "
                                 f"EPUBCheck {ec.get('errors', '-')} errors / {ec.get('fatal', '-')} fatal")
        return report

    def status(self, job_id):
        j = self.db.get_job(job_id)
        if not j:
            return None
        counts = {r["status"]: r["n"] for r in self.db.q("SELECT status, COUNT(*) n FROM blocks WHERE job_id=? GROUP BY status", (job_id,))}
        chapters = self.db.q("SELECT chapter, COUNT(*) n, SUM(status IN ('done','warn')) d FROM blocks WHERE job_id=? GROUP BY chapter ORDER BY MIN(idx)", (job_id,))
        s = json.loads(j["settings"])
        j = {k: v for k, v in j.items() if k not in ("report_json",)}
        j["settings"] = s
        j["counts"] = counts
        j["progress"] = round(100 * (j["done_blocks"] or 0) / max(1, j["total_blocks"] or 0), 2)
        j["chapters"] = chapters
        j["events"] = self.db.events(job_id, 30)
        j["running"] = job_id in self.tasks and not self.tasks[job_id].done()
        pages = self.db.q(
            "SELECT page_id, idx, stage, width, height, spread, method, error FROM pages WHERE job_id=? ORDER BY idx",
            (job_id,),
        )
        for page in pages:
            if page.get("error"):
                page["error"] = str(page["error"])[:200]
        j["page_progress"] = {
            "total": len(pages),
            "done": sum(1 for p in pages if p["stage"] == "done"),
            "failed": sum(1 for p in pages if p["stage"] == "failed"),
            "pages": pages,
        }
        review = self.db.q1("SELECT COUNT(*) n FROM regions WHERE job_id=? AND review=1", (job_id,))
        j["review_count"] = review["n"] if review else 0
        return j
