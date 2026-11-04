import asyncio
import hashlib
import json
import os
import re
import zipfile

import pytest

from epub_translator.pipeline import JobManager
from epub_translator.epub_model import EpubBook
from epub_translator.qa import find_epubcheck


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_job(jm, path, **settings):
    job = jm.create_job(path, {"engine": "mock", "chunk_size": 300, **settings})

    async def go():
        await jm.start(job["job_id"])
    asyncio.run(go())
    return jm.db.get_job(job["job_id"])


@pytest.mark.parametrize("ver", ["v3", "v2"])
def test_end_to_end_mock(tmp_path, samples, ver):
    src = samples[ver]
    h0 = sha(src)
    jm = JobManager(str(tmp_path / "data"))
    job = run_job(jm, src)
    assert sha(src) == h0, "original must not be modified"
    assert job["status"] == "completed", job["error"]
    assert job["failed_blocks"] == 0 and job["done_blocks"] == job["total_blocks"] > 20
    rep = json.load(open(os.path.join(os.path.dirname(job["out_path"]), "report.json"), encoding="utf-8"))
    assert rep["structure"]["ok"], rep["structure"]
    assert rep["translation"]["error"] == 0 and rep["translation"]["untranslated"] == 0
    assert rep["translation"]["dialogue_preserved"] == rep["translation"]["dialogue_source"] > 0
    assert os.path.exists(os.path.join(os.path.dirname(job["out_path"]), "report.html"))
    out = EpubBook(job["out_path"])
    assert out.language == "ko"
    z = zipfile.ZipFile(job["out_path"])
    assert z.infolist()[0].filename == "mimetype" and z.infolist()[0].compress_type == zipfile.ZIP_STORED
    assert set(z.namelist()) - {"OEBPS/kt-ko-readability.css"} == set(zipfile.ZipFile(src).namelist())
    assert b"keep-all" in z.read("OEBPS/kt-ko-readability.css")
    ch1 = z.read("OEBPS/text/chapter1.xhtml").decode()
    body = re.sub(r"<[^>]+>", "", ch1.split("<body>")[1])
    assert not re.search(r"[\u3041-\u309F\u30A1-\u30FA]", body), "no kana left in body"
    assert 'id="ch1"' in ch1 and 'href="notes.xhtml#n1"' in ch1 and 'id="r1"' in ch1
    assert 'src="../images/illust1.png"' in ch1
    css = z.read("OEBPS/styles/style.css").decode()
    assert "vertical-rl" not in css and "horizontal-tb" in css
    opf = z.read("OEBPS/content.opf").decode()
    assert 'page-progression-direction="rtl"' not in opf
    if ver == "v3":
        assert "<ruby>" in ch1 and "<rt>" in ch1     # meaningful ruby kept
        assert "みなせ" not in ch1                     # furigana removed
    ec = rep["epubcheck"]
    if find_epubcheck():
        assert ec["available"] and ec["fatal"] == 0 and ec["errors"] == 0, ec
    # block mapping: every source block has a stored target
    rows = jm.db.q("SELECT source,target,status FROM blocks WHERE job_id=?", (job["job_id"],))
    assert all(r["target"] and r["status"] in ("done", "warn") for r in rows)


def test_fault_injection_recovers(tmp_path, samples, monkeypatch):
    monkeypatch.setenv("MOCK_FAULTS", "fail_once,drop_first,truncate_once,japanese_once,dup_once")
    jm = JobManager(str(tmp_path / "data"))
    job = run_job(jm, samples["v3"], max_retries=3)
    assert job["status"] == "completed", job["error"]
    assert job["failed_blocks"] == 0
    assert job["retries"] >= 1
    ev = " ".join(e["message"] for e in jm.db.events(job["job_id"], 200))
    assert "API 오류" in ev and "분할 재번역" in ev and "검증 실패" in ev
    rep = json.load(open(os.path.join(os.path.dirname(job["out_path"]), "report.json"), encoding="utf-8"))
    assert rep["retranslation"]["rounds"] >= 1 and rep["translation"]["duplicates"] == 0


def test_resume_uses_cache_and_job_hash(tmp_path, samples):
    jm = JobManager(str(tmp_path / "data"))
    job = run_job(jm, samples["v3"])
    calls = job["api_calls"]
    # same file + settings -> same job id; reset blocks to simulate an interrupted run
    jm.db.x("UPDATE blocks SET status='pending', target=NULL WHERE job_id=?", (job["job_id"],))
    jm.db.x("UPDATE chunks SET status='pending' WHERE job_id=?", (job["job_id"],))
    jm2 = JobManager(str(tmp_path / "data"))          # "program restart"
    job2 = run_job(jm2, samples["v3"])
    assert job2["job_id"] == job["job_id"]
    assert job2["status"] == "completed" and job2["failed_blocks"] == 0
    assert job2["api_calls"] == calls, "cached chunks must not be re-sent to the API"


def test_pause_cancel_then_resume(tmp_path, samples):
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(samples["v3"], {"engine": "mock", "chunk_size": 100, "concurrency": 1})
    jid = job["job_id"]

    async def go():
        t = jm.start(jid)
        await asyncio.sleep(0.03)
        jm.pause(jid)
        await asyncio.sleep(0.1)
        d1 = jm.db.get_job(jid)["done_blocks"]
        await asyncio.sleep(0.1)
        assert jm.db.get_job(jid)["done_blocks"] == d1, "no progress while paused"
        jm.cancel_job(jid)
        await t
        return d1
    d1 = asyncio.run(go())
    st = jm.db.get_job(jid)
    assert st["status"] == "cancelled" and st["done_blocks"] < st["total_blocks"]

    async def go2():
        await jm.start(jid)
    asyncio.run(go2())
    st = jm.db.get_job(jid)
    assert st["status"] == "completed" and st["done_blocks"] == st["total_blocks"]


def test_quota_exhaustion_stops_and_resumes(tmp_path, samples, monkeypatch):
    monkeypatch.setenv("MOCK_FAULTS", "quota_once")
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(samples["v3"], {"engine": "mock", "chunk_size": 100, "concurrency": 2})
    jid = job["job_id"]

    async def go():
        await jm.start(jid)
    asyncio.run(go())
    st = jm.db.get_job(jid)
    assert st["status"] == "error" and "quota" in st["error"]
    assert 0 < st["done_blocks"] < st["total_blocks"]
    asyncio.run(go())                      # quota recovered -> resume from unfinished blocks
    st = jm.db.get_job(jid)
    assert st["status"] == "completed" and st["done_blocks"] == st["total_blocks"]
