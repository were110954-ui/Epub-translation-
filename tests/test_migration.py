import os
import sqlite3
import time

from epub_translator.db import DB
from epub_translator.pipeline import JobManager

OLD_SCHEMA = """
CREATE TABLE jobs(
  job_id TEXT PRIMARY KEY, file_hash TEXT, file_name TEXT, src_path TEXT, out_path TEXT,
  status TEXT, settings TEXT, title TEXT, created REAL, updated REAL,
  total_blocks INTEGER DEFAULT 0, done_blocks INTEGER DEFAULT 0, failed_blocks INTEGER DEFAULT 0,
  current_chapter TEXT, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
  api_calls INTEGER DEFAULT 0, retries INTEGER DEFAULT 0, error TEXT, report_json TEXT, report_html TEXT);
CREATE TABLE blocks(
  job_id TEXT, block_id TEXT, idx INTEGER, file TEXT, chapter TEXT, kind TEXT, xpath TEXT,
  source TEXT, target TEXT, status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0,
  chunk_id TEXT, furigana TEXT, flags TEXT DEFAULT '[]', error TEXT, updated REAL,
  PRIMARY KEY(job_id, block_id));
CREATE TABLE chunks(
  job_id TEXT, chunk_id TEXT, seq INTEGER, chapter TEXT, file TEXT, block_ids TEXT,
  status TEXT DEFAULT 'pending', summary TEXT, attempts INTEGER DEFAULT 0, error TEXT,
  PRIMARY KEY(job_id, chunk_id));
CREATE TABLE cache(key TEXT PRIMARY KEY, response TEXT, created REAL);
CREATE TABLE glossary(
  job_id TEXT, source TEXT, target TEXT, type TEXT, reading TEXT, note TEXT, locked INTEGER DEFAULT 0,
  PRIMARY KEY(job_id, source));
CREATE TABLE characters(
  job_id TEXT, name TEXT, korean TEXT, speech TEXT, honorifics TEXT, note TEXT,
  PRIMARY KEY(job_id, name));
CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts REAL, level TEXT, message TEXT);
"""


def test_migration_preserves_rows_and_secrets(tmp_path, samples):
    data = tmp_path / "data"
    data.mkdir()
    db_path = data / "state.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO jobs(job_id,file_hash,file_name,src_path,out_path,status,settings,title,created,updated,total_blocks,done_blocks)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        ("old-job", "abc", "novel.epub", "/tmp/novel.epub", "/tmp/out.epub", "completed",
         '{"engine":"mock","model":"mock"}', "옛 제목", 1, 2, 10, 10),
    )
    conn.execute(
        "INSERT INTO glossary(job_id,source,target,type,locked) VALUES(?,?,?,?,1)",
        ("old-job", "塔野", "토노", "character"),
    )
    conn.execute("INSERT INTO events(job_id,ts,level,message) VALUES(?,?,?,?)", ("old-job", 1, "info", "기존 로그"))
    conn.commit()
    conn.close()
    secret = data / "secrets.json"
    secret.write_text('{"GEMINI_API_KEY":"do-not-touch"}', encoding="utf-8")
    before = secret.read_bytes()
    db = DB(str(db_path))
    cols = db._colnames("jobs")
    assert "mode" in cols and "input_kind" in cols
    assert db.q1("SELECT name FROM sqlite_master WHERE type='table' AND name='pages'")
    assert db.q1("SELECT name FROM sqlite_master WHERE type='table' AND name='regions'")
    old = db.get_job("old-job")
    assert old["title"] == "옛 제목" and old["done_blocks"] == 10
    assert old["mode"] == "epub"
    assert db.q1("SELECT target FROM glossary WHERE job_id=?", ("old-job",))["target"] == "토노"
    assert "기존 로그" in db.events("old-job")[0]["message"]
    # opening the manager must not reset the database or the secret file
    jm = JobManager(str(data))
    job = jm.create_job(samples["v3"], {"engine": "mock", "chunk_size": 800})
    assert jm.db.get_job("old-job")["title"] == "옛 제목"
    assert job["job_id"] != "old-job"
    assert secret.read_bytes() == before
    assert time.time() > 0
