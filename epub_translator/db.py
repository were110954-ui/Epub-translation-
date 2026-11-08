"""SQLite persistence: jobs, blocks (per-block status), chunk cache, glossary, characters, summaries, events."""
from __future__ import annotations
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  job_id TEXT PRIMARY KEY, file_hash TEXT, file_name TEXT, src_path TEXT, out_path TEXT,
  status TEXT, settings TEXT, title TEXT, created REAL, updated REAL,
  total_blocks INTEGER DEFAULT 0, done_blocks INTEGER DEFAULT 0, failed_blocks INTEGER DEFAULT 0,
  current_chapter TEXT, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
  api_calls INTEGER DEFAULT 0, retries INTEGER DEFAULT 0, error TEXT, report_json TEXT, report_html TEXT);
CREATE TABLE IF NOT EXISTS blocks(
  job_id TEXT, block_id TEXT, idx INTEGER, file TEXT, chapter TEXT, kind TEXT, xpath TEXT,
  source TEXT, target TEXT, status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0,
  chunk_id TEXT, furigana TEXT, flags TEXT DEFAULT '[]', error TEXT, updated REAL,
  PRIMARY KEY(job_id, block_id));
CREATE INDEX IF NOT EXISTS ix_blocks_status ON blocks(job_id, status);
CREATE TABLE IF NOT EXISTS chunks(
  job_id TEXT, chunk_id TEXT, seq INTEGER, chapter TEXT, file TEXT, block_ids TEXT,
  status TEXT DEFAULT 'pending', summary TEXT, attempts INTEGER DEFAULT 0, error TEXT,
  PRIMARY KEY(job_id, chunk_id));
CREATE TABLE IF NOT EXISTS cache(
  key TEXT PRIMARY KEY, response TEXT, created REAL);
CREATE TABLE IF NOT EXISTS glossary(
  job_id TEXT, source TEXT, target TEXT, type TEXT, reading TEXT, note TEXT, locked INTEGER DEFAULT 0,
  PRIMARY KEY(job_id, source));
CREATE TABLE IF NOT EXISTS characters(
  job_id TEXT, name TEXT, korean TEXT, speech TEXT, honorifics TEXT, note TEXT,
  PRIMARY KEY(job_id, name));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts REAL, level TEXT, message TEXT);
"""


class DB:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _colnames(self, table: str) -> set[str]:
        return {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self):
        """Additive only. Existing rows, glossary, and settings JSON are left in place."""
        cols = self._colnames("jobs")
        if "mode" not in cols:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN mode TEXT DEFAULT 'epub'")
        if "input_kind" not in cols:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN input_kind TEXT DEFAULT 'epub'")
        self.conn.executescript("""
CREATE TABLE IF NOT EXISTS schema_migrations(id TEXT PRIMARY KEY, applied REAL);
CREATE TABLE IF NOT EXISTS pages(
  job_id TEXT NOT NULL, page_id TEXT NOT NULL, idx INTEGER NOT NULL, src_name TEXT,
  width INTEGER DEFAULT 0, height INTEGER DEFAULT 0, media_type TEXT, method TEXT,
  sha256 TEXT, byte_size INTEGER DEFAULT 0, spread INTEGER DEFAULT 0,
  stage TEXT DEFAULT 'planned', attempts INTEGER DEFAULT 0, error TEXT, updated REAL,
  PRIMARY KEY(job_id, page_id));
CREATE INDEX IF NOT EXISTS ix_pages_job ON pages(job_id, idx);
CREATE TABLE IF NOT EXISTS regions(
  job_id TEXT NOT NULL, region_id TEXT NOT NULL, page_id TEXT NOT NULL, idx INTEGER NOT NULL,
  x INTEGER, y INTEGER, w INTEGER, h INTEGER, direction TEXT, ocr_text TEXT, translation TEXT,
  confidence REAL, font TEXT, font_size REAL, status TEXT DEFAULT 'pending', flags TEXT DEFAULT '[]',
  review INTEGER DEFAULT 0, edited INTEGER DEFAULT 0, updated REAL,
  PRIMARY KEY(job_id, region_id));
CREATE INDEX IF NOT EXISTS ix_regions_page ON regions(job_id, page_id, idx);
""")
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(id, applied) VALUES('2026-11-image-modes', ?)",
            (time.time(),),
        )
        region_cols = self._colnames("regions")
        if region_cols and "group_id" not in region_cols:
            self.conn.execute("ALTER TABLE regions ADD COLUMN group_id TEXT")
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(id, applied) VALUES('2026-11-line-groups', ?)",
                (time.time(),),
            )
            self.conn.commit()
        page_cols = self._colnames("pages")
        if page_cols and "hq_json" not in page_cols:
            self.conn.execute("ALTER TABLE pages ADD COLUMN hq_json TEXT")
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(id, applied) VALUES('2026-11-hq-review', ?)",
                (time.time(),),
            )
            self.conn.commit()

    def x(self, sql, args=(), many=False):
        with self.lock:
            cur = self.conn.executemany(sql, args) if many else self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def q1(self, sql, args=()):
        r = self.q(sql, args)
        return r[0] if r else None

    # jobs
    def get_job(self, job_id):
        return self.q1("SELECT * FROM jobs WHERE job_id=?", (job_id,))

    def update_job(self, job_id, **kw):
        kw["updated"] = time.time()
        cols = ",".join(f"{k}=?" for k in kw)
        self.x(f"UPDATE jobs SET {cols} WHERE job_id=?", (*kw.values(), job_id))

    def add_usage(self, job_id, p, c, calls=1, retries=0):
        self.x("UPDATE jobs SET prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+?, "
               "api_calls=api_calls+?, retries=retries+? WHERE job_id=?", (p, c, calls, retries, job_id))

    def event(self, job_id, level, msg):
        self.x("INSERT INTO events(job_id,ts,level,message) VALUES(?,?,?,?)", (job_id, time.time(), level, msg))

    def events(self, job_id, limit=50):
        return self.q("SELECT * FROM events WHERE job_id=? ORDER BY id DESC LIMIT ?", (job_id, limit))

    # cache
    def cache_get(self, key):
        r = self.q1("SELECT response FROM cache WHERE key=?", (key,))
        return json.loads(r["response"]) if r else None

    def cache_put(self, key, value):
        self.x("INSERT OR REPLACE INTO cache(key,response,created) VALUES(?,?,?)",
               (key, json.dumps(value, ensure_ascii=False), time.time()))

    # counters
    def refresh_counts(self, job_id):
        r = self.q1("SELECT COUNT(*) t, SUM(status IN ('done','warn')) d, SUM(status='failed') f FROM blocks WHERE job_id=?", (job_id,))
        self.update_job(job_id, total_blocks=r["t"] or 0, done_blocks=r["d"] or 0, failed_blocks=r["f"] or 0)
        return r
