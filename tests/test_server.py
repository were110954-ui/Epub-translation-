import time

from fastapi.testclient import TestClient


def test_web_flow(tmp_path, samples, monkeypatch):
    monkeypatch.setenv("EPUB_TRANSLATOR_DATA", str(tmp_path))
    import epub_translator.server as srv
    srv.JM = None
    with TestClient(srv.app) as c:
        assert "EPUB AI Translator" in c.get("/").text
        with open(samples["v3"], "rb") as f:
            r = c.post("/api/jobs", files={"file": ("s3.epub", f, "application/epub+zip")},
                       data={"engine": "mock", "chunk_size": "400", "autostart": "true"})
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
        assert r.json()["estimate"]["blocks"] > 20
        for _ in range(200):
            s = c.get(f"/api/jobs/{jid}").json()
            if s["status"].startswith("completed"):
                break
            time.sleep(0.05)
        assert s["status"] == "completed" and s["progress"] == 100
        d = c.get(f"/api/jobs/{jid}/download")
        assert d.status_code == 200 and d.content[:2] == b"PK"
        assert c.get(f"/api/jobs/{jid}/report.html").status_code == 200
        assert c.get(f"/api/jobs/{jid}/glossary").status_code == 200
        bad = c.post("/api/jobs", files={"file": ("x.epub", b"not a zip", "application/epub+zip")}, data={"engine": "mock"})
        assert bad.status_code == 400
