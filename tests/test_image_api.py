import time

import pytest
from fastapi.testclient import TestClient

from epub_translator.cli import main
from tests.factories import balloon_page, jpeg_bytes, build_image_pdf, write_cbz


def test_web_pdf_requires_rights_and_comic_editor(tmp_path, monkeypatch):
    monkeypatch.setenv("EPUB_TRANSLATOR_DATA", str(tmp_path))
    import epub_translator.server as srv
    srv.JM = None
    jpg = jpeg_bytes(32, 40, (4, 5, 6))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(build_image_pdf([(jpg, 32, 40, 240, 300)]))
    page = tmp_path / "c.png"
    balloon_page(str(page))
    cbz = tmp_path / "c.cbz"
    write_cbz(str(cbz), [("001.png", open(page, "rb").read())])
    with TestClient(srv.app) as c:
        html = c.get("/").text
        assert "EPUB AI Translator" in html and "적법한" in html and "만화" in html
        assert "번역 없이 EPUB으로만 바꾸는 기능은 없습니다" in html
        assert "PDF → EPUB3" not in html
        before = len(c.get("/api/jobs").json())
        denied = c.post("/api/jobs", files={"file": ("a.pdf", pdf.read_bytes(), "application/pdf")},
                        data={"engine": "mock", "mode": "pdf", "rights_ack": "true"})
        assert denied.status_code == 400 and "변환" in denied.text
        auto = c.post("/api/jobs", files={"file": ("a.pdf", pdf.read_bytes(), "application/pdf")},
                      data={"engine": "mock", "mode": "auto", "rights_ack": "true"})
        assert auto.status_code == 400 and "변환" in auto.text
        assert len(c.get("/api/jobs").json()) == before
        ok = c.post("/api/jobs", files={"file": ("a.pdf", pdf.read_bytes(), "application/pdf")},
                    data={"engine": "mock", "mode": "comic", "ocr_backend": "fixture", "rights_ack": "true",
                          "autostart": "false", "dpi": "300"})
        assert ok.status_code == 200, ok.text
        jid = ok.json()["job_id"]
        assert c.get(f"/api/jobs/{jid}").json()["api_calls"] == 0
        c.post(f"/api/jobs/{jid}/start")
        for _ in range(400):
            s = c.get(f"/api/jobs/{jid}").json()
            if str(s["status"]).startswith("completed") or s["status"] == "error":
                break
            time.sleep(0.05)
        assert str(s["status"]).startswith("completed"), s.get("error")
        assert s["api_calls"] == 0
        assert s["page_progress"]["total"] == 1
        body = c.get(f"/api/jobs/{jid}/download").content
        assert body[:2] == b"PK" and jpg in body
        comic = c.post("/api/jobs", files={"file": ("c.cbz", cbz.read_bytes(), "application/zip")},
                       data={"engine": "mock", "mode": "comic", "rights_ack": "true", "ocr_backend": "fixture",
                             "autostart": "true", "chunk_size": "40"})
        assert comic.status_code == 200, comic.text
        cid = comic.json()["job_id"]
        for _ in range(400):
            s = c.get(f"/api/jobs/{cid}").json()
            if str(s["status"]).startswith("completed") or s["status"] == "error":
                break
            time.sleep(0.05)
        assert str(s["status"]).startswith("completed"), s.get("error")
        ed = c.get(f"/api/jobs/{cid}/editor").json()
        assert len(ed["regions"]) == 3
        rid = ed["regions"][0]["region_id"]
        saved = c.post(f"/api/jobs/{cid}/regions/{rid}", data={
            "translation": "편집기에서 고친 문장", "ocr_text": ed["regions"][0]["ocr_text"],
            "font": "gothic", "font_size": "20", "x": "12", "y": "18", "w": "80", "h": "40",
        })
        assert saved.status_code == 200, saved.text
        rebuilt = c.post(f"/api/jobs/{cid}/rebuild")
        assert rebuilt.status_code == 200, rebuilt.text
        epub = c.get(f"/api/jobs/{cid}/download").content
        assert "편집기에서 고친 문장".encode() in epub
        img = c.get(f"/api/jobs/{cid}/pages/p0001/image?variant=original")
        assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_translate_src_lang_en(tmp_path, samples):
    rc = main(["translate", samples["v3"], "--engine", "mock", "--src-lang", "en",
               "--data-dir", str(tmp_path), "--no-epubcheck", "--chunk-size", "800"])
    assert rc == 0


def test_historical_pdf_conversion_download_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("EPUB_TRANSLATOR_DATA", str(tmp_path))
    import epub_translator.server as srv
    srv.JM = None
    from epub_translator.safety import SafetyError
    with TestClient(srv.app) as c:
        jm = srv.jm()
        out = tmp_path / "jobs" / "old-pdf" / "kept.fxl.epub"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"PK\x03\x04historical-pdf")
        jm.db.x(
            "INSERT INTO jobs(job_id,file_hash,file_name,src_path,out_path,status,settings,title,created,updated,total_blocks,done_blocks,mode,input_kind)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("old-pdf", "abc", "scan.pdf", str(tmp_path / "scan.pdf"), str(out), "completed",
             '{"mode":"pdf","engine":"mock","model":"mock","rights_ack":true}', "옛 PDF 변환", 1, 2, 3, 3, "pdf", "pdf"),
        )
        with pytest.raises(SafetyError):
            jm.start("old-pdf")
        row = jm.db.get_job("old-pdf")
        assert row["status"] == "completed" and row["title"] == "옛 PDF 변환"
        assert open(row["out_path"], "rb").read() == b"PK\x03\x04historical-pdf"
        refused = c.post("/api/jobs/old-pdf/resume")
        assert refused.status_code == 400 and "변환" in refused.text
        assert jm.db.get_job("old-pdf")["status"] == "completed"
        got = c.get("/api/jobs/old-pdf/download")
        assert got.status_code == 200 and got.content == b"PK\x03\x04historical-pdf"


def test_cli_pdf_without_rights(tmp_path, capsys):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(build_image_pdf([(jpeg_bytes(8, 8), 8, 8, 72, 72)]))
    rc = main(["translate", str(pdf), "--engine", "mock", "--rights-ack", "--data-dir", str(tmp_path / "d")])
    assert rc == 2
    assert "변환" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["translate", str(pdf), "--mode", "pdf", "--engine", "mock", "--data-dir", str(tmp_path / "d2")])
