import time

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
        denied = c.post("/api/jobs", files={"file": ("a.pdf", pdf.read_bytes(), "application/pdf")},
                        data={"engine": "mock", "mode": "pdf", "rights_ack": "false"})
        assert denied.status_code == 422
        ok = c.post("/api/jobs", files={"file": ("a.pdf", pdf.read_bytes(), "application/pdf")},
                    data={"engine": "mock", "mode": "pdf", "rights_ack": "true", "autostart": "true", "dpi": "300"})
        assert ok.status_code == 200, ok.text
        jid = ok.json()["job_id"]
        assert ok.json()["estimate"]["cost_usd"] == 0
        for _ in range(400):
            s = c.get(f"/api/jobs/{jid}").json()
            if str(s["status"]).startswith("completed") or s["status"] == "error":
                break
            time.sleep(0.05)
        assert s["status"] == "completed", s.get("error")
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


def test_cli_pdf_without_rights(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(build_image_pdf([(jpeg_bytes(8, 8), 8, 8, 72, 72)]))
    rc = main(["translate", str(pdf), "--engine", "mock", "--mode", "pdf", "--data-dir", str(tmp_path / "d")])
    assert rc == 2
