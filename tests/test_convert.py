import hashlib
import os
import zipfile

import pytest

from epub_translator.epub_model import DRMError
from epub_translator.media import image_info, natural_key
from epub_translator.pipeline import JobManager
from epub_translator.qa import find_epubcheck
from epub_translator.safety import SafetyError, clamp_dpi
from tests.factories import build_image_pdf, build_mixed_pdf, build_text_pdf, huge_png_header, jpeg_bytes, png_bytes, run_job, write_cbz


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_natural_sort():
    names = ["dir/p10.png", "dir/p2.png", "dir/p1.png", "a/p1.png"]
    assert sorted(names, key=natural_key) == ["a/p1.png", "dir/p1.png", "dir/p2.png", "dir/p10.png"]


def test_dpi_bounds():
    assert clamp_dpi(300) == 300 and clamp_dpi(600) == 600
    with pytest.raises(ValueError):
        clamp_dpi(299)
    with pytest.raises(ValueError):
        clamp_dpi(601)


def test_rights_required(tmp_path):
    jpg = jpeg_bytes(40, 60, (1, 2, 3))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(build_image_pdf([(jpg, 40, 60, 300, 450)]))
    jm = JobManager(str(tmp_path / "data"))
    with pytest.raises(Exception) as ei:
        jm.create_job(str(pdf), {"engine": "mock", "mode": "pdf", "rights_ack": False})
    assert "권한" in str(ei.value)


def test_path_traversal_and_pixel_bomb(tmp_path):
    from epub_translator.cbz_convert import scan_cbz
    good = png_bytes(8, 8)
    bomb = tmp_path / "bomb.cbz"
    with zipfile.ZipFile(bomb, "w") as z:
        z.writestr("../escape.png", good)
    with pytest.raises(SafetyError):
        scan_cbz(str(bomb))
    pix = tmp_path / "pix.cbz"
    with zipfile.ZipFile(pix, "w") as z:
        z.writestr("huge.png", huge_png_header())
    with pytest.raises(SafetyError) as ei:
        scan_cbz(str(pix))
    assert "허용" in str(ei.value) or "픽셀" in str(ei.value)
    zeros = tmp_path / "ratio.cbz"
    with zipfile.ZipFile(zeros, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("pad.png", b"\x00" * 200_000)
    with pytest.raises(SafetyError):
        scan_cbz(str(zeros))


def test_pdf_jpeg_bytes_survive_and_order(tmp_path):
    jpg1 = jpeg_bytes(48, 64, (200, 10, 10))
    jpg2 = jpeg_bytes(80, 40, (10, 10, 200))  # wide spread
    jpg3 = jpeg_bytes(40, 70, (10, 180, 10))
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(build_image_pdf([
        (jpg1, 48, 64, 360, 480),
        (jpg2, 80, 40, 480, 240),
        (jpg3, 40, 70, 240, 420),
    ]))
    before = sha(pdf)
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(pdf), {"engine": "mock", "mode": "pdf", "rights_ack": True, "dpi": 300, "rtl": True})

    import asyncio
    run_job(jm, job["job_id"])
    st = jm.db.get_job(job["job_id"])
    assert sha(pdf) == before
    assert st["status"] == "completed", st["error"]
    assert st["file_hash"] == sha(jm.db.get_job(job["job_id"])["src_path"])
    z = zipfile.ZipFile(st["out_path"])
    assert z.infolist()[0].filename == "mimetype" and z.infolist()[0].compress_type == zipfile.ZIP_STORED
    opf = z.read("OEBPS/content.opf").decode()
    assert 'page-progression-direction="rtl"' in opf
    assert "pre-paginated" in opf
    assert "rendition:page-spread-center" in opf
    assert "page-spread-right" in opf or "page-spread-left" in opf
    # page 2 is the wide image and must be a centered spread, not stretched
    blob = z.read("OEBPS/images/p0001.jpg")
    assert blob == jpg1
    assert z.read("OEBPS/images/p0002.jpg") == jpg2
    info = image_info(blob)
    assert info[1:] == (48, 64)
    rep = __import__("json").load(open(os.path.join(os.path.dirname(st["out_path"]), "report.json"), encoding="utf-8"))
    assert rep["image_pipeline"]["quality"]["byte_identical_to_stored_file"] == 3
    assert rep["image_pipeline"]["quality"]["order_preserved"] is True
    assert rep["image_pipeline"]["quality"]["source_copy_matches_upload"] is True
    assert rep["structure"]["ok"], rep["structure"]
    if find_epubcheck():
        assert rep["epubcheck"]["available"] and rep["epubcheck"].get("fatal", 1) == 0 and rep["epubcheck"].get("errors", 1) == 0, rep["epubcheck"]


def test_cbz_sort_webp_png_and_ltr(tmp_path):
    members = [
        ("ch/p10.png", png_bytes(12, 20, (1, 1, 1))),
        ("ch/p2.png", png_bytes(12, 20, (2, 2, 2))),
        ("ch/p1.jpg", jpeg_bytes(12, 18, (3, 3, 3))),
    ]
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), (9, 9, 9)).save(buf, format="WEBP")
        members.append(("ch/p3.webp", buf.getvalue()))
    except Exception:
        pass
    cbz = tmp_path / "in.cbz"
    write_cbz(str(cbz), members)
    before = sha(cbz)
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(cbz), {"engine": "mock", "mode": "cbz", "rights_ack": True, "rtl": False})
    import asyncio
    run_job(jm, job["job_id"])
    st = jm.db.get_job(job["job_id"])
    assert sha(cbz) == before
    assert st["status"] == "completed", st["error"]
    z = zipfile.ZipFile(st["out_path"])
    opf = z.read("OEBPS/content.opf").decode()
    assert 'page-progression-direction="ltr"' in opf
    # natural order: p1.jpg, p2.png, p3.webp?, p10.png
    first = z.read("OEBPS/images/p0001.jpg")
    assert first == members[2][1]
    nav = z.read("OEBPS/nav.xhtml").decode()
    assert nav.find("p0001") < nav.find("p0002")


def test_render_fallback_and_missing_renderer(tmp_path, monkeypatch):
    pytest.importorskip("pymupdf")
    pdf = tmp_path / "text.pdf"
    pdf.write_bytes(build_text_pdf())
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(pdf), {"engine": "mock", "mode": "pdf", "rights_ack": True, "dpi": 300, "run_epubcheck": False})
    import asyncio
    run_job(jm, job["job_id"])
    page = jm.db.q1("SELECT * FROM pages WHERE job_id=?", (job["job_id"],))
    assert page["stage"] == "done" and page["method"] == "render", page
    assert page["width"] in range(290, 311) and page["height"] in range(290, 311)

    mixed = tmp_path / "mixed.pdf"
    jpg = jpeg_bytes(30, 40, (8, 8, 8))
    mixed.write_bytes(build_mixed_pdf(jpg, 30, 40))
    from epub_translator import pdf_convert

    def boom(*a, **k):
        from epub_translator.safety import RenderUnavailable
        raise RenderUnavailable("렌더 백엔드가 없습니다")

    monkeypatch.setattr(pdf_convert, "render_pdf_page", boom)
    jm2 = JobManager(str(tmp_path / "data2"))
    job2 = jm2.create_job(str(mixed), {"engine": "mock", "mode": "pdf", "rights_ack": True, "dpi": 300, "run_epubcheck": False})
    run_job(jm2, job2["job_id"])
    st2 = jm2.db.get_job(job2["job_id"])
    assert st2["status"] == "completed_with_issues", st2["error"]
    pages = jm2.db.q("SELECT * FROM pages WHERE job_id=? ORDER BY idx", (job2["job_id"],))
    assert pages[0]["method"] == "extract-jpeg" and pages[0]["stage"] == "done"
    assert pages[1]["stage"] == "failed" and "렌더" in (pages[1]["error"] or "")
    z = zipfile.ZipFile(st2["out_path"])
    assert z.read("OEBPS/images/p0001.jpg") == jpg
    body = z.read("OEBPS/text/p0002.xhtml").decode()
    assert "실패" in body
    # spine order keeps the failed page
    opf = z.read("OEBPS/content.opf").decode()
    assert opf.find('idref="p0001"') < opf.find('idref="p0002"')


def test_encrypted_pdf_refused(tmp_path):
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(72, 72)
    writer.encrypt("secret")
    path = tmp_path / "enc.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    jm = JobManager(str(tmp_path / "data"))
    with pytest.raises(DRMError):
        jm.create_job(str(path), {"engine": "mock", "mode": "pdf", "rights_ack": True})


def test_image_pause_cancel_resume(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setenv("EPUB_PAGE_DELAY", "0.25")
    cbz = tmp_path / "pause.cbz"
    write_cbz(str(cbz), [(f"p{i}.png", png_bytes(8, 8, (i, 0, 0))) for i in range(6)])
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(cbz), {"engine": "mock", "mode": "cbz", "rights_ack": True, "page_parallel": 1, "run_epubcheck": False})
    jid = job["job_id"]

    def progressed():
        return jm.db.q1("SELECT COUNT(*) n FROM pages WHERE job_id=? AND stage!='planned'", (jid,))["n"]

    async def go():
        task = jm.start(jid)
        await asyncio.sleep(0.12)
        jm.pause(jid)
        await asyncio.sleep(0.7)
        first = progressed()
        await asyncio.sleep(0.45)
        assert progressed() == first
        assert first < 6
        jm.cancel_job(jid)
        await task
        return first

    asyncio.run(go())
    assert jm.db.get_job(jid)["status"] == "cancelled"
    monkeypatch.setenv("EPUB_PAGE_DELAY", "0")
    run_job(jm, jid)
    st = jm.db.get_job(jid)
    assert st["status"] == "completed", st["error"]
    assert jm.db.q1("SELECT COUNT(*) n FROM pages WHERE job_id=? AND stage='done'", (jid,))["n"] == 6


def test_retry_only_failed_page(tmp_path, monkeypatch):
    import epub_translator.image_jobs as ij
    cbz = tmp_path / "p.cbz"
    write_cbz(str(cbz), [
        ("a.png", png_bytes(16, 16, (1, 0, 0))),
        ("b.png", png_bytes(16, 16, (0, 1, 0))),
        ("c.png", png_bytes(16, 16, (0, 0, 1))),
    ])
    real = ij.materialize_page
    state = {"n": 0}

    def flaky(jm, job_id, page, settings):
        if page["idx"] == 1 and state["n"] == 0:
            state["n"] += 1
            raise RuntimeError("transient page failure")
        return real(jm, job_id, page, settings)

    monkeypatch.setattr(ij, "materialize_page", flaky)
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(cbz), {"engine": "mock", "mode": "cbz", "rights_ack": True, "run_epubcheck": False})
    run_job(jm, job["job_id"])
    mid = {p["idx"]: p for p in jm.db.q("SELECT * FROM pages WHERE job_id=?", (job["job_id"],))}
    assert mid[1]["stage"] == "failed" and mid[1]["attempts"] == 1
    assert mid[0]["attempts"] == 1 and mid[2]["attempts"] == 1
    first = jm.db.get_job(job["job_id"])["out_path"]
    assert "실패" in zipfile.ZipFile(first).read("OEBPS/text/p0002.xhtml").decode()
    monkeypatch.setattr(ij, "materialize_page", real)
    run_job(jm, job["job_id"])
    done = {p["idx"]: p for p in jm.db.q("SELECT * FROM pages WHERE job_id=?", (job["job_id"],))}
    assert done[1]["stage"] == "done" and done[1]["attempts"] == 2
    assert done[0]["attempts"] == 1 and done[2]["attempts"] == 1
    st = jm.db.get_job(job["job_id"])
    assert st["status"] == "completed", st["error"]
