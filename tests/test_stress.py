"""Bounded 300-page conversion. Images stay on disk; parallelism stays capped."""
import os

from epub_translator.image_jobs import SLOTS
from epub_translator.pipeline import JobManager
from epub_translator.qa import find_epubcheck, structure_check
from tests.factories import build_image_pdf, jpeg_bytes, png_bytes, run_job, write_cbz


def test_three_hundred_cbz_pages(tmp_path):
    SLOTS.reset()
    members = []
    for i in range(300):
        # tiny distinct PNGs; natural names are zero-padded so sort matches index
        members.append((f"p{i:04d}.png", png_bytes(8, 12, (i % 255, (i * 3) % 255, 40))))
    cbz = tmp_path / "long.cbz"
    write_cbz(str(cbz), members)
    before = open(cbz, "rb").read()
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(cbz), {
        "engine": "mock", "mode": "cbz", "rights_ack": True, "page_parallel": 2, "run_epubcheck": bool(find_epubcheck()),
    })
    run_job(jm, job["job_id"])
    st = jm.db.get_job(job["job_id"])
    assert st["status"] == "completed", st["error"]
    assert SLOTS.peak <= 2
    pages = jm.db.q("SELECT page_id, stage FROM pages WHERE job_id=? ORDER BY idx", (job["job_id"],))
    assert len(pages) == 300 and all(p["stage"] == "done" for p in pages)
    import zipfile
    z = zipfile.ZipFile(st["out_path"])
    assert sum(1 for n in z.namelist() if n.startswith("OEBPS/text/")) == 300
    assert z.read("OEBPS/images/p0001.png") == members[0][1]
    assert z.read("OEBPS/images/p0300.png") == members[299][1]
    struct = structure_check(st["out_path"])
    assert struct["ok"], struct
    assert open(cbz, "rb").read() == before
    if find_epubcheck():
        import json
        rep = json.load(open(os.path.join(os.path.dirname(st["out_path"]), "report.json"), encoding="utf-8"))
        assert rep["epubcheck"]["available"] and rep["epubcheck"].get("fatal") == 0 and rep["epubcheck"].get("errors") == 0, rep["epubcheck"]


def test_three_hundred_pdf_embedded_jpegs(tmp_path):
    """Same bound for PDF extraction. Uses tiny JPEGs so the test is about page count, not pixels."""
    SLOTS.reset()
    pages = []
    raws = []
    for i in range(300):
        raw = jpeg_bytes(8, 10, ((i * 5) % 255, 20, 30), quality=40)
        raws.append(raw)
        pages.append((raw, 8, 10, 72, 90))
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(build_image_pdf(pages))
    before = pdf.read_bytes()
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(pdf), {
        "engine": "mock", "mode": "pdf", "rights_ack": True, "dpi": 300, "page_parallel": 2,
        "run_epubcheck": False,
    })
    run_job(jm, job["job_id"])
    st = jm.db.get_job(job["job_id"])
    assert st["status"] == "completed", st["error"]
    assert SLOTS.peak <= 2
    assert jm.db.q1("SELECT COUNT(*) n FROM pages WHERE job_id=? AND stage='done'", (job["job_id"],))["n"] == 300
    import zipfile
    z = zipfile.ZipFile(st["out_path"])
    assert z.read("OEBPS/images/p0001.jpg") == raws[0]
    assert z.read("OEBPS/images/p0300.jpg") == raws[299]
    assert pdf.read_bytes() == before
