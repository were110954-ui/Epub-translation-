import asyncio
import hashlib
import json
import os
import zipfile

from epub_translator.image_jobs import SLOTS
from epub_translator.inpaint import restore_regions
from epub_translator.ocr import detect_balloons, direction_of, gpu_info, make_backend, order_regions
from epub_translator.pipeline import JobManager
from epub_translator.qa import find_epubcheck
from PIL import Image, ImageDraw
from tests.factories import balloon_page, png_bytes, run_job, write_cbz


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_direction_and_reading_order():
    assert direction_of(40, 100) == "vertical"
    assert direction_of(100, 40) == "horizontal"
    regions = [
        {"x": 10, "y": 10, "w": 40, "h": 20},
        {"x": 10, "y": 80, "w": 30, "h": 20},
        {"x": 80, "y": 80, "w": 30, "h": 20},
    ]
    ordered = order_regions(regions, rtl=True)
    assert [r["x"] for r in ordered] == [10, 80, 10]


def test_detector_finds_outlined_balloons(tmp_path):
    path = tmp_path / "p.png"
    balloon_page(str(path))
    im = Image.open(path)
    found = detect_balloons(im, rtl=True)
    assert len(found) == 3
    # lower row is right-to-left
    lower = [b for b in found if b["y"] > 140]
    assert lower[0]["x"] > lower[1]["x"]


def test_inpaint_removes_flat_ink_and_flags_pattern():
    im = Image.new("RGB", (80, 80), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle((30, 30, 50, 50), fill=(0, 0, 0))
    out, notes = restore_regions(im, [(10, 10, 60, 60)])
    assert notes[0]["method"] == "flat-fill" and notes[0]["confident"]
    assert out.getpixel((40, 40))[0] > 200
    assert im.getpixel((40, 40)) == (0, 0, 0)  # source image unchanged
    noisy = Image.new("RGB", (80, 80), (255, 255, 255))
    px = noisy.load()
    for i in range(80):
        px[i, 8] = ((i * 7) % 255, 10, 200)
        px[i, 71] = (30, (i * 5) % 255, 20)
        px[8, i] = (200, (i * 3) % 255, 10)
        px[71, i] = ((i * 9) % 255, 80, 40)
    out2, notes2 = restore_regions(noisy, [(5, 5, 70, 70)])
    assert notes2[0]["method"] == "flat-plate" and notes2[0]["confident"] is False


def test_comic_fixture_glossary_editor_and_secrets(tmp_path):
    page = tmp_path / "page.png"
    balloon_page(str(page))
    src_bytes = open(page, "rb").read()
    cbz = tmp_path / "comic.cbz"
    write_cbz(str(cbz), [("001.png", src_bytes)])
    before = sha(cbz)
    jm = JobManager(str(tmp_path / "data"))
    secret = "supersecret-should-not-leak"
    job = jm.create_job(str(cbz), {
        "engine": "mock", "mode": "comic", "rights_ack": True, "ocr_backend": "fixture",
        "rtl": True, "chunk_size": 30, "api_key": secret, "run_epubcheck": True,
    })
    stored = json.loads(jm.db.get_job(job["job_id"])["settings"])
    assert "api_key" not in stored and secret not in json.dumps(stored)
    jm.db.x("INSERT INTO glossary(job_id,source,target,type,locked) VALUES(?,?,?,?,1)",
            (job["job_id"], "塔野", "토노", "character",))
    run_job(jm, job["job_id"])
    st = jm.db.get_job(job["job_id"])
    assert sha(cbz) == before
    assert st["status"] in ("completed", "completed_with_issues"), st["error"]
    blob = json.dumps(jm.db.q("SELECT message FROM events WHERE job_id=?", (job["job_id"],)), ensure_ascii=False)
    assert secret not in blob
    assert secret not in (st.get("error") or "")
    report = json.load(open(os.path.join(os.path.dirname(st["out_path"]), "report.json"), encoding="utf-8"))
    assert secret not in json.dumps(report)
    assert report["image_pipeline"]["ocr_real"] is False
    assert "fixture" in report["image_pipeline"]["ocr_backend"]
    assert report["image_pipeline"]["quality"]["source_copy_matches_upload"] is True
    assert any("휴리스틱" in line for line in report["image_pipeline"]["limits"])
    regions = jm.db.q("SELECT * FROM regions WHERE job_id=? ORDER BY idx", (job["job_id"],))
    assert len(regions) == 3
    assert "detection-count-mismatch" not in " ".join(regions[0]["flags"] or "")
    # RTL: the right-hand lower balloon (塔野) comes before the left one (本当)
    texts = [r["ocr_text"] for r in regions]
    assert texts[0].startswith("「こんにちは")
    assert "塔野" in texts[1] and "本当" in texts[2]
    low = [r for r in regions if r["review"]]
    assert any("本当" in (r["ocr_text"] or "") for r in low)
    translated = " ".join(r["translation"] or "" for r in regions)
    assert "토노" in translated
    assert not any("\u3041" <= ch <= "\u30ff" for ch in translated)
    z = zipfile.ZipFile(st["out_path"])
    page = z.read("OEBPS/text/p0001.xhtml").decode()
    assert "토노" in page and "keep-all" in z.read("OEBPS/styles/fixed.css").decode()
    orig = jm.db.q1("SELECT sha256 FROM pages WHERE job_id=?", (job["job_id"],))["sha256"]
    assert orig == hashlib.sha256(src_bytes).hexdigest()
    # editor: move and rewrite one balloon, rebuild without another full OCR
    calls = st["api_calls"]
    target = regions[0]["region_id"]
    from epub_translator.image_jobs import rebuild_edited, update_region
    update_region(jm, job["job_id"], target, {"translation": "수정된 대사입니다", "x": 30, "y": 40, "font": "serif", "font_size": 18})
    asyncio.run(rebuild_edited(jm, job["job_id"]))
    st2 = jm.db.get_job(job["job_id"])
    assert st2["api_calls"] == calls
    page2 = zipfile.ZipFile(st2["out_path"]).read("OEBPS/text/p0001.xhtml").decode()
    assert "수정된 대사입니다" in page2
    assert "left:30px" in page2 and "font-size:18" in page2
    # original extract still byte-identical after rebuild
    orig_path = os.path.join(os.path.dirname(st2["out_path"]), "orig", "p0001.png")
    assert sha(orig_path) == hashlib.sha256(src_bytes).hexdigest()
    if find_epubcheck():
        rep2 = json.load(open(os.path.join(os.path.dirname(st2["out_path"]), "report.json"), encoding="utf-8"))
        assert rep2["epubcheck"]["available"] and rep2["epubcheck"].get("fatal", 1) == 0 and rep2["epubcheck"].get("errors", 1) == 0, rep2["epubcheck"]


def test_retranslate_one_region(tmp_path):
    page = tmp_path / "page.png"
    balloon_page(str(page))
    cbz = tmp_path / "c.cbz"
    write_cbz(str(cbz), [("001.png", open(page, "rb").read())])
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(cbz), {"engine": "mock", "mode": "comic", "rights_ack": True, "ocr_backend": "fixture", "chunk_size": 20})
    run_job(jm, job["job_id"])
    calls = jm.db.get_job(job["job_id"])["api_calls"]
    region = jm.db.q("SELECT * FROM regions WHERE job_id=? ORDER BY idx", (job["job_id"],))[0]
    from epub_translator.image_jobs import retranslate_region, update_region
    update_region(jm, job["job_id"], region["region_id"], {"ocr_text": "「明日も晴れです。」"})
    asyncio.run(retranslate_region(jm, job["job_id"], region["region_id"]))
    st = jm.db.get_job(job["job_id"])
    assert st["api_calls"] == calls + 1
    row = jm.db.q1("SELECT * FROM regions WHERE job_id=? AND region_id=?", (job["job_id"], region["region_id"]))
    assert row["translation"] and "明日" not in row["translation"]


def test_gpu_autodetect_is_safe():
    info = gpu_info()
    assert info["selected"] in ("cpu", "cuda")
    assert "note" in info


def test_tesseract_real_horizontal_and_vertical():
    backend = make_backend("tesseract", {"selected": "cpu", "providers": [], "note": ""})
    assert backend.real is True and backend.id == "tesseract"
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if not os.path.exists(font_path):
        import pytest
        pytest.skip("no CJK font")
    from PIL import ImageFont
    font = ImageFont.truetype(font_path, 64)
    im = Image.new("RGB", (640, 180), "white")
    ImageDraw.Draw(im).text((20, 40), "こんにちは", font=font, fill="black")
    text, conf, flags = backend.recognize(im, "horizontal")
    assert text == "こんにちは", (text, conf, flags)
    assert conf > 0.5 and not flags
    tall = Image.new("RGB", (140, 460), "white")
    draw = ImageDraw.Draw(tall)
    y = 20
    font2 = ImageFont.truetype(font_path, 52)
    for ch in "こんにちは":
        draw.text((36, y), ch, font=font2, fill="black")
        y += 80
    vtext, vconf, vflags = backend.recognize(tall, "vertical")
    assert vtext == "こんにちは", (vtext, vconf, vflags)


def test_parallel_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("EPUB_PAGE_DELAY", "0.05")
    members = [(f"p{i}.png", png_bytes(8, 8, (i, 0, 0))) for i in range(8)]
    cbz = tmp_path / "p.cbz"
    write_cbz(str(cbz), members)
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(str(cbz), {"engine": "mock", "mode": "cbz", "rights_ack": True, "page_parallel": 2, "run_epubcheck": False})
    run_job(jm, job["job_id"])
    assert SLOTS.peak == 2
    assert jm.db.get_job(job["job_id"])["status"] == "completed"
