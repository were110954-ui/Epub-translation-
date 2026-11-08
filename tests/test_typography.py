"""Korean novel typography defaults. These do not call a paid translation API."""
import os
import zipfile

from lxml import etree

from epub_translator.fxl import write_fxl_epub
from epub_translator.pipeline import JobManager
from epub_translator.qa import find_epubcheck, run_epubcheck
from epub_translator.rebuild import KO_CSS, KO_FONT, KO_LINE, is_fixed_layout, korean_css, normalize_korean_document, restyle_epub
from tests.factories import png_bytes, run_job


XHTML = "http://www.w3.org/1999/xhtml"


def test_legacy_english_css_becomes_korean_novel_metrics():
    src = """
    p.body { font-family: "Hiragino Sans", "Malgun Gothic", sans-serif; line-height: 1.9;
      margin-bottom: 1.4em; letter-spacing: 0.04em; font-size: 17px; font-weight: bold; text-align: justify; }
    h1.title { font-family: Arial, sans-serif; font-size: 22px; font-weight: bold; line-height: 1.2; }
    em.em { font-style: italic; }
    p::first-letter { font-weight: bold; font-size: 2.4em; float: left; }
    """
    out = korean_css(src)
    assert "Hiragino" not in out and "Malgun Gothic" not in out and "sans-serif" not in out
    assert "Nanum Myeongjo" in out and "1.65" in out and "17px" not in out and "22px" not in out
    assert "letter-spacing: normal" in out and "justify" not in out
    assert "font-size: 2.4em" not in out and "float: none" in out
    assert "font-size: 1.4em" in out and "font-weight: bold" in out
    assert "font-style: italic" in out
    assert "!important" not in out
    assert KO_LINE == "1.65"
    assert "Apple SD Gothic Neo" not in KO_FONT and "serif" in KO_FONT
    assert "font-size: 100%" in KO_CSS and "font-size:" in KO_CSS and "px;" not in KO_CSS


def test_dropcap_removed_emphasis_and_scene_kept():
    root = etree.fromstring(f"""<html xmlns="{XHTML}" xml:lang="ko">
      <body>
        <h1>아침 골목</h1>
        <p>　<span class="dropcap" style="font-size:2.2em; font-weight:bold; float:left">골</span>목 끝에서 비가 잦아들었다. <strong>젖은 돌</strong>이 <em>조용히</em> 빛났다.</p>
        <p class="center">◇</p>
        <div class="illus"><img src="a.png" alt=""/></div>
      </body></html>""")
    normalize_korean_document(root)
    p = root.find(f".//{{{XHTML}}}p")
    assert "dropcap" not in etree.tostring(p, encoding="unicode")
    assert "float" not in etree.tostring(p, encoding="unicode")
    assert p.text.startswith("골목")
    assert p.find(f"{{{XHTML}}}strong") is not None
    assert p.find(f"{{{XHTML}}}em") is not None
    assert root.find(f".//{{{XHTML}}}h1") is not None
    scene = root.findall(f".//{{{XHTML}}}p")[1]
    assert "kt-scene" in (scene.get("class") or "")
    assert root.find(f".//{{{XHTML}}}img") is not None


def _novel(path, lang="ko"):
    png = png_bytes(8, 8, (20, 20, 20))
    css = ('p.body{font-family:"Hiragino Sans",sans-serif;line-height:1.9;margin-bottom:1.2em;font-size:16px}'
           'h1{font-size:1.5em;font-weight:bold} .illus img{max-width:100%}')
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML}" xml:lang="{lang}">
<head><title>샘플</title><link rel="stylesheet" type="text/css" href="style.css"/></head>
<body>
<h1>아침 골목</h1>
<p class="body">　<span style="font-size:2em;font-weight:bold">골</span>목 끝에서 비가 잦아들었다. <strong>젖은 돌</strong>이 빛났다.</p>
<p class="body">창가의 고양이가 꼬리를 한 번 흔들었다.</p>
<p class="center">◇</p>
<div class="illus"><img src="pic.png" alt=""/></div>
</body></html>"""
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="3.0" xml:lang="{lang}">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="uid">urn:uuid:6f1c7a52-6a1e-4b3e-9d55-0a1b2c3d4e71</dc:identifier>
<dc:title>아침 골목</dc:title>
<dc:language>{lang}</dc:language>
<meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
</metadata>
<manifest>
<item id="css" href="style.css" media-type="text/css"/>
<item id="c1" href="chap.xhtml" media-type="application/xhtml+xml"/>
<item id="img" href="pic.png" media-type="image/png"/>
<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
</manifest>
<spine><itemref idref="c1"/></spine>
</package>"""
    nav = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML}" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}">
<head><title>목차</title></head>
<body><nav epub:type="toc" id="toc"><ol><li><a href="chap.xhtml">아침 골목</a></li></ol></nav></body></html>"""
    with zipfile.ZipFile(path, "w") as z:
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, b"application/epub+zip")
        z.writestr("META-INF/container.xml", """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>""")
        z.writestr("content.opf", opf)
        z.writestr("style.css", css)
        z.writestr("chap.xhtml", body)
        z.writestr("nav.xhtml", nav)
        z.writestr("pic.png", png)


def test_restyle_cascade_language_and_epubcheck(tmp_path):
    src = tmp_path / "in.epub"
    out = tmp_path / "out.epub"
    _novel(src, "ko")
    before = src.read_bytes()
    restyle_epub(str(src), str(out))
    assert src.read_bytes() == before
    z = zipfile.ZipFile(out)
    css = z.read("style.css").decode()
    assert "Hiragino" not in css and "Nanum Myeongjo" in css and "16px" not in css
    assert "margin-bottom: 0" in css or "margin: 0" in css
    assert "line-height: 1.65" in css
    extra = z.read("kt-ko-readability.css").decode()
    assert "1.65" in extra and "text-indent: 1em" in extra and " !important" not in extra
    chap = z.read("chap.xhtml").decode()
    assert chap.rfind("kt-ko-readability.css") > chap.rfind("style.css")
    assert "font-size:2em" not in chap and "font-weight:bold" not in chap.replace("strong", "")
    assert "<strong>젖은 돌</strong>" in chap
    assert 'xml:lang="ko"' in chap
    opf = z.read("content.opf").decode()
    assert "<dc:language>ko</dc:language>" in opf
    assert "kt-ko-readability.css" in opf
    assert z.read("pic.png")[:8] == b"\x89PNG\r\n\x1a\n"
    # heading size stays relative
    assert "font-size: 1.5em" in css and "font-weight: bold" in css
    if find_epubcheck():
        report = run_epubcheck(str(out))
        assert report["available"] and report.get("fatal") == 0 and report.get("errors") == 0, report


def test_fixed_layout_is_not_restyled(tmp_path):
    img = tmp_path / "p.png"
    img.write_bytes(png_bytes(12, 16, (1, 2, 3)))
    epub = tmp_path / "comic.epub"
    write_fxl_epub(str(epub), [{
        "page_id": "p0001", "idx": 0, "width": 12, "height": 16, "image_path": str(img),
        "media_type": "image/png", "overlays": [], "title": "1",
    }], {"title": "comic", "language": "ja", "rtl": True, "uid": "6f1c7a52-6a1e-4b3e-9d55-0a1b2c3d4e72"})
    from epub_translator.epub_model import EpubBook
    assert is_fixed_layout(EpubBook(str(epub)))
    try:
        restyle_epub(str(epub), str(tmp_path / "no.epub"))
        raise AssertionError("fixed layout must be refused")
    except ValueError as e:
        assert "고정 레이아웃" in str(e)
    assert not (tmp_path / "no.epub").exists()


def test_cached_novel_rebuild_applies_defaults_without_api(tmp_path, samples):
    jm = JobManager(str(tmp_path / "data"))
    job = jm.create_job(samples["v3"], {"engine": "mock", "chunk_size": 400, "run_epubcheck": False})
    run_job(jm, job["job_id"])
    calls = jm.db.get_job(job["job_id"])["api_calls"]
    import asyncio
    asyncio.run(jm.finalize(job["job_id"]))
    again = jm.db.get_job(job["job_id"])
    assert again["api_calls"] == calls
    css = zipfile.ZipFile(again["out_path"]).read("OEBPS/kt-ko-readability.css").decode()
    assert "Nanum Myeongjo" in css and "1.65" in css and "text-indent: 1em" in css
    assert "Malgun Gothic" not in css and " !important" not in css
    chap = zipfile.ZipFile(again["out_path"]).read("OEBPS/text/chapter1.xhtml").decode()
    assert "xml:lang" in chap and "ko" in chap
    assert "<em>" in chap or "<strong>" in chap
    assert 'src="../images/illust1.png"' in chap
