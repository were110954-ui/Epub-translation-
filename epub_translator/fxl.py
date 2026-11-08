"""EPUB 3 fixed-layout writer.

Images are stored (not deflated) so the bytes we put in are the bytes we wrote to disk.
Page order is the order of `pages`. Reading direction is metadata, not a reversal of that order.
"""
from __future__ import annotations
import hashlib
import os
import zipfile
from datetime import datetime, timezone
from uuid import uuid4

from .media import ext_for

XHTML = "http://www.w3.org/1999/xhtml"
OPF = "http://www.idpf.org/2007/opf"
DC = "http://purl.org/dc/elements/1.1/"

FONTS = {
    "gothic": "'Malgun Gothic','Noto Sans CJK KR','Nanum Gothic',sans-serif",
    "serif": "'Noto Serif CJK KR','Nanum Myeongjo','Batang',serif",
    "sans": "'Apple SD Gothic Neo','Malgun Gothic',sans-serif",
}

CSS = """html,body{margin:0;padding:0;background:#fff}
.page{position:relative;overflow:hidden;margin:0;padding:0}
.page img.bg{position:absolute;left:0;top:0}
.balloon{position:absolute;margin:0;padding:2px;box-sizing:border-box;display:flex;align-items:center;
 justify-content:center;text-align:center;line-height:1.25;color:#111;overflow:hidden;
 word-break:keep-all;overflow-wrap:break-word;line-break:strict}
.balloon.v{writing-mode:vertical-rl;text-orientation:mixed}
.fail{position:absolute;left:24px;right:24px;top:40%;font-family:sans-serif;font-size:18px;color:#7f1d1d}
"""


def _xe(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def spread_properties(pages: list[dict], rtl: bool) -> None:
    """Annotate each page with itemref `properties`. Cover and wide images are centered spreads."""
    side_right = True
    for i, page in enumerate(pages):
        wide = False
        w, h = int(page.get("width") or 0), int(page.get("height") or 0)
        if w and h and (w / h) >= 1.35:
            wide = True
        page["spread"] = 1 if wide else 0
        if i == 0 or wide or page.get("placeholder"):
            page["prop"] = "rendition:page-spread-center"
            continue
        if rtl:
            page["prop"] = "page-spread-right" if side_right else "page-spread-left"
        else:
            page["prop"] = "page-spread-left" if side_right else "page-spread-right"
        side_right = not side_right


def _font_css(key: str | None) -> str:
    return FONTS.get(key or "gothic", FONTS["gothic"])


def _page_xhtml(page: dict, image_href: str | None) -> str:
    w, h = int(page.get("width") or 600), int(page.get("height") or 800)
    title = _xe(page.get("title") or page.get("page_id") or "")
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<html xmlns="{XHTML}" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{_xe(page.get("lang") or "ko")}" lang="{_xe(page.get("lang") or "ko")}">',
        "<head>",
        '<meta charset="utf-8"/>',
        f"<title>{title}</title>",
        f'<meta name="viewport" content="width={w}, height={h}"/>',
        '<link rel="stylesheet" type="text/css" href="../styles/fixed.css"/>',
        "</head>",
        "<body>",
        f'<div class="page" style="width:{w}px;height:{h}px">',
    ]
    if image_href:
        parts.append(f'<img class="bg" src="{_xe(image_href)}" alt="" width="{w}" height="{h}"/>')
    else:
        message = page.get("error") or "이 페이지는 변환에 실패했습니다."
        if "실패" not in message:
            message = "이 페이지는 변환에 실패했습니다. " + message
        parts.append(f'<p class="fail">{_xe(message)}</p>')
    for region in page.get("overlays") or []:
        if not (region.get("text") or "").strip():
            continue
        cls = "balloon v" if region.get("direction") == "vertical" else "balloon"
        style = (
            f'left:{int(region["x"])}px;top:{int(region["y"])}px;width:{int(region["w"])}px;height:{int(region["h"])}px;'
            f'font-size:{float(region.get("font_size") or 16):.1f}px;font-family:{_font_css(region.get("font"))}'
        )
        parts.append(f'<div class="{cls}" style="{style}">{_xe(region["text"])}</div>')
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _nav(pages: list[dict]) -> str:
    items = "\n".join(
        f'<li><a href="text/{_xe(p["page_id"])}.xhtml">{_xe(p.get("title") or str(p["idx"] + 1))}</a></li>'
        for p in pages
    )
    return f'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML}" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="ko" lang="ko">
<head><meta charset="utf-8"/><title>목차</title></head>
<body>
<nav epub:type="toc" id="toc"><h1>목차</h1><ol>
{items}
</ol></nav>
</body></html>
'''


def _opf(pages: list[dict], meta: dict, manifest_images: list[tuple[str, str, str]]) -> str:
    lang = meta.get("language") or "ko"
    title = _xe(meta.get("title") or "untitled")
    uid = meta.get("uid") or str(uuid4())
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    direction = "rtl" if meta.get("rtl", True) else "ltr"
    items = ['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
             '<item id="css" href="styles/fixed.css" media-type="text/css"/>']
    spine = []
    cover_done = False
    for href, media, page in ((h, m, p) for (h, m, p) in manifest_images):
        props = ""
        if page.get("image_href") and not cover_done and not page.get("placeholder"):
            props = ' properties="cover-image"'
            cover_done = True
        item_id = "img-" + page["page_id"]
        items.append(f'<item id="{item_id}" href="{href}" media-type="{media}"{props}/>')
    for page in pages:
        pid = page["page_id"]
        items.append(f'<item id="{pid}" href="text/{pid}.xhtml" media-type="application/xhtml+xml"/>')
        prop = page.get("prop") or ""
        if page.get("placeholder"):
            prop = "rendition:page-spread-center"
        extra = f' properties="{prop}"' if prop else ""
        spine.append(f'<itemref idref="{pid}"{extra}/>')
    return f'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="{OPF}" unique-identifier="uid" version="3.0" xml:lang="{lang}" prefix="rendition: http://www.idpf.org/vocab/rendition/#">
<metadata xmlns:dc="{DC}">
<dc:identifier id="uid">urn:uuid:{uid}</dc:identifier>
<dc:title>{title}</dc:title>
<dc:language>{_xe(lang)}</dc:language>
<meta property="dcterms:modified">{modified}</meta>
<meta property="rendition:layout">pre-paginated</meta>
<meta property="rendition:orientation">auto</meta>
<meta property="rendition:spread">landscape</meta>
</metadata>
<manifest>
{"\n".join(items)}
</manifest>
<spine page-progression-direction="{direction}">
{"\n".join(spine)}
</spine>
</package>
'''


def _container() -> str:
    return '''<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>
'''


def write_fxl_epub(out_path: str, pages: list[dict], meta: dict) -> None:
    """pages: page_id, idx, width, height, image_path|None, media_type, overlays, error, title."""
    spread_properties(pages, bool(meta.get("rtl", True)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".part"
    images = []
    with zipfile.ZipFile(tmp, "w") as z:
        mime = zipfile.ZipInfo("mimetype")
        mime.compress_type = zipfile.ZIP_STORED
        z.writestr(mime, b"application/epub+zip")
        z.writestr("META-INF/container.xml", _container())
        z.writestr("OEBPS/styles/fixed.css", CSS)
        for page in pages:
            href = None
            if page.get("image_path") and os.path.isfile(page["image_path"]):
                media = page.get("media_type") or "image/png"
                href = "images/" + page["page_id"] + ext_for(media)
                with open(page["image_path"], "rb") as f:
                    data = f.read()
                info = zipfile.ZipInfo("OEBPS/" + href)
                info.compress_type = zipfile.ZIP_STORED
                z.writestr(info, data)
                images.append((href, media, page))
                page["image_href"] = href
            else:
                page["placeholder"] = True
                page["image_href"] = None
                images.append((None, None, page))
            xhtml = _page_xhtml(page, ("../" + href) if href else None)
            z.writestr(f"OEBPS/text/{page['page_id']}.xhtml", xhtml.encode("utf-8"))
        z.writestr("OEBPS/nav.xhtml", _nav(pages).encode("utf-8"))
        # Rebuild the image list for the manifest from pages that actually have files.
        manifest_images = []
        for page in pages:
            if page.get("image_href"):
                manifest_images.append((page["image_href"], page.get("media_type") or "image/png", page))
        z.writestr("OEBPS/content.opf", _opf(pages, meta, manifest_images).encode("utf-8"))
    os.replace(tmp, out_path)


def compare_output(out_path: str, pages: list[dict]) -> dict:
    """Hash images inside the EPUB against the files we intended to store. One image at a time."""
    notes = []
    identical = 0
    rendered = 0
    copied = 0
    dim_ok = 0
    order_ok = True
    worst_aspect = 0.0
    with zipfile.ZipFile(out_path) as z:
        names = z.namelist()
        opf = z.read("OEBPS/content.opf").decode("utf-8")
        for i, page in enumerate(pages):
            if f"OEBPS/text/{page['page_id']}.xhtml" not in names:
                order_ok = False
                notes.append(f"{page['page_id']} xhtml 없음")
                continue
            # Spine order follows page order: the Nth itemref is this page.
            token = f'idref="{page["page_id"]}"'
            if opf.find(token) < 0:
                order_ok = False
            if page.get("placeholder") or not page.get("image_path"):
                notes.append(f"{page['page_id']}: 이미지 없이 실패 자리표시")
                continue
            href = "OEBPS/images/" + page["page_id"] + ext_for(page.get("media_type") or "image/png")
            if href not in names:
                order_ok = False
                notes.append(f"{page['page_id']} 이미지 항목 없음")
                continue
            blob = z.read(href)
            digest = hashlib.sha256(blob).hexdigest()
            src_digest = _file_sha(page["image_path"])
            method = page.get("method") or ""
            if digest == src_digest:
                identical += 1
            else:
                notes.append(f"{page['page_id']}: EPUB 이미지 해시가 기록 파일과 다릅니다")
            if method == "render":
                rendered += 1
            elif method.startswith("extract") or method == "copy":
                copied += 1
            from .media import image_info
            info = image_info(blob[: 512 * 1024]) or image_info(blob)
            if info and info[1] == int(page["width"]) and info[2] == int(page["height"]):
                dim_ok += 1
            if info and page.get("width") and page.get("height"):
                src_aspect = page["width"] / page["height"]
                out_aspect = info[1] / info[2]
                if src_aspect:
                    worst_aspect = max(worst_aspect, abs(src_aspect - out_aspect) / src_aspect)
        # Confirm document order of itemrefs matches page list.
        cursor = 0
        for page in pages:
            pos = opf.find(f'idref="{page["page_id"]}"', cursor)
            if pos < 0:
                order_ok = False
                break
            cursor = pos + 1
    return {
        "pages": len(pages),
        "byte_identical_to_stored_file": identical,
        "rendered_pages": rendered,
        "direct_or_copy_pages": copied,
        "dimension_match": dim_ok,
        "order_preserved": order_ok,
        "max_aspect_delta": round(worst_aspect, 6),
        "notes": notes[:50],
    }


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_font_size(text: str, w: int, h: int, vertical: bool) -> tuple[float, bool]:
    """Rough balloon fit. The editor can override it. Returns (px, tight)."""
    n = max(1, len((text or "").strip()))
    if vertical:
        size = min(h / n * 0.95, w * 0.72)
    else:
        cols = max(1, min(n, max(1, int(w / 16) or 1)))
        rows = max(1, (n + cols - 1) // cols)
        size = min(w / cols * 0.95, h / rows * 0.82)
    tight = size < 11
    size = max(10.0, min(42.0, size))
    return round(size, 1), tight
