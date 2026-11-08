"""Build the translated EPUB: copy every entry of the original zip, replacing only translated text
nodes, convert vertical writing to horizontal, update language metadata. Original file is never touched."""
from __future__ import annotations
import posixpath
import re
import zipfile

from lxml import etree

from .epub_model import EpubBook, NS, XHTML, rebuild_inline, parse_xml, local, merge_leadins

VERTICAL_CSS = [
    (re.compile(r"((?:-epub-|-webkit-|-ms-)?writing-mode\s*:\s*)(vertical-rl|vertical-lr|tb-rl|tb)", re.I), r"\1horizontal-tb"),
    (re.compile(r"((?:-epub-|-webkit-)?text-orientation\s*:\s*)(upright|sideways|sideways-right)", re.I), r"\1mixed"),
    (re.compile(r"((?:-epub-|-webkit-)?text-combine(?:-upright|-horizontal)?\s*:\s*)(all|horizontal|digits\s*\d*)", re.I), r"\1none"),
]


# Serif/Myeongjo only. Gothic faces that ship with iOS and Windows were previously listed
# before `serif`, so a reader set to the book font picked a sans face when the serif names
# were not installed. Exact metrics of any one store app are not reproduced.
KO_FONT = ('"Nanum Myeongjo", "NanumMyeongjo", "Noto Serif KR", "Noto Serif CJK KR", '
           '"KoPubWorld Batang", "KoPub Batang", "RIDIBatang", "Batang", "BatangChe", serif')
KO_LINE = "1.65"
KO_INDENT = "1em"
KO_CSS_FILE = "kt-ko-readability.css"
KO_CSS = """/* Korean novel defaults for reflowable text. Relative units only, so a reader can still override the book font, size, and spacing. */
html { font-family: %s; font-size: 100%%; line-height: %s; letter-spacing: normal; word-spacing: normal;
  word-break: keep-all; overflow-wrap: break-word; line-break: strict;
  hyphens: none; -webkit-hyphens: none; -epub-hyphens: none; }
body { font-family: %s; font-size: 1em; font-weight: normal; line-height: %s; letter-spacing: normal;
  word-spacing: normal; margin: 0; padding: 0; }
p { font-family: inherit; font-size: 1em; font-weight: normal; line-height: %s; letter-spacing: normal;
  word-spacing: normal; margin: 0; padding: 0; text-indent: %s; text-align: left; orphans: 2; widows: 2; }
h1, h2, h3, h4, h5, h6 { font-family: inherit; font-weight: bold; line-height: 1.35; letter-spacing: normal;
  text-indent: 0; margin-top: 1.4em; margin-bottom: 0.6em; }
p.center, p.kt-scene, .center, .kt-scene { text-indent: 0; text-align: center; margin-top: 1em; margin-bottom: 1em; }
.illus, figure, figcaption { text-indent: 0; margin-top: 1em; margin-bottom: 1em; }
p::first-letter, .kt-leadin::first-letter { font-weight: inherit; font-size: 1em; float: none;
  line-height: inherit; margin: 0; padding: 0; }
em, i { font-style: italic; }
strong, b { font-weight: bold; }
""" % (KO_FONT, KO_LINE, KO_FONT, KO_LINE, KO_LINE, KO_INDENT)

_DROP_CLASS = re.compile(r"drop|initial|first-?letter|small-?caps|scaps|drophat", re.I)
_SCENE = re.compile(r"^[\s\u3000◇◆○●＊*·・…―─\-—~※.]+$")
_EMPHASIS_SEL = re.compile(r"(^|[\s>+~])(em|strong|i|b|mark)\b", re.I)


def _ko_css_path(book) -> str:
    base = posixpath.dirname(book.opf_path)
    return posixpath.join(base, KO_CSS_FILE) if base else KO_CSS_FILE


def _selector_kind(selector: str) -> str:
    raw = selector.strip().lower()
    if raw.startswith("@") or not raw:
        return "skip"
    if re.search(r"::?first-letter|drop-?cap|drophat|firstletter", raw):
        return "dropcap"
    if re.search(r"\b(img|svg|image|figure|figcaption|video|hr|canvas)\b", raw):
        return "media"
    if re.search(r"center|scene|asterism|ornament|separator|kt-scene", raw):
        return "scene"
    if re.search(r"(^|[\s>+~,:])h[1-6]\b", raw):
        return "heading"
    if _EMPHASIS_SEL.search(raw) and not re.search(r"\bp\b", raw):
        return "emphasis"
    return "prose"


def _paragraph_selector(selector: str) -> bool:
    for part in selector.split(","):
        piece = part.strip().split()[-1] if part.strip() else ""
        piece = piece.split("::")[0].split(":")[0]
        if re.search(r"(^|[^a-z])p($|[^a-z])", piece):
            return True
    return False


def _serif_family(value: str) -> str:
    val = value.strip().rstrip("!important").strip()
    if re.search(r"monospace", val, re.I):
        return value.strip()
    if re.search(r"myeongjo|batang|noto serif|kopub|ridi", val, re.I) and not re.search(
            r"sans-serif|gothic|고딕|dotum|gulim|malgun", val, re.I):
        return value.strip()
    return KO_FONT


def _font_size(value: str, kind: str) -> str:
    m = re.match(r"^([\d.]+)\s*(px|pt|em|rem|%)?$", value.strip(), re.I)
    if not m:
        return value.strip()
    num, unit = float(m.group(1)), (m.group(2) or "").lower()
    if kind == "heading":
        return "1.4em" if unit in ("px", "pt") else value.strip()
    if unit in ("px", "pt"):
        return "1em"
    if unit == "%" and (num >= 125 or 90 <= num <= 115):
        return "1em"
    if unit in ("em", "rem", "") and num >= 1.35:
        return "1em"
    return value.strip()


def _line_height(value: str, kind: str) -> str:
    raw = value.strip().lower()
    if raw in ("inherit", "normal"):
        return "1.35" if kind == "heading" else KO_LINE
    m = re.match(r"^([\d.]+)\s*(px|pt|em|rem|%)?$", raw, re.I)
    if not m:
        return value.strip()
    if kind == "heading":
        num, unit = float(m.group(1)), (m.group(2) or "")
        ratio = num / 100 if unit == "%" else num
        if unit in ("px", "pt") or ratio > 1.55 or ratio < 1.15:
            return "1.35"
        return value.strip()
    return KO_LINE


def _rewrite_body(selector: str, body: str, ensure: bool = True) -> str:
    kind = _selector_kind(selector)
    if kind == "skip":
        return body
    if kind == "dropcap":
        body = re.sub(r"font-weight\s*:\s*[^;]+", "font-weight: inherit", body, flags=re.I)
        body = re.sub(r"font-size\s*:\s*[^;]+", "font-size: 1em", body, flags=re.I)
        body = re.sub(r"float\s*:\s*[^;]+", "float: none", body, flags=re.I)
        body = re.sub(r"line-height\s*:\s*[^;]+", "line-height: inherit", body, flags=re.I)
        body = re.sub(r"margin[^:]*\s*:\s*[^;]+", "margin: 0", body, flags=re.I)
        return body
    decls = []
    for raw in body.split(";"):
        if ":" not in raw:
            if raw.strip():
                decls.append(raw.strip())
            continue
        key, val = raw.split(":", 1)
        key, val = key.strip(), val.strip()
        low = key.lower()
        important = bool(re.search(r"!important\s*$", val, re.I))
        val = re.sub(r"\s*!important\s*$", "", val, flags=re.I).strip()
        if low == "font-family" and kind != "media":
            val = _serif_family(val)
        elif low == "letter-spacing" and kind != "media":
            val = "normal"
        elif low == "word-spacing" and kind != "media":
            val = "normal"
        elif low in ("font-variant", "font-variant-caps") and kind != "emphasis":
            val = "normal"
        elif low == "text-transform" and val.lower() in ("uppercase", "capitalize"):
            val = "none"
        elif low.endswith("hyphens") and val.lower() == "auto":
            val = "none"
        elif low == "text-align" and val.lower() == "justify":
            val = "left"
        elif low == "line-height" and kind in ("prose", "heading", "scene"):
            val = _line_height(val, "heading" if kind == "heading" else "prose")
        elif low == "font-size" and kind in ("prose", "heading", "scene"):
            val = _font_size(val, "heading" if kind == "heading" else "prose")
        elif low == "font-weight" and kind == "prose" and _paragraph_selector(selector):
            val = "normal"
        elif low in ("margin", "margin-top", "margin-bottom") and kind == "prose" and _paragraph_selector(selector):
            val = "0"
        elif low in ("padding-top", "padding-bottom") and kind == "prose" and _paragraph_selector(selector):
            val = "0"
        elif low == "text-indent" and kind == "prose" and _paragraph_selector(selector):
            val = KO_INDENT
        elif low == "text-indent" and kind == "scene":
            val = "0"
        suffix = " !important" if important and low not in (
            "font-family", "font-size", "line-height", "letter-spacing", "word-spacing", "margin",
            "margin-top", "margin-bottom", "text-indent", "font-weight") else ""
        decls.append(f"{key}: {val}{suffix}")
    if ensure and kind == "prose" and _paragraph_selector(selector):
        keys = {d.split(":", 1)[0].strip().lower() for d in decls if ":" in d}
        if "text-indent" not in keys:
            decls.append(f"text-indent: {KO_INDENT}")
        if "margin" not in keys and "margin-top" not in keys and "margin-bottom" not in keys:
            decls.append("margin: 0")
        if "line-height" not in keys:
            decls.append(f"line-height: {KO_LINE}")
        if "font-family" not in keys:
            decls.append(f"font-family: {KO_FONT}")
    return "; ".join(decls) + (";" if decls else "")


def korean_css(css: str) -> str:
    """Rewrite author CSS so Korean body text is a serif novel, without touching fixed-layout comics.

    Class rules beat a later element selector, so metrics are rewritten inside the original rules.
    `!important` is removed from those metrics so a reading app can still override them.
    """
    def repl(match):
        selector, body = match.group(1), match.group(2)
        if selector.strip().startswith("@"):
            return match.group(0)
        return selector + "{" + _rewrite_body(selector, body, ensure=True) + "}"
    return re.sub(r"([^{}]+)\{([^{}]*)\}", repl, css, flags=re.S)


def _font_size_em(style: str) -> float | None:
    m = re.search(r"font-size\s*:\s*([\d.]+)\s*(em|rem|%)?", style or "", re.I)
    if not m:
        return None
    num, unit = float(m.group(1)), (m.group(2) or "").lower()
    if unit == "%":
        return num / 100
    return num


def _is_dropcap(el) -> bool:
    if not isinstance(el.tag, str):
        return False
    name = local(el.tag)
    if name not in ("span", "b", "big", "font", "small"):
        return False
    if any(local(c.tag) in ("img", "ruby", "svg", "image") for c in el.iter() if isinstance(c.tag, str) and c is not el):
        return False
    text = "".join(el.itertext()).strip()
    if not text or len(text) > 4:
        return False
    cls = el.get("class") or ""
    style = el.get("style") or ""
    if _DROP_CLASS.search(cls) or _DROP_CLASS.search(style):
        return True
    size = _font_size_em(style)
    if size is not None and size >= 1.45:
        return True
    if re.search(r"float\s*:\s*(left|start)", style, re.I):
        return True
    if re.search(r"font-variant(?:-caps)?\s*:\s*(?:all-)?small-caps", style, re.I):
        return True
    parent = el.getparent()
    at_start = parent is not None and el.getprevious() is None and not (parent.text or "").strip()
    if at_start and len(text) <= 2 and (
            re.search(r"font-weight\s*:\s*(bold|[6-9]00)", style, re.I) or name in ("b", "big")):
        return True
    return False


def _unwrap(el):
    parent = el.getparent()
    if parent is None:
        return
    index = parent.index(el)
    if el.text:
        if index == 0:
            parent.text = (parent.text or "") + el.text
        else:
            prev = parent[index - 1]
            prev.tail = (prev.tail or "") + el.text
    children = list(el)
    for offset, child in enumerate(children):
        parent.insert(index + offset, child)
    if el.tail:
        if children:
            moved = parent[index + len(children) - 1]
            moved.tail = (moved.tail or "") + el.tail
        elif index == 0:
            parent.text = (parent.text or "") + el.tail
        else:
            prev = parent[index - 1]
            prev.tail = (prev.tail or "") + el.tail
    parent.remove(el)


def normalize_korean_document(root) -> None:
    """Drop source-language drop caps and a duplicate indent space. Keep em/strong and headings."""
    for el in list(root.iter()):
        if _is_dropcap(el):
            _unwrap(el)
    for el in root.iter():
        if not isinstance(el.tag, str) or local(el.tag) != "p":
            continue
        bits = [el.text or ""]
        bits.extend((c.tail or "") for c in el)
        plain = "".join(bits).strip()
        cls = (el.get("class") or "").split()
        if plain and _SCENE.match(plain):
            if "kt-scene" not in cls:
                el.set("class", " ".join(cls + ["kt-scene"]))
            continue
        if any(c in cls for c in ("center", "kt-scene")):
            continue
        if el.text and re.match(r"^[\u3000 ]", el.text):
            el.text = re.sub(r"^[\u3000 ]+", "", el.text, count=1)
        elif not (el.text or "").strip():
            for child in el:
                if isinstance(child.tag, str) and child.text and re.match(r"^[\u3000 ]", child.text):
                    child.text = re.sub(r"^[\u3000 ]+", "", child.text, count=1)
                    break
                if isinstance(child.tag, str) and "".join(child.itertext()).strip():
                    break


def _ensure_style_link(root, href: str) -> None:
    head = next((h for h in root if isinstance(h.tag, str) and local(h.tag) == "head"), None)
    if head is None:
        return
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    for link in list(head):
        if isinstance(link.tag, str) and local(link.tag) == "link" and (link.get("href") or "").endswith(KO_CSS_FILE):
            head.remove(link)
    link = etree.SubElement(head, ns + "link")
    link.set("rel", "stylesheet")
    link.set("type", "text/css")
    link.set("href", href)


def convert_css(css: str) -> str:
    for rx, rep in VERTICAL_CSS:
        css = rx.sub(rep, css)
    return css


def serialize(root, original: bytes) -> bytes:
    decl = original.lstrip()[:5] == b"<?xml"
    doctype = root.getroottree().docinfo.doctype
    body = etree.tostring(root, encoding="unicode")
    out = ""
    if decl:
        out += "<?xml version='1.0' encoding='utf-8'?>\n"
    if doctype:
        out += doctype + "\n"
    return (out + body).encode("utf-8")


def apply_translations(book: EpubBook, rows: list[dict], tgt_lang="ko", drop_furigana=True,
                       horizontal=True, translate_title=True) -> dict[str, bytes]:
    """Return {zip_path: new_bytes} for modified entries."""
    by_file: dict[str, list] = {}
    for r in rows:
        if r.get("target"):
            by_file.setdefault(r["file"], []).append(r)
    out = {}
    for path in book.xhtml_items() + ([book.ncx_path] if book.ncx_path else []):
        if path is None:
            continue
        raw = book.read(path)
        root = parse_xml(raw)
        if path != book.ncx_path:
            merge_leadins(root)
        tree = root.getroottree()
        for r in by_file.get(path, []):
            kind = r["kind"]
            if kind == "text":
                ppath, attr, cpath = r["xpath"].split("|")
                parent = tree.xpath(ppath)[0]
                if attr == "text":
                    lead = re.match(r"^\s*", parent.text or "").group(0)
                    trail = re.search(r"\s*$", parent.text or "").group(0)
                    parent.text = lead + r["target"] + trail
                else:
                    ch = tree.xpath(cpath)[0]
                    lead = re.match(r"^\s*", ch.tail or "").group(0)
                    trail = re.search(r"\s*$", ch.tail or "").group(0)
                    ch.tail = lead + r["target"] + trail
            else:
                found = tree.xpath(r["xpath"])
                if not found:
                    continue
                el = found[0]
                if kind in ("title", "ncx"):
                    el.text = r["target"]
                else:
                    rebuild_inline(el, r["target"], drop_furigana)
        if path != book.ncx_path:
            if root.get("{http://www.w3.org/XML/1998/namespace}lang") is not None:
                root.set("{http://www.w3.org/XML/1998/namespace}lang", tgt_lang)
            if root.get("lang") is not None:
                root.set("lang", tgt_lang)
            ko = tgt_lang.lower().startswith("ko")
            if ko:
                normalize_korean_document(root)
            if horizontal or ko:
                for el in root.iter():
                    if isinstance(el.tag, str) and el.get("style"):
                        st = convert_css(el.get("style")) if horizontal else el.get("style")
                        el.set("style", _rewrite_body(local(el.tag) or "span", st, ensure=False) if ko else st)
                    if isinstance(el.tag, str) and local(el.tag) == "style" and el.text:
                        st = convert_css(el.text) if horizontal else el.text
                        el.text = korean_css(st) if ko else st
            if ko:
                href = posixpath.relpath(_ko_css_path(book), posixpath.dirname(path) or ".")
                _ensure_style_link(root, href)
        else:
            if root.get("{http://www.w3.org/XML/1998/namespace}lang") is not None:
                root.set("{http://www.w3.org/XML/1998/namespace}lang", tgt_lang)
        out[path] = serialize(root, raw)
    # CSS
    ko = tgt_lang.lower().startswith("ko")
    if horizontal or ko:
        for m in book.manifest.values():
            if m["media_type"] == "text/css" and m["path"] in book.names:
                css = book.read(m["path"]).decode("utf-8", "replace")
                new = convert_css(css) if horizontal else css
                if ko:
                    new = korean_css(new)
                if new != css:
                    out[m["path"]] = new.encode("utf-8")
    if ko:
        out[_ko_css_path(book)] = KO_CSS.encode("utf-8")
    # OPF
    raw = book.read(book.opf_path)
    opf = parse_xml(raw)
    lang = opf.find("opf:metadata/dc:language", NS)
    if lang is not None:
        lang.text = tgt_lang
    if translate_title:
        tr = next((r for r in rows if r["kind"] == "meta" and r.get("target")), None)
        t = opf.find("opf:metadata/dc:title", NS)
        if tr and t is not None:
            t.text = tr["target"]
    if ko:
        man = opf.find("opf:manifest", NS)
        href = posixpath.relpath(_ko_css_path(book), posixpath.dirname(book.opf_path) or ".")
        if man is not None and not any(i.get("href") == href for i in man):
            it = etree.SubElement(man, f"{{{NS['opf']}}}item")
            it.set("id", "kt-ko-readability-css"); it.set("href", href); it.set("media-type", "text/css")
    spine = opf.find("opf:spine", NS)
    if horizontal and spine is not None and spine.get("page-progression-direction") == "rtl":
        spine.set("page-progression-direction", "ltr")
    if horizontal:
        for meta in opf.findall("opf:metadata/opf:meta", NS):
            if meta.get("name") == "primary-writing-mode":
                meta.set("content", "horizontal-lr")
            if meta.get("property") == "rendition:spread" and False:
                pass
    out[book.opf_path] = serialize(opf, raw)
    return out


def is_fixed_layout(book: EpubBook) -> bool:
    for meta in book.opf.findall("opf:metadata/opf:meta", NS):
        if meta.get("property") == "rendition:layout" and (meta.text or "").strip() == "pre-paginated":
            return True
    return False


def restyle_epub(src_path: str, out_path: str) -> str:
    """Reapply Korean novel defaults to an existing reflowable EPUB. No translation API calls.

    The source file is not modified. Fixed-layout comics are refused.
    """
    book = EpubBook(src_path)
    if is_fixed_layout(book):
        raise ValueError("고정 레이아웃 EPUB에는 소설 서체를 적용하지 않습니다.")
    out: dict[str, bytes] = {}
    href_base = _ko_css_path(book)
    for path in book.xhtml_items():
        raw = book.read(path)
        root = parse_xml(raw)
        normalize_korean_document(root)
        for el in root.iter():
            if isinstance(el.tag, str) and el.get("style"):
                el.set("style", _rewrite_body(local(el.tag) or "span", convert_css(el.get("style")), ensure=False))
            if isinstance(el.tag, str) and local(el.tag) == "style" and el.text:
                el.text = korean_css(convert_css(el.text))
        _ensure_style_link(root, posixpath.relpath(href_base, posixpath.dirname(path) or "."))
        out[path] = serialize(root, raw)
    for m in book.manifest.values():
        if m["media_type"] == "text/css" and m["path"] in book.names:
            css = book.read(m["path"]).decode("utf-8", "replace")
            new = korean_css(convert_css(css))
            if new != css:
                out[m["path"]] = new.encode("utf-8")
    out[href_base] = KO_CSS.encode("utf-8")
    raw = book.read(book.opf_path)
    opf = parse_xml(raw)
    lang = opf.find("opf:metadata/dc:language", NS)
    if lang is not None and not (lang.text or "").lower().startswith("ko"):
        # Keep a non-Korean language tag if the book was not translated; still attach the stylesheet.
        pass
    man = opf.find("opf:manifest", NS)
    href = posixpath.relpath(href_base, posixpath.dirname(book.opf_path) or ".")
    if man is not None and not any(i.get("href") == href for i in man):
        it = etree.SubElement(man, f"{{{NS['opf']}}}item")
        it.set("id", "kt-ko-readability-css")
        it.set("href", href)
        it.set("media-type", "text/css")
    out[book.opf_path] = serialize(opf, raw)
    write_epub(book, out, out_path)
    return out_path


def write_epub(book: EpubBook, replacements: dict[str, bytes], out_path: str):
    with zipfile.ZipFile(out_path, "w") as z:
        # mimetype first, stored, no extra field
        z.writestr(zipfile.ZipInfo("mimetype"), book.read("mimetype") if "mimetype" in book.names
                   else b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for info in book.zf.infolist():
            if info.filename == "mimetype" or info.is_dir():
                continue
            data = replacements.get(info.filename, None)
            if data is None:
                data = book.read(info.filename)
            z.writestr(zipfile.ZipInfo(info.filename, date_time=info.date_time), data,
                       compress_type=zipfile.ZIP_DEFLATED)
        for name, data in replacements.items():   # new entries (e.g. Korean readability CSS)
            if name not in book.names:
                z.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
