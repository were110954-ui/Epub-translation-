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


KO_FONT = '"Noto Serif KR", "Noto Serif CJK KR", "Nanum Myeongjo", "RIDIBatang", "KoPubWorld Batang", "Apple SD Gothic Neo", "Malgun Gothic", serif'
KO_CSS_FILE = "kt-ko-readability.css"
KO_CSS = """/* EPUB AI Translator: Korean readability override (target=ko). Reference: Japanese-source output. */
html, body { line-height: 1.8; word-break: keep-all; overflow-wrap: break-word; line-break: strict;
  font-family: %s; letter-spacing: normal; word-spacing: normal; hyphens: none; -webkit-hyphens: none; -epub-hyphens: none; }
p { orphans: 2; widows: 2; }
.kt-leadin::first-letter { font-weight: bold; }
""" % KO_FONT


def _lh(m):
    v, unit = float(m.group(2)), m.group(3) or ""
    ratio = v / 100 if unit == "%" else v
    if ratio < 1.6:
        return m.group(1) + "1.8"
    return m.group(0)


def _ko_css_path(book) -> str:
    base = posixpath.dirname(book.opf_path)
    return posixpath.join(base, KO_CSS_FILE) if base else KO_CSS_FILE


def korean_css(css: str) -> str:
    """Rewrite Latin-typography CSS for Korean text (also used on inline style attributes)."""
    css = re.sub(r"(text-align\s*:\s*)justify", r"\1start", css, flags=re.I)
    css = re.sub(r"(line-height\s*:\s*)normal", r"\g<1>1.8", css, flags=re.I)
    css = re.sub(r"(line-height\s*:\s*)([\d.]+)(%|em)?(?=\s*[;}!]|\s*$)", _lh, css, flags=re.I)
    css = re.sub(r"(font-variant(?:-caps)?\s*:\s*)(all-)?small-caps", r"\1normal", css, flags=re.I)
    css = re.sub(r"(letter-spacing\s*:\s*)-?[\d.]+\s*[a-z%]*", r"\1normal", css, flags=re.I)
    css = re.sub(r"(text-transform\s*:\s*)(uppercase|capitalize)", r"\1none", css, flags=re.I)
    css = re.sub(r"(-(?:webkit|epub|ms)-)?hyphens\s*:\s*auto", "hyphens: none", css, flags=re.I)
    css = re.sub(r"(text-indent\s*:\s*)(1\.[3-9]\d*|[2-3](?:\.\d+)?)em", r"\g<1>1em", css, flags=re.I)
    def ff(m):
        val = m.group(2)
        if re.search(r"KR|Myeongjo|Batang|Gothic|Malgun|monospace", val, re.I):
            return m.group(0)
        return m.group(1) + KO_FONT
    css = re.sub(r"(font-family\s*:\s*)([^;}]+)", ff, css, flags=re.I)
    return css


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
            if horizontal or ko:
                for el in root.iter():
                    if isinstance(el.tag, str) and el.get("style"):
                        st = convert_css(el.get("style")) if horizontal else el.get("style")
                        el.set("style", korean_css(st) if ko else st)
                    if isinstance(el.tag, str) and local(el.tag) == "style" and el.text:
                        st = convert_css(el.text) if horizontal else el.text
                        el.text = korean_css(st) if ko else st
            if ko:
                head = next((h for h in root if isinstance(h.tag, str) and local(h.tag) == "head"), None)
                if head is not None:
                    href = posixpath.relpath(_ko_css_path(book), posixpath.dirname(path) or ".")
                    link = etree.SubElement(head, (root.tag.split("}")[0] + "}link") if root.tag.startswith("{") else "link")
                    link.set("rel", "stylesheet"); link.set("type", "text/css"); link.set("href", href)
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
