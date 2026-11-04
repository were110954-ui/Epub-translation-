"""EPUB container / OPF / spine / navigation parsing and block extraction.

Works directly on the ZIP and lxml trees: only text inside block elements is
replaced. Inline markup (em, span, a, ruby, br, img ...) is serialized as
numbered placeholder tokens <tN>..</tN> / <tN/> and rebuilt from the original
element objects, so attributes, ids and links survive translation.
"""
from __future__ import annotations
import copy
import hashlib
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from urllib.parse import unquote

from lxml import etree

from .textutil import is_translatable, kana_only, KANJI, HIRA

NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "x": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
}
XHTML = "{http://www.w3.org/1999/xhtml}"

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "td", "th",
              "caption", "figcaption", "blockquote", "div", "pre", "aside", "section",
              "article", "header", "footer", "nav", "ol", "ul", "dl", "table", "tr",
              "tbody", "thead", "tfoot", "figure", "body", "hr", "main", "address", "details", "summary"}
SKIP_TAGS = {"script", "style", "math", "svg", "head", "title", "code"}
# tags removed (with content) from translation text but kept in place: rt for furigana
RUBY_READING_TAGS = {"rt", "rp"}


class DRMError(Exception):
    pass


class EpubError(Exception):
    pass


def local(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[1] if "}" in tag else tag


@dataclass
class Block:
    block_id: str          # stable id: <file>#<n>
    file: str              # zip path
    index: int             # order within the whole book
    chapter: str           # chapter title (best effort)
    kind: str              # 'block' | 'text' (mixed-content text node) | 'title' | 'ncx' | 'nav' | 'meta'
    source: str            # token markup
    xpath: str = ""
    furigana: dict = field(default_factory=dict)  # base -> reading hints


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_xml(data: bytes):
    parser = etree.XMLParser(recover=False, resolve_entities=False, remove_blank_text=False, huge_tree=True)
    try:
        return etree.fromstring(data, parser)
    except etree.XMLSyntaxError:
        # Fallback: recover mode (some calibre exports have small errors / HTML entities)
        parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
        root = etree.fromstring(data, parser)
        if root is None:
            raise
        return root


class EpubBook:
    """Read-only view on an EPUB file (the original is never modified)."""

    def __init__(self, path: str):
        self.path = path
        self.zf = zipfile.ZipFile(path, "r")
        self.names = self.zf.namelist()
        self.check_drm()
        self.opf_path = self._find_opf()
        self.opf_dir = posixpath.dirname(self.opf_path)
        self.opf = parse_xml(self.zf.read(self.opf_path))
        self.version = self.opf.get("version", "2.0")
        self.manifest = {}   # id -> dict(href, path, media_type, properties)
        for it in self.opf.findall("opf:manifest/opf:item", NS):
            href = it.get("href")
            self.manifest[it.get("id")] = {
                "href": href,
                "path": self.resolve(self.opf_dir, href),
                "media_type": it.get("media-type", ""),
                "properties": it.get("properties", ""),
            }
        spine = self.opf.find("opf:spine", NS)
        self.spine_el = spine
        self.spine = [self.manifest[i.get("idref")]["path"] for i in spine.findall("opf:itemref", NS)
                      if i.get("idref") in self.manifest]
        self.ncx_path = None
        toc_id = spine.get("toc")
        if toc_id and toc_id in self.manifest:
            self.ncx_path = self.manifest[toc_id]["path"]
        else:
            for m in self.manifest.values():
                if m["media_type"] == "application/x-dtbncx+xml":
                    self.ncx_path = m["path"]
        self.nav_path = next((m["path"] for m in self.manifest.values() if "nav" in m["properties"].split()), None)
        self.title = self.meta("title")
        self.language = self.meta("language")
        self._chapter_titles = self._toc_titles()

    # ---------------------------------------------------------------- basics
    @staticmethod
    def resolve(base_dir: str, href: str) -> str:
        href = unquote(href.split("#", 1)[0])
        return posixpath.normpath(posixpath.join(base_dir, href)) if base_dir else posixpath.normpath(href)

    def check_drm(self):
        if "META-INF/encryption.xml" in self.names:
            enc = self.zf.read("META-INF/encryption.xml").decode("utf-8", "ignore")
            # font obfuscation (IDPF/Adobe) is allowed; anything else = DRM
            algos = re.findall(r'Algorithm="([^"]+)"', enc)
            font_only = all(a in ("http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC") for a in algos)
            if not font_only:
                raise DRMError("이 EPUB은 DRM(암호화)으로 보호되어 있습니다. DRM 우회/제거는 지원하지 않습니다.")
        if "META-INF/rights.xml" in self.names or any(n.endswith(".sinf") for n in self.names):
            raise DRMError("이 EPUB은 DRM 권한 파일(rights.xml/sinf)을 포함합니다. 처리할 수 없습니다.")

    def _find_opf(self) -> str:
        if "META-INF/container.xml" not in self.names:
            raise EpubError("META-INF/container.xml 이 없습니다 (EPUB 아님)")
        c = parse_xml(self.zf.read("META-INF/container.xml"))
        rf = c.find(".//c:rootfile", NS)
        if rf is None:
            raise EpubError("container.xml 에 rootfile 이 없습니다")
        return rf.get("full-path")

    def meta(self, name: str) -> str:
        el = self.opf.find(f"opf:metadata/dc:{name}", NS)
        return (el.text or "").strip() if el is not None else ""

    def read(self, name: str) -> bytes:
        return self.zf.read(name)

    def xhtml_items(self) -> list[str]:
        """Spine documents first (reading order), then other XHTML in manifest."""
        out = list(dict.fromkeys(self.spine))
        for m in self.manifest.values():
            if m["media_type"] in ("application/xhtml+xml", "text/html") and m["path"] not in out:
                out.append(m["path"])
        return [p for p in out if p in self.names]

    def _toc_titles(self) -> dict:
        titles = {}
        try:
            if self.nav_path:
                nav = parse_xml(self.read(self.nav_path))
                base = posixpath.dirname(self.nav_path)
                for a in nav.iter(XHTML + "a"):
                    if a.get("href"):
                        titles.setdefault(self.resolve(base, a.get("href")), "".join(a.itertext()).strip())
            if self.ncx_path:
                ncx = parse_xml(self.read(self.ncx_path))
                base = posixpath.dirname(self.ncx_path)
                for np_ in ncx.iter("{%s}navPoint" % NS["ncx"]):
                    lab = np_.find("ncx:navLabel/ncx:text", NS)
                    con = np_.find("ncx:content", NS)
                    if lab is not None and con is not None:
                        titles.setdefault(self.resolve(base, con.get("src")), (lab.text or "").strip())
        except Exception:
            pass
        return titles

    def chapter_title_for(self, path: str, last: str) -> str:
        return self._chapter_titles.get(path) or last or posixpath.basename(path)

    # -------------------------------------------------------- block extraction
    def extract_blocks(self, drop_furigana: bool = True) -> tuple[list[Block], dict]:
        """Return all translatable blocks and the parsed trees {path: root}."""
        blocks: list[Block] = []
        trees = {}
        chapter = ""
        for path in self.xhtml_items():
            root = parse_xml(self.read(path))
            merge_leadins(root)
            trees[path] = root
            chapter = self.chapter_title_for(path, chapter)
            for el, kind in iter_block_elements(root):
                if kind == "title":
                    src = el.text or ""
                    if is_translatable(src):
                        blocks.append(Block(f"{path}#title", path, len(blocks), chapter, "title", src, root.getroottree().getpath(el)))
                    continue
                if kind == "text":
                    # el is (parent, attr) for mixed-content loose text
                    parent, attr, child = el
                    src = parent.text if attr == "text" else child.tail
                    if src and is_translatable(src):
                        n = len([b for b in blocks if b.file == path])
                        blocks.append(Block(f"{path}#{n}", path, len(blocks), chapter, "text", src.strip(),
                                            root.getroottree().getpath(parent) + f"|{attr}|" + (root.getroottree().getpath(child) if child is not None else "")))
                    continue
                markup, _, furi = serialize_inline(el, drop_furigana)
                if is_translatable(markup):
                    n = len([b for b in blocks if b.file == path])
                    blocks.append(Block(f"{path}#{n}", path, len(blocks), chapter, "block", markup,
                                        root.getroottree().getpath(el), furi))
        # NCX labels
        if self.ncx_path:
            ncx = parse_xml(self.read(self.ncx_path))
            trees[self.ncx_path] = ncx
            for i, t in enumerate(ncx.iter("{%s}text" % NS["ncx"])):
                if t.text and is_translatable(t.text):
                    blocks.append(Block(f"{self.ncx_path}#ncx{i}", self.ncx_path, len(blocks), "目次", "ncx",
                                        t.text.strip(), ncx.getroottree().getpath(t)))
        # OPF title
        if self.title and is_translatable(self.title):
            blocks.append(Block(f"{self.opf_path}#title", self.opf_path, len(blocks), "metadata", "meta", self.title, ""))
        return blocks, trees


def has_block_descendant(el) -> bool:
    for d in el.iterdescendants():
        if local(d.tag) in BLOCK_TAGS:
            return True
    return False


def iter_block_elements(root):
    """Yield (element, kind). kind='block' for leaf blocks; 'text' for loose text in mixed containers; 'title'."""
    head = root.find(XHTML + "head")
    if head is not None:
        t = head.find(XHTML + "title")
        if t is not None:
            yield t, "title"
    body = root.find(XHTML + "body")
    if body is None:
        return

    def walk(el):
        tag = local(el.tag)
        if tag in SKIP_TAGS or not isinstance(el.tag, str):
            return
        if tag in BLOCK_TAGS and not has_block_descendant(el):
            yield el, "block"
            return
        # container with block descendants: handle loose text + inline runs, then recurse
        if el.text and el.text.strip():
            yield (el, "text", None), "text"
        for ch in el:
            if not isinstance(ch.tag, str):
                if ch.tail and ch.tail.strip():
                    yield (el, "tail", ch), "text"
                continue
            ctag = local(ch.tag)
            if ctag in BLOCK_TAGS or has_block_descendant(ch):
                yield from walk(ch)
            elif ctag not in SKIP_TAGS and "".join(ch.itertext()).strip():
                # inline element directly inside a container -> treat as its own block
                yield ch, "block"
            if ch.tail and ch.tail.strip():
                yield (el, "tail", ch), "text"

    yield from walk(body)


def is_reading_ruby(ruby) -> tuple[bool, str, str]:
    """Furigana check: rt is pure kana and base has kanji => reading aid (droppable for Korean)."""
    base = "".join(t for t in _ruby_base_texts(ruby))
    rt = "".join("".join(r.itertext()) for r in ruby if local(r.tag) == "rt")
    # hiragana reading over kanji = furigana (droppable). Katakana / kanji / latin glosses (e.g. 蜃気楼→ミラージュ,
    # 本気→マジ) carry meaning and are preserved as ruby with a translated annotation.
    hira_only = bool(rt.strip()) and all(HIRA.match(c) or c in "ー・ 　" for c in rt.strip())
    return (bool(KANJI.search(base)) and hira_only), base, rt


def _ruby_base_texts(ruby):
    if ruby.text:
        yield ruby.text
    for ch in ruby:
        if local(ch.tag) in RUBY_READING_TAGS:
            pass
        elif isinstance(ch.tag, str):
            yield "".join(ch.itertext())
        if ch.tail:
            yield ch.tail


def serialize_inline(el, drop_furigana=True):
    """Serialize children of `el` to token markup. Returns (markup, token_map, furigana_hints).
    token_map: n -> original element (shallow semantics; children rebuilt from markup)."""
    tokens = {}
    furi = {}
    counter = [0]

    def ser(node) -> str:
        out = []
        if node.text:
            out.append(node.text)
        for ch in node:
            if not isinstance(ch.tag, str):          # comments / PIs: keep as token
                counter[0] += 1
                n = counter[0]
                tokens[n] = ch
                out.append(f"<t{n}/>")
            else:
                tag = local(ch.tag)
                if tag == "ruby":
                    reading, base, rt = is_reading_ruby(ch)
                    if reading:
                        furi[base] = rt
                    if reading and drop_furigana:
                        out.append(base)
                    else:
                        counter[0] += 1
                        n = counter[0]
                        tokens[n] = ch
                        out.append(f"<t{n}>{ser(ch)}</t{n}>")
                elif tag == "rp":
                    counter[0] += 1
                    n = counter[0]
                    tokens[n] = ch
                    out.append(f"<t{n}/>")
                elif len(ch) == 0 and not ch.text:
                    counter[0] += 1
                    n = counter[0]
                    tokens[n] = ch
                    out.append(f"<t{n}/>")
                else:
                    counter[0] += 1
                    n = counter[0]
                    tokens[n] = ch
                    out.append(f"<t{n}>{ser(ch)}</t{n}>")
            if ch.tail:
                out.append(ch.tail)
        return "".join(out)

    markup = ser(el)
    return markup, tokens, furi


TOKEN_RE = re.compile(r"<(/?)t(\d+)(/?)>")


class TokenError(ValueError):
    pass


def check_tokens(src: str, dst: str) -> list[str]:
    """Return list of problems if dst token structure doesn't match src (same set, well nested)."""
    probs = []
    s = sorted(TOKEN_RE.findall(src))
    d = sorted(TOKEN_RE.findall(dst))
    if s != d:
        probs.append("tag-mismatch")
    stack = []
    for close, n, selfc in TOKEN_RE.findall(dst):
        if selfc:
            continue
        if not close:
            stack.append(n)
        elif not stack or stack.pop() != n:
            probs.append("tag-nesting")
            break
    if stack:
        probs.append("tag-unclosed")
    return probs


def rebuild_inline(el, translated: str, drop_furigana=True):
    """Replace the content of `el` with translated token markup, re-creating original inline elements."""
    _, tokens, _ = serialize_inline(el, drop_furigana)
    tokens = {n: copy.deepcopy(t) for n, t in tokens.items()}
    # build new children
    for ch in list(el):
        el.remove(ch)
    el.text = None
    stack = [el]
    pos = 0
    last = {id(el): None}

    def add_text(txt):
        if not txt:
            return
        parent = stack[-1]
        lc = last.get(id(parent))
        if lc is None:
            parent.text = (parent.text or "") + txt
        else:
            lc.tail = (lc.tail or "") + txt

    for m in TOKEN_RE.finditer(translated):
        add_text(translated[pos:m.start()])
        pos = m.end()
        close, n, selfc = m.group(1), int(m.group(2)), m.group(3)
        tmpl = tokens.get(n)
        if tmpl is None:
            continue
        if selfc:
            node = copy.deepcopy(tmpl)
            node.tail = None
            stack[-1].append(node)
            last[id(stack[-1])] = node
        elif not close:
            node = copy.deepcopy(tmpl)
            node.tail = None
            node.text = None
            for c in list(node):
                node.remove(c)
            stack[-1].append(node)
            last[id(stack[-1])] = node
            stack.append(node)
            last[id(node)] = None
        else:
            if len(stack) > 1:
                stack.pop()
    add_text(translated[pos:])


def strip_markup(s: str) -> str:
    return TOKEN_RE.sub("", s)


# ------------------------------------------------------------------ drop-cap / small-caps lead-ins
_LEAD_LETTER = re.compile(r"^[A-Za-z\u201c\"']{1,2}$")
_LEAD_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 ,.'\u2019\-!?]*$")


def merge_leadins(root) -> int:
    """English (and other Latin) books often open a chapter with a drop cap + small-caps lead-in:
    <p><span big>H</span><span small>AVING ANGUISHED</span><span> till I could...</span></p>.
    Translated token-by-token, the lead-in stays English and the sentence is cut in half.
    Merge the two spans back into the following text (sentence case) so the whole sentence is one
    translation unit. Applied identically at extraction and rebuild time, so tokens always match."""
    n = 0
    for el in root.iter():
        if not isinstance(el.tag, str) or local(el.tag) not in BLOCK_TAGS or (el.text or "").strip():
            continue
        kids = [k for k in el if isinstance(k.tag, str)]
        if len(kids) < 2:
            continue
        a, b = kids[0], kids[1]
        if len(a) or len(b) or (a.tail or "").strip() or local(a.tag) not in ("span", "b", "strong", "big", "small") \
                or local(b.tag) not in ("span", "b", "strong", "big", "small"):
            continue
        at, bt = (a.text or ""), (b.text or "")
        if not _LEAD_LETTER.match(at.strip()) or not _LEAD_CAPS.match(bt.strip()) or sum(c.isalpha() for c in bt) < 2:
            continue
        word = at.strip() + bt.rstrip()
        word = word[:1] + word[1:].lower() if word[:1].isalpha() else word[:2] + word[2:].lower()
        word = re.sub(r"\bi\b", "I", word)
        rest = b.tail or ""
        nxt = b.getnext()
        if not rest.strip() and nxt is not None and isinstance(nxt.tag, str) and local(nxt.tag) == local(b.tag) and len(nxt) == 0:
            nxt.text = word + (nxt.text or "")
        else:
            el.text = (el.text or "") + word + rest
            rest = None
        el.remove(a)
        el.remove(b)
        cls = (el.get("class") or "").split()
        if "kt-leadin" not in cls:
            el.set("class", " ".join(cls + ["kt-leadin"]))
        n += 1
    return n
