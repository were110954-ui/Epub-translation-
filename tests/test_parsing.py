import copy
from lxml import etree

from epub_translator.epub_model import EpubBook, serialize_inline, rebuild_inline, check_tokens, DRMError
from epub_translator.chunker import make_chunks


def test_extract_blocks_epub3(samples):
    b = EpubBook(samples["v3"])
    assert b.version == "3.0" and b.nav_path and b.ncx_path
    blocks, _ = b.extract_blocks()
    ids = [x.block_id for x in blocks]
    assert len(ids) == len(set(ids)), "block ids must be unique"
    kinds = {x.kind for x in blocks}
    assert {"block", "title", "ncx", "meta"} <= kinds
    src = "\n".join(x.source for x in blocks)
    # furigana (hiragana reading) dropped from translation text, meaningful katakana ruby kept as tokens
    assert "みなせ" not in src and "水瀬" in src
    mir = next(x for x in blocks if "蜃気楼" in x.source)
    assert "ミラージュ" in mir.source and "<t" in mir.source
    first = next(x for x in blocks if "水瀬" in x.source)
    assert first.furigana.get("水瀬") == "みなせ"
    # nav labels are blocks of the nav document
    assert any(x.file.endswith("nav.xhtml") and "第一章" in x.source for x in blocks)


def test_extract_blocks_epub2(samples):
    b = EpubBook(samples["v2"])
    assert b.version == "2.0" and b.nav_path is None and b.ncx_path
    blocks, _ = b.extract_blocks()
    assert len(blocks) > 20


def test_inline_roundtrip_identity():
    xml = ('<p xmlns="http://www.w3.org/1999/xhtml">前<em class="x">強<b>調</b></em>中<br/>'
           '<ruby>蜃気楼<rt>ミラージュ</rt></ruby><a href="#n1" id="r1">＊1</a>後</p>')
    el = etree.fromstring(xml)
    before = etree.tostring(el)
    markup, tokens, _ = serialize_inline(el, drop_furigana=True)
    assert markup.count("<t") == markup.count("</t") + markup.count("/>")
    el2 = copy.deepcopy(el)
    rebuild_inline(el2, markup)
    assert etree.tostring(el2) == before


def test_rebuild_translated_keeps_attributes():
    xml = '<p xmlns="http://www.w3.org/1999/xhtml">彼は<a href="n.xhtml#n1" id="r1">＊1</a>と<em>言った</em>。</p>'
    el = etree.fromstring(xml)
    markup, _, _ = serialize_inline(el)
    assert markup == "彼は<t1>＊1</t1>と<t2>言った</t2>。"
    rebuild_inline(el, "그는 <t2>말했다</t2><t1>＊1</t1>.")
    s = etree.tostring(el, encoding="unicode")
    assert 'href="n.xhtml#n1"' in s and 'id="r1"' in s and "<em>말했다</em>" in s


def test_furigana_dropped_on_rebuild():
    xml = '<p xmlns="http://www.w3.org/1999/xhtml"><ruby>香<rt>こう</rt></ruby>崎</p>'
    el = etree.fromstring(xml)
    markup, _, furi = serialize_inline(el)
    assert markup == "香崎" and furi == {"香": "こう"}
    rebuild_inline(el, "코자키")
    assert etree.tostring(el, encoding="unicode").endswith(">코자키</p>")


def test_check_tokens():
    assert check_tokens("a<t1>b</t1><t2/>", "x<t1>y</t1><t2/>") == []
    assert "tag-mismatch" in check_tokens("a<t1>b</t1>", "xy")
    assert "tag-nesting" in check_tokens("<t1><t2>a</t2></t1>", "<t1><t2>a</t1></t2>")


def test_chunker_covers_all_blocks(samples):
    blocks, _ = EpubBook(samples["v3"]).extract_blocks()
    rows = [dict(block_id=b.block_id, file=b.file, chapter=b.chapter, source=b.source, kind=b.kind) for b in blocks]
    chunks = make_chunks(rows, chunk_size=120)
    flat = [i for c in chunks for i in c["block_ids"]]
    assert flat == [r["block_id"] for r in rows]          # every block exactly once, in order
    for c in chunks:
        files = {r["file"] for r in rows if r["block_id"] in c["block_ids"]}
        # a chunk spans several files only when merging tiny files (title/image pages)
        assert len(files) == 1 or c["chars"] <= 60


def test_drm_rejected(tmp_path, samples):
    import zipfile
    p = tmp_path / "drm.epub"
    with zipfile.ZipFile(samples["v3"]) as zi, zipfile.ZipFile(p, "w") as zo:
        for i in zi.infolist():
            zo.writestr(i, zi.read(i))
        zo.writestr("META-INF/encryption.xml", '<encryption><EncryptedData><EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#aes128-cbc"/></EncryptedData></encryption>')
    import pytest
    with pytest.raises(DRMError):
        EpubBook(str(p))


def test_dropcap_leadin_merged_and_korean_css():
    from lxml import etree
    from epub_translator.epub_model import merge_leadins, serialize_inline
    from epub_translator.rebuild import korean_css
    x = etree.fromstring('<html xmlns="http://www.w3.org/1999/xhtml"><body><p class="P"><span style="font-size:2.4em">H</span>'
                         '<span style="font-size:0.8em">AVING ANGUISHED AND LANGUISHED</span><span> till I could.</span></p></body></html>')
    assert merge_leadins(x) == 1
    p = x.find(".//{http://www.w3.org/1999/xhtml}p")
    markup = serialize_inline(p, True)[0]
    assert markup == "<t1>Having anguished and languished till I could.</t1>"
    assert "kt-leadin" in p.get("class")
    c = korean_css("p{text-align: justify; line-height: 120%; font-variant: small-caps; letter-spacing: 0.2mm; text-indent: 2.54em; font-family: Georgia, serif}")
    assert "justify" not in c and "line-height: 1.8" in c and "small-caps" not in c and "letter-spacing: normal" in c
    assert "text-indent: 1em" in c and "Noto Serif KR" in c
    assert "line-height: 1.75" in korean_css("p{line-height: 1.75}")
