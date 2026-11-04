"""Create self-authored Japanese sample EPUBs (EPUB3 vertical + EPUB2) for testing. Text is original (written for this project)."""
import io
import os
import sys
import zipfile

from PIL import Image, ImageDraw

CH1 = [
    '<h1 id="ch1">第一章　<ruby>夏<rt>なつ</rt></ruby>の<ruby>駅<rt>えき</rt></ruby></h1>',
    '<p>　七月の終わり、<ruby>水瀬<rt>みなせ</rt></ruby><ruby>遥<rt>はるか</rt></ruby>は小さな無人駅のホームに立っていた。蝉の声がじりじりと耳に刺さる。</p>',
    '<p>「先輩、本当にここで合ってるんですか？」</p>',
    '<p>　後ろから声をかけてきたのは、後輩の<ruby>柏木<rt>かしわぎ</rt></ruby><ruby>蓮<rt>れん</rt></ruby>だった。彼はいつものように敬語で話す。</p>',
    '<p>「合ってるよ。地図には<em>ちゃんと</em>書いてあったんだから」</p>',
    '<p>　遥はそう言って、くしゃくしゃになった地図を広げた。そこには赤いペンで<span class="bouten">星</span>の印が描かれている<a epub:type="noteref" href="notes.xhtml#n1" id="r1">＊1</a>。</p>',
    '<p>「でも、電車はあと二時間来ませんよ」</p>',
    '<p>「じゃあ、歩けばいいじゃん」</p>',
    '<p class="center">◇</p>',
    '<p>　二人は線路沿いの道をゆっくりと歩き始めた。<br/>風が稲の穂をさらさらと揺らしている。</p>',
    '<div class="illus"><img src="../images/illust1.png" alt="駅のホーム"/></div>',
    '<p>　遥は振り返って、<ruby>蜃気楼<rt>ミラージュ</rt></ruby>のように揺れる駅舎を見つめた。</p>',
]
CH2 = [
    '<h1 id="ch2">第二章　星の<ruby>欠片<rt>かけら</rt></ruby></h1>',
    '<p>　夕方になって、二人はようやく丘の上の神社にたどり着いた。</p>',
    '<p>「柏木くん、見て。あれが地図の星だよ」</p>',
    '<p>「……ただの石じゃないですか、先輩」</p>',
    '<p>　蓮は<strong>呆れた</strong>ように溜息をついたが、その目は少しだけ笑っていた。</p>',
    '<p>　石の表面には、<ruby>流星<rt>りゅうせい</rt></ruby>の形をした模様がきらきらと光っていた。遥は思わずそれに手を伸ばす。</p>',
    '<p>「冷たい。……夏なのに、氷みたい」</p>',
    '<p>　その瞬間、どこかで鈴の音がちりんと鳴った<a epub:type="noteref" href="notes.xhtml#n2" id="r2">＊2</a>。</p>',
    '<p>　見上げると、空には一番星が浮かんでいた。<a href="chapter1.xhtml#ch1">第一章</a>で見た地図の印と、まったく同じ位置に。</p>',
]
NOTES = [
    '<h2>注</h2>',
    '<aside epub:type="footnote" id="n1"><p><a href="chapter1.xhtml#r1">＊1</a>　この地図は遥の祖父が残したものである。</p></aside>',
    '<aside epub:type="footnote" id="n2"><p><a href="chapter2.xhtml#r2">＊2</a>　神社の鈴は風がなくても鳴ると言われている。</p></aside>',
]
CSS = """@charset "utf-8";
html{ -epub-writing-mode: vertical-rl; -webkit-writing-mode: vertical-rl; writing-mode: vertical-rl; }
body{ font-family: serif; line-height: 1.8; }
h1{ font-size: 1.4em; }
.center{ text-align: center; }
.bouten{ -epub-text-emphasis-style: filled sesame; text-emphasis-style: filled sesame; }
.tcy{ -epub-text-combine: horizontal; -webkit-text-combine: horizontal; text-combine-upright: all; }
.illus img{ max-width: 100%; }
"""


def xhtml(title, body, v3=True):
    ns = ' xmlns:epub="http://www.idpf.org/2007/ops"' if v3 else ""
    head = '<!DOCTYPE html>\n' if v3 else '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
    if not v3:
        body = [b.replace(' epub:type="noteref"', "").replace(' epub:type="footnote"', "").replace("<aside", "<div").replace("</aside>", "</div>") for b in body]
        import re as _re
        body = [_re.sub(r"<ruby>(.*?)<rt>.*?</rt></ruby>", r"\1", b) for b in body]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
{head}<html xmlns="http://www.w3.org/1999/xhtml"{ns} xml:lang="ja"{' lang="ja" class="vrtl"' if v3 else ''}>
<head><title>{title}</title><link rel="stylesheet" type="text/css" href="../styles/style.css"/></head>
<body>
{chr(10).join(body)}
</body>
</html>
"""


def png(text_color, size, label):
    im = Image.new("RGB", size, (30, 60, 110))
    d = ImageDraw.Draw(im)
    d.ellipse((size[0] // 3, size[1] // 5, size[0] // 3 + 120, size[1] // 5 + 120), fill=(250, 220, 90))
    d.text((20, size[1] - 40), label, fill=text_color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def make(path, v3=True):
    title = "星の駅（サンプル）"
    uid = "urn:uuid:3f1c7a52-6a1e-4b3e-9d55-0a1b2c3d4e5f" if v3 else "urn:uuid:3f1c7a52-6a1e-4b3e-9d55-0a1b2c3d4e60"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        D = zipfile.ZIP_DEFLATED
        z.writestr("META-INF/container.xml", """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>""", D)
        z.writestr("OEBPS/styles/style.css", CSS, D)
        z.writestr("OEBPS/images/cover.png", png((255, 255, 255), (600, 800), "HOSHI NO EKI - sample cover"), D)
        z.writestr("OEBPS/images/illust1.png", png((255, 255, 255), (400, 300), "illustration"), D)
        cover_body = ['<div class="cover"><img src="../images/cover.png" alt="表紙"/></div>']
        z.writestr("OEBPS/text/cover.xhtml", xhtml("表紙", cover_body, v3), D)
        z.writestr("OEBPS/text/chapter1.xhtml", xhtml("第一章　夏の駅", CH1, v3), D)
        z.writestr("OEBPS/text/chapter2.xhtml", xhtml("第二章　星の欠片", CH2, v3), D)
        z.writestr("OEBPS/text/notes.xhtml", xhtml("注", NOTES, v3), D)
        z.writestr("OEBPS/toc.ncx", f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="ja">
<head><meta name="dtb:uid" content="{uid}"/><meta name="dtb:depth" content="1"/><meta name="dtb:totalPageCount" content="0"/><meta name="dtb:maxPageNumber" content="0"/></head>
<docTitle><text>{title}</text></docTitle>
<navMap>
<navPoint id="np1" playOrder="1"><navLabel><text>表紙</text></navLabel><content src="text/cover.xhtml"/></navPoint>
<navPoint id="np2" playOrder="2"><navLabel><text>第一章　夏の駅</text></navLabel><content src="text/chapter1.xhtml#ch1"/></navPoint>
<navPoint id="np3" playOrder="3"><navLabel><text>第二章　星の欠片</text></navLabel><content src="text/chapter2.xhtml#ch2"/></navPoint>
<navPoint id="np4" playOrder="4"><navLabel><text>注</text></navLabel><content src="text/notes.xhtml"/></navPoint>
</navMap></ncx>""", D)
        nav_item = ""
        if v3:
            z.writestr("OEBPS/nav.xhtml", """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="ja" lang="ja">
<head><title>目次</title><link rel="stylesheet" type="text/css" href="styles/style.css"/></head>
<body><nav epub:type="toc" id="toc"><h1>目次</h1><ol>
<li><a href="text/cover.xhtml">表紙</a></li>
<li><a href="text/chapter1.xhtml#ch1">第一章　夏の駅</a></li>
<li><a href="text/chapter2.xhtml#ch2">第二章　星の欠片</a></li>
<li><a href="text/notes.xhtml">注</a></li></ol></nav>
<nav epub:type="landmarks" hidden=""><ol><li><a epub:type="cover" href="text/cover.xhtml">表紙</a></li><li><a epub:type="bodymatter" href="text/chapter1.xhtml">本文</a></li></ol></nav>
</body></html>""", D)
            nav_item = '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        meta3 = '<meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>' if v3 else '<meta name="cover" content="cover-img"/>'
        cover_prop = ' properties="cover-image"' if v3 else ""
        z.writestr("OEBPS/content.opf", f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="{'3.0' if v3 else '2.0'}" unique-identifier="bookid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
<dc:identifier id="bookid">{uid}</dc:identifier><dc:title>{title}</dc:title><dc:language>ja</dc:language>
<dc:creator>EPUB AI Translator テスト</dc:creator>{meta3}
<meta name="primary-writing-mode" content="vertical-rl"/>
</metadata>
<manifest>
{nav_item}
<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
<item id="css" href="styles/style.css" media-type="text/css"/>
<item id="cover-img" href="images/cover.png" media-type="image/png"{cover_prop}/>
<item id="illust1" href="images/illust1.png" media-type="image/png"/>
<item id="cover" href="text/cover.xhtml" media-type="application/xhtml+xml"/>
<item id="ch1" href="text/chapter1.xhtml" media-type="application/xhtml+xml"/>
<item id="ch2" href="text/chapter2.xhtml" media-type="application/xhtml+xml"/>
<item id="notes" href="text/notes.xhtml" media-type="application/xhtml+xml"/>
</manifest>
<spine toc="ncx"{' page-progression-direction="rtl"' if v3 else ''}>
<itemref idref="cover" linear="no"/><itemref idref="ch1"/><itemref idref="ch2"/><itemref idref="notes"/>
</spine>
{'<guide><reference type="cover" title="表紙" href="text/cover.xhtml"/></guide>' if not v3 else ''}
</package>""", D)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "samples"
    os.makedirs(out, exist_ok=True)
    make(os.path.join(out, "sample_ja_epub3.epub"), True)
    make(os.path.join(out, "sample_ja_epub2.epub"), False)
    print("written to", out)
