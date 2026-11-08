"""Line-level manga detection. Synthetic pages only. No paid translation API."""
from PIL import Image, ImageDraw

from epub_translator.fxl import layout_korean_text
from epub_translator.inpaint import restore_regions
from epub_translator.ocr import detect_balloons


def _glyphs(draw, boxes):
    for x, y, w, h in boxes:
        draw.rectangle((x, y, x + w, y + h), fill="black")


def test_horizontal_line_is_one_tight_box():
    im = Image.new("RGB", (280, 80), "white")
    draw = ImageDraw.Draw(im)
    for i in range(5):
        _glyphs(draw, [(20 + i * 28, 30, 16, 16)])
    found = detect_balloons(im, rtl=True)
    assert len(found) == 1
    line = found[0]
    assert line["direction"] == "horizontal"
    assert line["h"] <= 28 and line["w"] >= 100
    assert line["y"] < 40 and line["y"] + line["h"] < 60


def test_vertical_columns_are_separate_and_right_to_left():
    im = Image.new("RGB", (180, 220), "white")
    draw = ImageDraw.Draw(im)
    for x in (120, 80):
        for i in range(5):
            _glyphs(draw, [(x, 20 + i * 30, 14, 16)])
    found = detect_balloons(im, rtl=True)
    vertical = [b for b in found if b["direction"] == "vertical"]
    assert len(vertical) == 2
    assert all(b["w"] <= 30 for b in vertical)
    assert vertical[0]["x"] > vertical[1]["x"]
    assert vertical[0]["group_id"] == vertical[1]["group_id"]


def test_panel_border_is_not_a_text_line():
    im = Image.new("RGB", (200, 160), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle((6, 6, 190, 150), outline="black", width=4)
    found = detect_balloons(im, rtl=True)
    assert found == []


def test_uncertain_speck_is_not_erased_and_source_file_stays(tmp_path):
    im = Image.new("RGB", (80, 80), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle((10, 10, 16, 16), fill="black")
    draw.rectangle((40, 40, 70, 48), fill=(20, 20, 20))
    src = tmp_path / "page.png"
    im.save(src)
    before = src.read_bytes()
    outside = im.getpixel((4, 4))
    found = detect_balloons(im, rtl=True)
    assert all(b["confidence"] < 0.55 or "effect-uncertain" in b["flags"] for b in found) or found == []
    out, notes = restore_regions(im, [{"x": 36, "y": 36, "w": 40, "h": 16, "erase": False}])
    assert notes[0]["method"] == "skipped-review"
    assert out.getpixel((50, 44)) == (20, 20, 20)
    assert out.getpixel((4, 4)) == outside
    assert src.read_bytes() == before


def test_text_mask_stays_inside_the_line_and_skips_a_solid_blob(tmp_path):
    im = Image.new("RGB", (90, 70), (250, 250, 250))
    draw = ImageDraw.Draw(im)
    draw.rectangle((30, 20, 38, 48), fill=(0, 0, 0))
    im.putpixel((2, 2), (255, 0, 0))
    src = tmp_path / "page.png"
    im.save(src)
    before = src.read_bytes()
    out, notes = restore_regions(im, [(24, 14, 20, 40)])
    assert notes[0]["method"] == "text-mask"
    assert out.getpixel((34, 30))[0] > 200
    assert out.getpixel((2, 2)) == (255, 0, 0)
    assert src.read_bytes() == before
    solid = Image.new("RGB", (60, 60), "white")
    ImageDraw.Draw(solid).rectangle((8, 8, 50, 50), fill="black")
    kept = solid.copy()
    _out, solid_notes = restore_regions(solid, [(4, 4, 52, 52)])
    assert solid_notes[0]["method"] == "skipped-uncertain"
    assert solid.getpixel((20, 20)) == kept.getpixel((20, 20))


def test_distant_column_is_another_group():
    im = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(im)
    for x in (180, 150):
        for i in range(4):
            _glyphs(draw, [(x, 16 + i * 28, 12, 14)])
    for i in range(4):
        _glyphs(draw, [(20, 16 + i * 28, 12, 14)])
    found = [b for b in detect_balloons(im, rtl=True) if b["confidence"] >= 0.55]
    right = [b for b in found if b["x"] > 100]
    left = [b for b in found if b["x"] < 60]
    assert right and left
    assert len({b["group_id"] for b in right}) == 1
    assert right[0]["group_id"] != left[0]["group_id"]


def test_korean_layout_flags_overflow():
    fitted = layout_korean_text("짧은 대사", 180, 80, False)
    assert fitted["overflow"] is False and fitted["font_size"] >= 10
    overflow = layout_korean_text("가" * 40, 24, 24, False)
    assert overflow["overflow"] is True and overflow["font_size"] == 8
    wrapped = layout_korean_text("가나다라마바사아자차", 80, 60, False)
    assert wrapped["overflow"] is False and "\n" in wrapped["text"]
