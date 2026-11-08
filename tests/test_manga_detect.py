"""Automatic manga layout. No copyrighted pages and no paid translation API."""
from PIL import Image, ImageDraw

from epub_translator.fxl import layout_korean_text
from epub_translator.inpaint import restore_regions
from epub_translator.ocr import detect_balloons, order_manga_regions


def test_vertical_columns_read_right_to_left():
    regions = [
        {"x": 12, "y": 20, "w": 18, "h": 90},
        {"x": 70, "y": 16, "w": 18, "h": 100},
        {"x": 30, "y": 180, "w": 40, "h": 24},
    ]
    ordered = order_manga_regions(regions, rtl=True, page_h=280)
    assert [r["x"] for r in ordered[:2]] == [70, 12]
    assert ordered[2]["y"] == 180


def test_ink_inside_balloon_marks_vertical_and_outside_text_is_review():
    im = Image.new("RGB", (220, 160), (180, 180, 180))
    draw = ImageDraw.Draw(im)
    draw.ellipse((120, 20, 200, 140), fill="white", outline="black", width=3)
    for x in (150, 165, 180):
        draw.rectangle((x, 40, x + 4, 110), fill="black")
    draw.rectangle((20, 30, 28, 90), fill="black")
    found = detect_balloons(im, rtl=True)
    balloons = [b for b in found if "outside-balloon" not in b["flags"]]
    outside = [b for b in found if "outside-balloon" in b["flags"]]
    assert balloons and any(b["direction"] == "vertical" for b in balloons)
    assert outside
    assert all(b["confidence"] <= 0.55 for b in outside)


def test_text_mask_clears_ink_without_touching_the_source_file(tmp_path):
    im = Image.new("RGB", (120, 80), (248, 248, 248))
    draw = ImageDraw.Draw(im)
    draw.rectangle((50, 28, 58, 54), fill=(10, 10, 10))
    src = tmp_path / "page.png"
    im.save(src)
    before = src.read_bytes()
    out, notes = restore_regions(im, [(8, 8, 104, 64)])
    assert notes[0]["method"] == "text-mask"
    assert src.read_bytes() == before
    assert out.getpixel((54, 40))[0] > 200


def test_korean_layout_flags_overflow():
    fitted = layout_korean_text("짧은 대사", 180, 80, False)
    assert fitted["overflow"] is False and fitted["font_size"] >= 10
    overflow = layout_korean_text("가" * 40, 24, 24, False)
    assert overflow["overflow"] is True and overflow["font_size"] == 8
    vertical = layout_korean_text("가나다라마바", 40, 120, True)
    assert "\n" in vertical["text"] or len(vertical["text"]) == 6
