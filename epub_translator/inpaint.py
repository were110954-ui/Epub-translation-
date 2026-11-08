"""Background restoration for text lines.

This is not generative inpainting. Only dark strokes inside a line box are replaced.
Uneven areas and low-confidence boxes keep their original pixels.
The caller's file is never modified; this function edits the image it is given.
"""
from __future__ import annotations


def restore_regions(image, boxes) -> tuple[object, list[dict]]:
    """Erase only dark strokes inside a box. A dict with erase=False is left untouched."""
    from PIL import Image, ImageDraw
    base = image.convert("RGB")
    out = base.copy() if base is image else base
    notes = []
    for box in boxes:
        if isinstance(box, dict):
            rect = (int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"]))
            if box.get("erase") is False:
                notes.append({
                    "method": "skipped-review", "confident": False,
                    "note": "확신이 낮거나 효과·배치가 불확실해 원본 픽셀을 유지했습니다.",
                })
                continue
        else:
            rect = tuple(int(v) for v in box)
        notes.append(_restore_one(out, rect, Image, ImageDraw))
    return out, notes


def _restore_one(image, box, Image, ImageDraw) -> dict:
    x, y, w, h = [int(v) for v in box]
    inset = 1
    x1, y1 = x + inset, y + inset
    x2, y2 = x + w - inset, y + h - inset
    if x2 - x1 < 6 or y2 - y1 < 6:
        return {"method": "skipped-small", "confident": False,
                "note": "영역이 너무 작아 배경을 건드리지 않았습니다."}
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return {"method": "skipped-small", "confident": False,
                "note": "영역이 페이지 밖으로 나가 배경을 건드리지 않았습니다."}
    crop = image.crop((x1, y1, x2, y2))
    border = []
    px = crop.load()
    cw, ch = crop.size
    for i in range(cw):
        border.append(px[i, 0])
        border.append(px[i, ch - 1])
    for j in range(ch):
        border.append(px[0, j])
        border.append(px[cw - 1, j])
    step = max(1, len(border) // 400)
    sample = border[::step] or border
    mean = tuple(sum(p[c] for p in sample) / len(sample) for c in range(3))
    var = sum(sum((p[c] - mean[c]) ** 2 for c in range(3)) for p in sample) / len(sample)
    color = tuple(max(0, min(255, int(round(v)))) for v in mean)
    luma = sum(color) / 3.0
    mask = crop.convert("L").point(lambda p, t=luma - 40: 255 if p < t else 0)
    dark = mask.histogram()[255]
    if dark > 0.45 * mask.size[0] * mask.size[1]:
        return {"method": "skipped-uncertain", "confident": False,
                "note": "어두운 부분이 글자 획보다 넓어 원본 픽셀을 유지했습니다."}
    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.MaxFilter(3))
    if mask.getextrema()[1] == 0:
        return {"method": "text-mask-empty", "confident": True,
                "note": "지울 획이 없어 원본 픽셀을 유지했습니다."}
    plate = Image.new("RGB", crop.size, color)
    cleaned = Image.composite(plate, crop, mask)
    image.paste(cleaned, (x1, y1))
    return {"method": "text-mask", "confident": True,
            "note": "글자 마스크만 팽창해 주변 배경색으로 채웠습니다. 칸 전체를 칠하지 않습니다."}
