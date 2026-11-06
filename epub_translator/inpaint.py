"""Background restoration for speech balloons.

This is not generative inpainting. Flat balloon interiors lose dark strokes (the ink).
Patterned or uneven areas are covered with one estimated color and flagged for review.
The original file is never modified; callers pass a working copy.
"""
from __future__ import annotations


def restore_regions(image, boxes: list[tuple[int, int, int, int]]) -> tuple[object, list[dict]]:
    from PIL import Image, ImageDraw
    base = image.convert("RGB")
    out = base.copy() if base is image else base
    notes = []
    for box in boxes:
        notes.append(_restore_one(out, box, Image, ImageDraw))
    return out, notes


def _restore_one(image, box, Image, ImageDraw) -> dict:
    x, y, w, h = [int(v) for v in box]
    inset = 3
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
    if var > 900:
        draw = ImageDraw.Draw(image)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=color)
        return {"method": "flat-plate", "confident": False,
                "note": "테두리 색이 고르지 않아 단색 플레이트로 가렸습니다. 생성형 인페인트가 아니며 무늬는 복원되지 않습니다."}
    luma = sum(color) / 3.0
    mask = crop.convert("L").point(lambda p, t=luma - 40: 255 if p < t else 0)
    plate = Image.new("RGB", crop.size, color)
    cleaned = Image.composite(plate, crop, mask)
    image.paste(cleaned, (x1, y1))
    return {"method": "flat-fill", "confident": True,
            "note": "평탄한 말풍선 안의 어두운 획만 추정 배경색으로 바꾸었습니다. 그라데이션은 복원되지 않습니다."}
