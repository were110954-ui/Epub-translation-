"""Line and column detection for manga pages.

A text region is one horizontal line or one vertical column. Panel interiors and
balloon outlines are not text boxes. Long rules are removed before components are
measured, and only glyph-sized pieces are linked. This is a CPU heuristic, not a
trained detector.
"""
from __future__ import annotations



# The short side of a line box. Wider than this is a panel or a drawing, not a line.
_MAX_THICK = 46


def detect_text_lines(image, rtl: bool = True) -> list[dict]:
    """Return tight text lines in reading order. Each box is one line or one column."""
    from .ocr import _cc_boxes, order_manga_regions

    gray = image.convert("L")
    width, height = gray.size
    if width < 8 or height < 8:
        return []
    scale = 1.0
    work = gray
    if max(width, height) > 1400:
        scale = 1400 / max(width, height)
        work = gray.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    ww, hh = work.size
    mask = _ink_mask(work)
    _suppress_rules(mask, ww, hh, run=max(32, int(min(ww, hh) * 0.05)))
    parts = _cc_boxes(mask, ww, hh, min_area=8)
    glyphs = []
    ready = []
    for part in parts:
        kind = _part_kind(part, ww, hh)
        if kind == "glyph":
            glyphs.append(part)
        elif kind == "column":
            ready.append(_from_parts([part], "vertical", ww, hh, mask))
        elif kind == "row":
            ready.append(_from_parts([part], "horizontal", ww, hh, mask))
    claimed = []
    for group in _link(glyphs, "vertical"):
        if len(group) < 2:
            continue
        box = _from_parts(group, "vertical", ww, hh, mask)
        if box:
            ready.append(box)
            claimed.append(box)
    rest = [glyph for glyph in glyphs if not _consumed(glyph, claimed)]
    for group in _link(rest, "horizontal"):
        if len(group) < 2:
            continue
        box = _from_parts(group, "horizontal", ww, hh, mask)
        if box:
            ready.append(box)
    lines = [box for box in ready if box and _thin(box)]
    lines = _drop_overlaps(lines)
    if scale != 1.0:
        lines = [_scale_box(box, 1.0 / scale, width, height) for box in lines]
    lines = _mark_furigana(lines)
    ordered = order_manga_regions(lines, rtl=rtl, page_h=height)
    return _assign_groups(ordered)


def _ink_mask(gray):
    width, height = gray.size
    small = gray.resize((max(1, width // 16), max(1, height // 16)))
    blur = small.resize((width, height))
    src = gray.tobytes()
    bg = blur.tobytes()
    mask = bytearray(width * height)
    for i, (pix, back) in enumerate(zip(src, bg)):
        if pix < 150 and pix + 28 < back:
            mask[i] = 1
    return mask


def _suppress_rules(mask, width, height, run: int) -> None:
    """Clear long straight rules and a few pixels of their thickness. Glyphs stay."""
    horizontal = bytearray(width * height)
    vertical = bytearray(width * height)
    for y in range(height):
        row = y * width
        x = 0
        while x < width:
            if not mask[row + x]:
                x += 1
                continue
            x2 = x + 1
            while x2 < width and mask[row + x2]:
                x2 += 1
            if x2 - x >= run:
                for k in range(x, x2):
                    horizontal[row + k] = 1
            x = x2
    for x in range(width):
        y = 0
        while y < height:
            if not mask[y * width + x]:
                y += 1
                continue
            y2 = y + 1
            while y2 < height and mask[y2 * width + x]:
                y2 += 1
            if y2 - y >= run:
                for k in range(y, y2):
                    vertical[k * width + x] = 1
            y = y2
    for y in range(height):
        row = y * width
        for x in range(width):
            i = row + x
            if horizontal[i]:
                for dy in range(-2, 3):
                    yy = y + dy
                    if 0 <= yy < height:
                        mask[yy * width + x] = 0
            if vertical[i]:
                for dx in range(-2, 3):
                    xx = x + dx
                    if 0 <= xx < width:
                        mask[row + xx] = 0


def _part_kind(part, width, height) -> str:
    bw, bh = part["w"], part["h"]
    if bw < 4 or bh < 4:
        return "noise"
    if bw > width * 0.45 or bh > height * 0.45:
        return "noise"
    if bw > 70 and bh > 70:
        return "noise"
    if bh >= bw * 1.8 and 6 <= bw <= _MAX_THICK and bh >= 28:
        return "column"
    if bw >= bh * 1.8 and 6 <= bh <= _MAX_THICK and bw >= 28:
        return "row"
    if 5 <= bw <= 42 and 5 <= bh <= 42:
        return "glyph"
    return "noise"


def _link(glyphs, axis: str) -> list[list]:
    items = [dict(g) for g in glyphs]
    if axis == "vertical":
        items.sort(key=lambda g: (g["x"] + g["w"] / 2, g["y"]))
    else:
        items.sort(key=lambda g: (g["y"] + g["h"] / 2, g["x"]))
    used = [False] * len(items)
    groups = []
    for i, glyph in enumerate(items):
        if used[i]:
            continue
        group = [glyph]
        used[i] = True
        grew = True
        while grew:
            grew = False
            if axis == "vertical":
                cx = sum(g["x"] + g["w"] / 2 for g in group) / len(group)
                y0 = min(g["y"] for g in group)
                y1 = max(g["y"] + g["h"] for g in group)
                heights = sorted(g["h"] for g in group)
                widths = sorted(g["w"] for g in group)
                band = max(8, widths[len(widths) // 2]) * 0.7
                limit = max(6, int(heights[len(heights) // 2] * 0.95))
            else:
                cy = sum(g["y"] + g["h"] / 2 for g in group) / len(group)
                x0 = min(g["x"] for g in group)
                x1 = max(g["x"] + g["w"] for g in group)
                heights = sorted(g["h"] for g in group)
                widths = sorted(g["w"] for g in group)
                band = max(8, heights[len(heights) // 2]) * 0.7
                limit = max(6, int(widths[len(widths) // 2] * 0.8))
            for j, other in enumerate(items):
                if used[j]:
                    continue
                if axis == "vertical":
                    if abs((other["x"] + other["w"] / 2) - cx) > band:
                        continue
                    if other["y"] > y1:
                        gap = other["y"] - y1
                    elif other["y"] + other["h"] < y0:
                        gap = y0 - (other["y"] + other["h"])
                    else:
                        gap = 0
                else:
                    if abs((other["y"] + other["h"] / 2) - cy) > band:
                        continue
                    if other["x"] > x1:
                        gap = other["x"] - x1
                    elif other["x"] + other["w"] < x0:
                        gap = x0 - (other["x"] + other["w"])
                    else:
                        gap = 0
                if gap > limit:
                    continue
                group.append(other)
                used[j] = True
                grew = True
                if axis == "vertical":
                    y0 = min(y0, other["y"])
                    y1 = max(y1, other["y"] + other["h"])
                    cx = sum(g["x"] + g["w"] / 2 for g in group) / len(group)
                else:
                    x0 = min(x0, other["x"])
                    x1 = max(x1, other["x"] + other["w"])
                    cy = sum(g["y"] + g["h"] / 2 for g in group) / len(group)
        groups.append(group)
    return groups


def _consumed(glyph, lines) -> bool:
    cx = glyph["x"] + glyph["w"] / 2
    cy = glyph["y"] + glyph["h"] / 2
    for line in lines:
        if line["x"] <= cx <= line["x"] + line["w"] and line["y"] <= cy <= line["y"] + line["h"]:
            return True
    return False


def _from_parts(parts, direction: str, width, height, mask) -> dict | None:
    x0 = min(p["x"] for p in parts)
    y0 = min(p["y"] for p in parts)
    x1 = max(p["x"] + p["w"] for p in parts)
    y1 = max(p["y"] + p["h"] for p in parts)
    pad = 2
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    bw, bh = x1 - x0, y1 - y0
    if bw < 6 or bh < 6:
        return None
    if direction == "vertical" and bw > _MAX_THICK:
        return None
    if direction == "horizontal" and bh > _MAX_THICK:
        return None
    density = _box_density(mask, width, x0, y0, bw, bh)
    if density < 0.08 or density > 0.72:
        return None
    count = len(parts)
    minor = bw if direction == "vertical" else bh
    regular = _regular(parts, direction)
    flags = []
    long_enough = bh >= 48 if direction == "vertical" else bw >= 36
    if direction == "horizontal" and (bh < 14 or bh > 24 or count < 2):
        return None
    if direction == "vertical":
        drift = _column_drift(mask, width, x0, y0, bw, bh)
        if drift is not None and drift > bw * 0.16:
            return None
    if count >= 4 and regular and minor <= 30 and long_enough:
        confidence = 0.78
    elif count >= 3 and regular and minor <= 26 and long_enough:
        confidence = 0.7
    elif count == 1 and direction == "vertical" and bh >= bw * 4 and bw <= 26 and bh >= 48:
        confidence = 0.64
    elif count >= 2 and minor <= 28:
        confidence = 0.46
        flags.append("effect-uncertain")
    else:
        return None
    if x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1:
        flags.append("placement-uncertain")
        confidence = min(confidence, 0.45)
    return {
        "x": int(x0), "y": int(y0), "w": int(bw), "h": int(bh),
        "direction": direction, "confidence": confidence, "flags": flags, "parts": count,
    }


def _column_drift(mask, width, x, y, bw, bh):
    """How far the ink wanders sideways. A text column stays straighter than a vine."""
    centers = []
    for yy in range(y, y + bh):
        row = yy * width
        xs = [xx for xx in range(x, x + bw) if mask[row + xx]]
        if xs:
            centers.append(sum(xs) / len(xs))
    if len(centers) < 8:
        return None
    mean = sum(centers) / len(centers)
    var = sum((c - mean) ** 2 for c in centers) / len(centers)
    return var ** 0.5


def _box_density(mask, width, x, y, bw, bh) -> float:
    dark = 0
    area = bw * bh
    for yy in range(y, y + bh):
        row = yy * width
        dark += sum(mask[row + xx] for xx in range(x, x + bw))
    return dark / area if area else 0.0


def _regular(parts, direction: str) -> bool:
    if len(parts) <= 1:
        bw, bh = parts[0]["w"], parts[0]["h"]
        return (bh >= bw * 1.6 and bw <= _MAX_THICK) or (bw >= bh * 1.6 and bh <= _MAX_THICK)
    sizes = [p["h"] if direction == "vertical" else p["w"] for p in parts]
    sizes.sort()
    return sizes[-1] <= sizes[0] * 3.2 + 4


def _thin(box) -> bool:
    if box["direction"] == "vertical":
        return box["w"] <= _MAX_THICK and box["h"] >= box["w"]
    return box["h"] <= _MAX_THICK and box["w"] >= box["h"]


def _drop_overlaps(lines: list[dict]) -> list[dict]:
    """If a short fragment sits inside a longer line, keep the longer one."""
    ranked = sorted(lines, key=lambda b: b["w"] * b["h"], reverse=True)
    kept = []
    for line in ranked:
        cx = line["x"] + line["w"] / 2
        cy = line["y"] + line["h"] / 2
        inside = False
        for other in kept:
            if other["x"] <= cx <= other["x"] + other["w"] and other["y"] <= cy <= other["y"] + other["h"]:
                if line["direction"] == other["direction"]:
                    inside = True
                    break
        if not inside:
            kept.append(line)
    return kept


def _scale_box(box, inv, width, height) -> dict:
    x0 = max(0, min(width - 1, int(round(box["x"] * inv))))
    y0 = max(0, min(height - 1, int(round(box["y"] * inv))))
    x1 = max(x0 + 1, min(width, int(round((box["x"] + box["w"]) * inv))))
    y1 = max(y0 + 1, min(height, int(round((box["y"] + box["h"]) * inv))))
    out = dict(box)
    out.update({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out


def _mark_furigana(lines: list[dict]) -> list[dict]:
    """A much smaller line beside a column is ruby, not its own sentence."""
    for line in lines:
        if line["direction"] != "vertical" or line["h"] > 80:
            continue
        cx = line["x"] + line["w"] / 2
        for other in lines:
            if other is line or other["direction"] != "vertical" or other["h"] < line["h"] * 1.8:
                continue
            ox = other["x"] + other["w"] / 2
            y_overlap = min(line["y"] + line["h"], other["y"] + other["h"]) - max(line["y"], other["y"])
            if y_overlap < line["h"] * 0.5:
                continue
            if abs(cx - ox) <= other["w"] + line["w"] + 8:
                if "furigana-uncertain" not in line["flags"]:
                    line["flags"].append("furigana-uncertain")
                line["confidence"] = min(line["confidence"], 0.44)
                break
    return lines


def _assign_groups(lines: list[dict]) -> list[dict]:
    """Adjacent lines of one paragraph share a group. Reading order does not have to be consecutive."""
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def alone(line):
        return line.get("confidence", 0) < 0.55 or "effect-uncertain" in line["flags"] or "furigana-uncertain" in line["flags"]

    for i, left in enumerate(lines):
        if alone(left):
            continue
        for j in range(i + 1, len(lines)):
            right = lines[j]
            if alone(right):
                continue
            if _joins(left, right):
                parent[find(j)] = find(i)
    clusters: dict[int, list] = {}
    order = []
    for i, line in enumerate(lines):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
            order.append(root)
        clusters[root].append(line)
    out = []
    for index, root in enumerate(order):
        gid = f"g{index}"
        for line in clusters[root]:
            item = dict(line)
            item["group_id"] = gid
            out.append(item)
    for i, line in enumerate(out):
        line["idx"] = i
    return out


def _joins(prev, line) -> bool:
    if prev["direction"] != line["direction"]:
        return False
    if line["direction"] == "vertical":
        gap = line["x"] - (prev["x"] + prev["w"]) if line["x"] >= prev["x"] else prev["x"] - (line["x"] + line["w"])
        if gap < 0:
            gap = 0
        overlap = min(prev["y"] + prev["h"], line["y"] + line["h"]) - max(prev["y"], line["y"])
        return gap <= max(prev["w"], line["w"]) * 1.6 and overlap >= 0.45 * min(prev["h"], line["h"])
    gap = line["y"] - (prev["y"] + prev["h"]) if line["y"] >= prev["y"] else prev["y"] - (line["y"] + line["h"])
    if gap < 0:
        gap = 0
    overlap = min(prev["x"] + prev["w"], line["x"] + line["w"]) - max(prev["x"], line["x"])
    return gap <= max(prev["h"], line["h"]) * 1.25 and overlap >= 0.45 * min(prev["w"], line["w"])
