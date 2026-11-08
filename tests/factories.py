"""Synthetic fixtures. No copyrighted pages."""
from __future__ import annotations
import asyncio
import io
import json
import os
import zipfile


def run_job(jm, job_id):
    async def go():
        await jm.start(job_id)
    asyncio.run(go())

from PIL import Image, ImageDraw, PngImagePlugin


def jpeg_bytes(w, h, color=(20, 40, 80), quality=90) -> bytes:
    im = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(w, h, color=(10, 120, 40)) -> bytes:
    im = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def build_image_pdf(pages: list[tuple[bytes, int, int, float, float]]) -> bytes:
    """One JPEG XObject per page, scaled to the media box. `pages` is (jpeg, px_w, px_h, pt_w, pt_h)."""
    blobs = {}
    pages_id = 2
    next_id = 3
    kids = []
    for jpeg, w, h, pw, ph in pages:
        page_id, cont_id, img_id = next_id, next_id + 1, next_id + 2
        next_id += 3
        content = f"q {pw:.4f} 0 0 {ph:.4f} 0 0 cm /Im0 Do Q\n".encode()
        blobs[page_id] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
            f"/Resources << /XObject << /Im0 {img_id} 0 R >> >> /Contents {cont_id} 0 R >>"
        ).encode()
        blobs[cont_id] = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream"
        blobs[img_id] = (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
            f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n".encode()
            + jpeg + b"\nendstream"
        )
        kids.append(page_id)
    kids_s = " ".join(f"{k} 0 R" for k in kids)
    blobs[pages_id] = f"<< /Type /Pages /Count {len(kids)} /Kids [{kids_s}] >>".encode()
    blobs[1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    return _serialize(blobs)


def build_text_pdf(text: str = "Hello", pw: float = 72, ph: float = 72) -> bytes:
    content = f"BT /F1 12 Tf 10 40 Td ({text}) Tj ET\n".encode()
    blobs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        3: (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode(),
        4: f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    return _serialize(blobs)


def build_mixed_pdf(jpeg: bytes, w: int, h: int) -> bytes:
    """Page 1 is a full-page JPEG. Page 2 is text and must be rendered."""
    content_img = b"q 300.0000 0 0 400.0000 0 0 cm /Im0 Do Q\n"
    content_text = b"BT /F1 12 Tf 20 40 Td (Hello) Tj ET\n"
    blobs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Count 2 /Kids [3 0 R 6 0 R] >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
        4: f"<< /Length {len(content_img)} >>\nstream\n".encode() + content_img + b"endstream",
        5: (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
            f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n".encode()
            + jpeg + b"\nendstream"
        ),
        6: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Resources << /Font << /F1 8 0 R >> >> /Contents 7 0 R >>",
        7: f"<< /Length {len(content_text)} >>\nstream\n".encode() + content_text + b"endstream",
        8: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    return _serialize(blobs)


def _serialize(blobs: dict[int, bytes]) -> bytes:
    maxid = max(blobs)
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (maxid + 1)
    for i in range(1, maxid + 1):
        offsets[i] = len(out)
        out += f"{i} 0 obj\n".encode() + blobs[i] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {maxid + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for i in range(1, maxid + 1):
        out += f"{offsets[i]:010d} 00000 n \n".encode()
    out += f"trailer << /Size {maxid + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def write_cbz(path: str, members: list[tuple[str, bytes]]):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members:
            z.writestr(name, data)


def balloon_page(path: str) -> list[dict]:
    """Gray page, three outlined balloons, fixture OCR metadata. Not a real scan."""
    regions = [
        {"x": 120, "y": 36, "w": 90, "h": 64, "text": "「こんにちは、今日はいい天気ですね。」", "direction": "horizontal", "confidence": 0.96},
        {"x": 130, "y": 190, "w": 80, "h": 58, "text": "「塔野は笑った。」", "direction": "horizontal", "confidence": 0.93},
        {"x": 24, "y": 188, "w": 78, "h": 56, "text": "「本当？」", "direction": "horizontal", "confidence": 0.42},
    ]
    im = Image.new("RGB", (240, 320), (188, 188, 188))
    draw = ImageDraw.Draw(im)
    for region in regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        draw.ellipse((x - 3, y - 3, x + w + 3, y + h + 3), fill="black")
        draw.ellipse((x, y, x + w, y + h), fill="white")
    info = PngImagePlugin.PngInfo()
    info.add_text("epub-ocr-fixture", json.dumps({"regions": regions}, ensure_ascii=False))
    im.save(path, pnginfo=info)
    return regions


def huge_png_header(w=100000, h=100000) -> bytes:
    import struct
    import zlib

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)


def solid_png(path, w, h, color):
    Image.new("RGB", (w, h), color).save(path)
    return open(path, "rb").read()
