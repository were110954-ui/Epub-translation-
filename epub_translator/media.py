"""Image header parsing (no full decode) and zip-name checks."""
from __future__ import annotations
import os
import re
import struct

from .safety import HEADER_READ, SafetyError, check_pixels

_NATURAL = re.compile(r"(\d+)")


def natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in _NATURAL.split(name)]


def safe_zip_name(name: str) -> str:
    name = (name or "").replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise SafetyError(f"절대 경로는 허용되지 않습니다: {name}")
    parts = []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise SafetyError(f"경로 탈출은 허용되지 않습니다: {name}")
        parts.append(part)
    if not parts:
        raise SafetyError("빈 경로입니다.")
    return "/".join(parts)


def sniff_bytes(data: bytes) -> str | None:
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def ext_for(media_type: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(media_type, ".img")


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i < n - 8:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            return None
        seglen = int.from_bytes(data[i:i + 2], "big")
        if seglen < 2:
            return None
        if marker == 0xDA:
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 7 >= n:
                return None
            h = int.from_bytes(data[i + 3:i + 5], "big")
            w = int.from_bytes(data[i + 5:i + 7], "big")
            return w, h
        i += seglen
    return None


def png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)


def webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 16 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    i = 12
    while i + 8 <= len(data):
        fourcc = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        payload = i + 8
        if fourcc == b"VP8X" and payload + 10 <= len(data):
            w = 1 + int.from_bytes(data[payload + 4:payload + 7], "little")
            h = 1 + int.from_bytes(data[payload + 7:payload + 10], "little")
            return w, h
        if fourcc == b"VP8 " and payload + 10 <= len(data) and data[payload + 3:payload + 6] == b"\x9d\x01\x2a":
            w = int.from_bytes(data[payload + 6:payload + 8], "little") & 0x3FFF
            h = int.from_bytes(data[payload + 8:payload + 10], "little") & 0x3FFF
            return w, h
        if fourcc == b"VP8L" and payload + 5 <= len(data) and data[payload] == 0x2F:
            bits = int.from_bytes(data[payload + 1:payload + 5], "little")
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return w, h
        i = payload + size + (size & 1)
    return None


def image_info(data: bytes) -> tuple[str, int, int] | None:
    kind = sniff_bytes(data)
    if kind == "image/jpeg":
        dims = jpeg_size(data)
    elif kind == "image/png":
        dims = png_size(data)
    elif kind == "image/webp":
        dims = webp_size(data)
    else:
        return None
    if not dims:
        return None
    return kind, dims[0], dims[1]


def image_info_file(path: str) -> tuple[str, int, int]:
    """Read only the header when possible. JPEG SOF markers can sit after a large APP segment."""
    with open(path, "rb") as f:
        head = f.read(HEADER_READ)
    kind = sniff_bytes(head)
    dims = None
    if kind == "image/png":
        dims = png_size(head)
    elif kind == "image/webp":
        dims = webp_size(head)
    elif kind == "image/jpeg":
        dims = jpeg_size(head)
        if not dims:
            with open(path, "rb") as f:
                dims = jpeg_size(f.read())
    if not kind or not dims:
        raise SafetyError(f"지원하는 이미지 헤더가 아닙니다 (JPG/PNG/WebP): {os.path.basename(path)}")
    check_pixels(dims[0], dims[1])
    return kind, dims[0], dims[1]


def jpeg_payload(data: bytes) -> bytes | None:
    """Return the original JPEG bitstream (SOI through EOI) without re-encoding."""
    start = data.find(b"\xff\xd8")
    if start < 0 or start > 16:
        return None
    end = data.rfind(b"\xff\xd9")
    if end < start:
        return None
    return data[start:end + 2]
