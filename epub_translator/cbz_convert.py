"""Safe CBZ/ZIP image listing and single-member extraction.

The archive is never rewritten. Members are checked for traversal, declared size,
compression ratio, and image-header dimensions before Pillow is allowed to decode anything.
"""
from __future__ import annotations
import hashlib
import os
import zipfile

from .epub_model import DRMError
from .media import ext_for, image_info, natural_key, safe_zip_name
from .safety import (
    HEADER_READ, MAX_FILES, MAX_PAGES, MAX_RATIO, MAX_SINGLE, MAX_UNCOMPRESSED, SafetyError, check_pixels,
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_SKIP = ("__MACOSX/", ".DS_Store", "Thumbs.db")


def _skip(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
        return True
    if base in (".DS_Store", "Thumbs.db") or base.startswith("._"):
        return True
    return False


def scan_cbz(path: str) -> list[dict]:
    if not zipfile.is_zipfile(path):
        raise SafetyError("CBZ/ZIP 아카이브가 아닙니다.")
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise SafetyError(f"ZIP을 열 수 없습니다: {e}") from e
    planned = []
    try:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_FILES:
            raise SafetyError(f"ZIP 항목 수가 한도({MAX_FILES})를 넘습니다.")
        total = 0
        kept = []
        for info in infos:
            if info.flag_bits & 0x1:
                raise DRMError("암호화된 ZIP/CBZ는 처리하지 않습니다. 암호를 우회하지 않습니다.")
            name = safe_zip_name(info.filename)
            if _skip(name):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXT:
                continue
            if info.file_size <= 0:
                raise SafetyError(f"빈 이미지입니다: {name}")
            if info.file_size > MAX_SINGLE:
                raise SafetyError(f"이미지가 너무 큽니다 ({info.file_size} bytes): {name}")
            if info.compress_size:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_RATIO and info.file_size > 100_000:
                    raise SafetyError(f"압축률이 비정상입니다 (zip bomb 의심): {name}")
            total += info.file_size
            if total > MAX_UNCOMPRESSED:
                raise SafetyError("압축 해제 총량이 한도를 넘습니다.")
            kept.append((name, info))
        if not kept:
            raise SafetyError("CBZ 안에 JPG, PNG, WebP 이미지가 없습니다.")
        if len(kept) > MAX_PAGES:
            raise SafetyError(f"페이지 수가 한도({MAX_PAGES})를 넘습니다.")
        kept.sort(key=lambda item: natural_key(item[0]))
        for idx, (name, info) in enumerate(kept):
            with zf.open(info, "r") as src:
                head = src.read(min(HEADER_READ, info.file_size))
            parsed = image_info(head)
            if not parsed:
                raise SafetyError(f"이미지 헤더를 확인할 수 없습니다: {name}")
            media, w, h = parsed
            check_pixels(w, h)
            planned.append({
                "index": idx, "src_name": name, "width": w, "height": h,
                "media_type": media, "byte_size": info.file_size,
            })
    finally:
        zf.close()
    return planned


def extract_cbz_member(path: str, src_name: str, dest_dir: str, page_id: str) -> dict:
    """Stream one member to disk. Stops if the decompressed size exceeds the declared size or the cap."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        match = None
        for info in zf.infolist():
            if info.is_dir():
                continue
            if safe_zip_name(info.filename) == src_name:
                match = info
                break
        if match is None:
            raise SafetyError(f"아카이브에서 이미지를 찾지 못했습니다: {src_name}")
        if match.flag_bits & 0x1:
            raise DRMError("암호화된 ZIP 항목입니다.")
        if match.file_size > MAX_SINGLE:
            raise SafetyError("이미지가 크기 한도를 넘습니다.")
        # Extension follows the bytes we will sniff after the write, but we need a name first.
        dest = os.path.join(dest_dir, page_id + os.path.splitext(src_name)[1].lower().replace(".jpeg", ".jpg"))
        digest = hashlib.sha256()
        written = 0
        with zf.open(match, "r") as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > match.file_size + 1024 or written > MAX_SINGLE:
                    out.close()
                    os.remove(dest)
                    raise SafetyError(f"압축 해제 크기가 선언값 또는 한도를 넘습니다: {src_name}")
                digest.update(chunk)
                out.write(chunk)
    from .media import image_info_file
    media, w, h = image_info_file(dest)
    final = os.path.join(dest_dir, page_id + ext_for(media))
    if final != dest:
        os.replace(dest, final)
        dest = final
    return {"path": dest, "width": w, "height": h, "media_type": media, "method": "copy",
            "byte_size": written, "sha256": digest.hexdigest(),
            "note": "원본 이미지 바이트를 재압축하지 않고 복사했습니다."}
