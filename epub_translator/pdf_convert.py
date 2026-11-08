"""PDF page intake.

Embedded images are copied out of the PDF stream when one displayed image is the whole page
(no text layer, no rotation). That path does not re-encode JPEG. Anything else is rendered at
a caller-chosen 300–600 DPI, which is a new bitmap and is reported as such.
Encrypted PDFs are refused. There is no password search.
"""
from __future__ import annotations
import os
import re
import shutil

from .media import ext_for, jpeg_payload
from .safety import MAX_SINGLE, RenderUnavailable, SafetyError, check_pixels


class PdfEncrypted(Exception):
    pass


def _reader(path: str):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise SafetyError("PDF를 읽으려면 pypdf가 필요합니다. pip install pypdf") from e
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise PdfEncrypted("암호화된 PDF는 열지 않습니다. 암호를 우회하거나 제거하지 않습니다.")
    return reader


def inspect_pdf(path: str) -> list[dict]:
    reader = _reader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            box = page.mediabox
            pw, ph = float(box.width), float(box.height)
        except Exception:
            pw, ph = 0.0, 0.0
        pages.append({"index": i, "src_name": f"page-{i + 1}", "width_pt": pw, "height_pt": ph})
    if not pages:
        raise SafetyError("PDF에 페이지가 없습니다.")
    from .safety import MAX_PAGES
    if len(pages) > MAX_PAGES:
        raise SafetyError(f"페이지 수가 한도({MAX_PAGES})를 넘습니다.")
    return pages


def _content_bytes(page) -> bytes:
    try:
        contents = page.get_contents()
    except Exception:
        return b""
    if contents is None:
        return b""
    if isinstance(contents, list):
        parts = []
        for c in contents:
            try:
                parts.append(c.get_data() or b"")
            except Exception:
                continue
        return b"\n".join(parts)
    try:
        return contents.get_data() or b""
    except Exception:
        return b""


def _page_text(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _rotation(page) -> int:
    try:
        return int(page.rotation or 0) % 360
    except Exception:
        return 0


def _iter_images(page):
    resources = page.get("/Resources")
    if resources is None:
        return
    resources = resources.get_object()
    xobj = resources.get("/XObject")
    if xobj is None:
        return
    xobj = xobj.get_object()
    content = _content_bytes(page)
    for name, ref in xobj.items():
        try:
            obj = ref.get_object()
        except Exception:
            continue
        if str(obj.get("/Subtype") or "") != "/Image":
            continue
        if obj.get("/ImageMask"):
            continue
        try:
            w, h = int(obj.get("/Width")), int(obj.get("/Height"))
        except (TypeError, ValueError):
            continue
        token = str(name).encode("latin1", "ignore")
        displayed = re.search(re.escape(token) + rb"\s+Do", content) is not None
        yield str(name), obj, w, h, displayed


def _filters(obj) -> str:
    f = obj.get("/Filter")
    if f is None:
        return ""
    if isinstance(f, list):
        return " ".join(str(x) for x in f)
    return str(f)


def _extract_original(obj, w: int, h: int) -> tuple[bytes, str, str] | None:
    """Return (bytes, media_type, method) without lossy recompression, or None."""
    check_pixels(w, h)
    try:
        raw = obj.get_data()
    except Exception:
        return None
    if not isinstance(raw, (bytes, bytearray)):
        return None
    raw = bytes(raw)
    if len(raw) > MAX_SINGLE:
        raise SafetyError("임베디드 이미지가 크기 한도를 넘습니다.")
    filters = _filters(obj)
    if "DCT" in filters:
        jpg = jpeg_payload(raw)
        if jpg:
            return jpg, "image/jpeg", "extract-jpeg"
    return None


def choose_direct(page) -> tuple[bytes, str, str, int, int] | None:
    """A page is directly extractable when it shows exactly one JPEG and no real text, unrotated."""
    if _rotation(page):
        return None
    if len("".join(_page_text(page).split())) > 8:
        return None
    displayed = []
    for name, obj, w, h, shown in _iter_images(page):
        if not shown:
            continue
        got = _extract_original(obj, w, h)
        if got:
            displayed.append((got, w, h))
    if len(displayed) != 1:
        return None
    (data, media, method), w, h = displayed[0]
    return data, media, method, w, h


def render_pdf_page(path: str, index: int, dpi: int, dest: str, width_pt: float, height_pt: float) -> tuple[int, int]:
    """Rasterize one page. This creates a new PNG; it is not the embedded file."""
    from .safety import clamp_dpi
    dpi = clamp_dpi(dpi)
    zoom = dpi / 72.0
    if width_pt and height_pt:
        wpx = max(1, int(round(width_pt * zoom)))
        hpx = max(1, int(round(height_pt * zoom)))
        check_pixels(wpx, hpx)
    if _render_pymupdf(path, index, zoom, dest):
        return _dims(dest)
    if _render_pdftoppm(path, index, dpi, dest):
        return _dims(dest)
    raise RenderUnavailable(
        "이 페이지는 임베디드 이미지 한 장과 같지 않아 렌더가 필요합니다. "
        "pymupdf 또는 poppler(pdftoppm)가 설치되어 있지 않습니다. 원본 PDF는 변경하지 않았습니다."
    )


def _dims(path: str) -> tuple[int, int]:
    from .media import image_info_file
    _kind, w, h = image_info_file(path)
    return w, h


def _render_pymupdf(path, index, zoom, dest) -> bool:
    try:
        import pymupdf
    except ImportError:
        return False
    doc = pymupdf.open(path)
    try:
        page = doc[index]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        check_pixels(pix.width, pix.height)
        pix.save(dest)
        return True
    finally:
        doc.close()


def _render_pdftoppm(path, index, dpi, dest) -> bool:
    bin_path = shutil.which("pdftoppm")
    if not bin_path:
        return False
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "p")
        subprocess.run(
            [bin_path, "-f", str(index + 1), "-l", str(index + 1), "-png", "-r", str(dpi), "-singlefile", path, prefix],
            check=True, capture_output=True,
        )
        src = prefix + ".png"
        if not os.path.exists(src):
            return False
        os.replace(src, dest)
    return True


def extract_pdf_page(path: str, index: int, dest_dir: str, dpi: int, page_id: str, width_pt: float = 0, height_pt: float = 0) -> dict:
    """Write exactly one page image into dest_dir. Does not keep other pages in memory."""
    os.makedirs(dest_dir, exist_ok=True)
    reader = _reader(path)
    page = reader.pages[index]
    if not width_pt:
        try:
            box = page.mediabox
            width_pt, height_pt = float(box.width), float(box.height)
        except Exception:
            width_pt, height_pt = 0.0, 0.0
    direct = choose_direct(page)
    if direct:
        data, media, method, w, h = direct
        dest = os.path.join(dest_dir, page_id + ext_for(media))
        with open(dest, "wb") as f:
            f.write(data)
        return {"path": dest, "width": w, "height": h, "media_type": media, "method": method,
                "byte_size": len(data), "width_pt": width_pt, "height_pt": height_pt,
                "note": "임베디드 JPEG을 재인코딩하지 않고 꺼냈습니다."}
    dest = os.path.join(dest_dir, page_id + ".png")
    w, h = render_pdf_page(path, index, dpi, dest, width_pt, height_pt)
    return {"path": dest, "width": w, "height": h, "media_type": "image/png", "method": "render",
            "byte_size": os.path.getsize(dest), "width_pt": width_pt, "height_pt": height_pt,
            "note": f"{dpi} DPI 렌더입니다. 원본 스트림을 그대로 둔 것이 아니라 새 PNG입니다."}
