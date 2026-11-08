"""Resource bounds for PDF/CBZ/comic jobs. These reject hostile archives; they do not rewrite the user's file."""
from __future__ import annotations

MAX_FILES = 8000
MAX_PAGES = 5000
MAX_SINGLE = 80 * 1024 * 1024
MAX_UNCOMPRESSED = 4 * 1024 * 1024 * 1024
MAX_RATIO = 80
MAX_PIXELS = 40_000_000
MAX_DIM = 20_000
MAX_UPLOAD = 2 * 1024 * 1024 * 1024
HEADER_READ = 512 * 1024


class SafetyError(Exception):
    """A file exceeded a safety bound (path traversal, zip bomb, pixel bomb, unsupported payload)."""


class RightsError(Exception):
    """The caller did not confirm they have the legal right to translate or convert the file."""


class RenderUnavailable(Exception):
    """A PDF page is not a single embedded image and no render backend is installed."""


class OcrUnavailable(Exception):
    """Comic OCR was requested and no real OCR backend is installed."""


def check_pixels(w: int, h: int):
    if w <= 0 or h <= 0 or w > MAX_DIM or h > MAX_DIM or w * h > MAX_PIXELS:
        raise SafetyError(f"이미지 크기가 허용 범위를 넘습니다 ({w}x{h}). 픽셀 폭탄으로 보고 거부했습니다.")


def configure_pillow():
    try:
        from PIL import Image
    except ImportError:
        return
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def clamp_dpi(value) -> int:
    dpi = int(value)
    if dpi < 300 or dpi > 600:
        raise ValueError("DPI는 300에서 600 사이여야 합니다. 임베디드 이미지가 페이지와 같으면 렌더를 하지 않습니다.")
    return dpi


def clamp_parallel(value) -> int:
    n = int(value)
    if n < 1:
        n = 1
    if n > 4:
        n = 4
    return n
