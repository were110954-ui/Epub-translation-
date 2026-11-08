"""Decide which of the four modes a file belongs to. EPUB bytes are only sniffed, never rewritten."""
from __future__ import annotations
import os
import zipfile

from .epub_model import DRMError
from .safety import SafetyError

PDF_CONVERSION_REMOVED = (
    "PDF를 EPUB으로만 바꾸는 기능은 없습니다. PDF는 리더 앱에서 그대로 열 수 있습니다. "
    "만화 AI 번역이 필요할 때만 PDF를 선택하세요. 파일을 올리기만 해서 번역 API가 호출되지는 않습니다. "
    "예전에 만든 PDF 변환 기록과 결과 파일은 지우지 않으며, 완료된 파일은 받을 수 있습니다. "
    "그 작업의 시작·재개는 거부됩니다."
)


def _is_epub_zip(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "mimetype" in names:
                raw = z.read("mimetype").strip()
                if raw.startswith(b"application/epub"):
                    return True
            if "META-INF/container.xml" in names:
                return True
    except zipfile.BadZipFile:
        return False
    return False


def sniff_kind(path: str, file_name: str | None = None) -> str:
    """Return 'epub', 'pdf', or 'cbz' from magic bytes. Extension is only a hint for zip files."""
    name = file_name or os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    with open(path, "rb") as f:
        magic = f.read(16)
    if magic.startswith(b"%PDF"):
        return "pdf"
    if magic.startswith(b"PK"):
        if ext == ".epub" or _is_epub_zip(path):
            return "epub"
        if ext in (".cbz", ".zip", ".epub", ""):
            return "cbz"
        return "cbz"
    if ext == ".epub":
        return "epub"
    raise SafetyError("지원하지 않는 파일입니다. EPUB, PDF, CBZ(ZIP)만 올릴 수 있습니다.")


def resolve_mode(path: str, file_name: str | None, requested: str | None) -> tuple[str, str]:
    """Return (mode, input_kind).

    mode is epub (novel), cbz, or comic. Standalone pdf conversion is rejected.
    Comic is explicit and accepts a PDF or CBZ. It does not OCR a novel EPUB.
    A PDF is not auto-selected into a job: the caller must ask for comic translation.
    """
    kind = sniff_kind(path, file_name)
    mode = (requested or "auto").strip().lower()
    if mode == "comic":
        if kind == "epub":
            raise SafetyError("만화 번역은 PDF 또는 CBZ 이미지 원본만 지원합니다. EPUB 소설은 기존 번역 모드를 사용하세요.")
        if kind not in ("pdf", "cbz"):
            raise SafetyError("만화 번역은 PDF 또는 CBZ만 지원합니다.")
        return "comic", kind
    if mode == "pdf" or (mode in ("", "auto") and kind == "pdf"):
        raise SafetyError(PDF_CONVERSION_REMOVED)
    if mode in ("", "auto"):
        mode = kind
    if mode != kind:
        raise SafetyError(f"파일 형식은 {kind} 인데 요청한 모드는 {mode} 입니다.")
    if kind == "epub":
        return "epub", "epub"
    return mode, kind


def reject_encrypted_zip(path: str):
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.flag_bits & 0x1:
                raise DRMError("암호화된 ZIP/CBZ는 처리하지 않습니다. 암호를 푸는 기능은 없습니다.")
