import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    import make_sample_epub
    d = tmp_path_factory.mktemp("samples")
    make_sample_epub.make(str(d / "s3.epub"), True)
    make_sample_epub.make(str(d / "s2.epub"), False)
    return {"v3": str(d / "s3.epub"), "v2": str(d / "s2.epub")}


@pytest.fixture(autouse=True)
def reset_mock(monkeypatch):
    from epub_translator.engines import MockEngine
    from epub_translator.image_jobs import SLOTS
    MockEngine._state = {}
    SLOTS.reset()
    monkeypatch.delenv("MOCK_FAULTS", raising=False)
    monkeypatch.delenv("EPUB_PAGE_DELAY", raising=False)
    yield


@pytest.fixture(scope="session", autouse=True)
def _local_epubcheck():
    """Use a downloaded EPUBCheck jar when the developer machine has one under /tmp. Not committed."""
    import shutil
    src_dir = "/tmp/epubcheck/epubcheck-5.1.0"
    src = os.path.join(src_dir, "epubcheck.jar")
    if not os.path.exists(src):
        return
    dest = os.path.join(ROOT, "tools", "epubcheck")
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, "epubcheck.jar")
    if not os.path.exists(target):
        shutil.copyfile(src, target)
    lib_src = os.path.join(src_dir, "lib")
    lib_dest = os.path.join(dest, "lib")
    if os.path.isdir(lib_src) and not os.path.isdir(lib_dest):
        shutil.copytree(lib_src, lib_dest)
