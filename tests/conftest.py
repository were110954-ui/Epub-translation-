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
    MockEngine._state = {}
    monkeypatch.delenv("MOCK_FAULTS", raising=False)
    yield
