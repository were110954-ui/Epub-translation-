"""Local configuration: .env loading and an API-key store (data/secrets.json, chmod 600). Keys are never
returned to the browser in clear text and never stored in the job database."""
from __future__ import annotations
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("EPUB_TRANSLATOR_DATA", os.path.join(ROOT, "data"))
SECRET_KEYS = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_BASE_URL"]


def load_dotenv(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def secrets_path():
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "secrets.json")


def load_secrets():
    p = secrets_path()
    if os.path.exists(p):
        for k, v in json.load(open(p, encoding="utf-8")).items():
            if v:
                os.environ.setdefault(k, v)


def save_secret(name: str, value: str):
    if name not in SECRET_KEYS:
        raise ValueError("unknown key")
    p = secrets_path()
    d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    if value:
        d[name] = value
        os.environ[name] = value
    else:
        d.pop(name, None)
        os.environ.pop(name, None)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def masked():
    out = {}
    for k in SECRET_KEYS:
        v = os.environ.get(k, "")
        out[k] = ("설정됨 (…" + v[-4:] + ")") if v and "KEY" in k else (v if v else "")
    return out
