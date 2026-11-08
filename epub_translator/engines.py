"""LLM engines: OpenAI, OpenAI-compatible, Anthropic, Gemini, and a deterministic Mock engine (tests/offline).

Image parts follow each provider's documented chat request. A model name is not treated as proof
that the model accepts images. Mock never reports a high-quality result.
"""
from __future__ import annotations
import asyncio
import base64
import json
import os
import random
import re
from dataclasses import dataclass

import httpx


@dataclass
class EngineResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False


class EngineError(Exception):
    def __init__(self, msg, retryable=True, retry_after: float | None = None, quota=False):
        super().__init__(msg)
        self.retryable = retryable
        self.retry_after = retry_after
        self.quota = quota


def _retry_after(r) -> float | None:
    ra = r.headers.get("retry-after")
    if ra and ra.replace(".", "").isdigit():
        return float(ra)
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', r.text) or re.search(r"retry in ([\d.]+)s", r.text)
    return float(m.group(1)) if m else None


# USD per 1M tokens (input, output). Editable estimate table; unknown models use default.
PRICES = {
    "gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6), "gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.4, 1.6),
    "gemini-flash-latest": (0.3, 2.5), "gemini-3.5-flash": (0.5, 3.0), "gemini-3.8-flash": (0.5, 3.0), "gemini-3.5-flash-lite": (0.1, 0.4), "gemini-2.5-flash": (0.3, 2.5), "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash-lite": (0.1, 0.4),
    "claude-sonnet-4-5": (3.0, 15.0), "claude-3-5-haiku-latest": (0.8, 4.0),
    "mock": (0.0, 0.0),
}
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini", "gemini": "gemini-3.5-flash", "anthropic": "claude-sonnet-4-5",
    "openai_compatible": "local-model", "mock": "mock",
}
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "openai_compatible": "OPENAI_COMPAT_API_KEY"}


def price_for(model: str):
    if model in PRICES:
        return PRICES[model]
    for k, v in PRICES.items():
        if model.startswith(k):
            return v
    return (1.0, 4.0)


_VISION_MIME = {"image/jpeg", "image/png"}


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def vision_parts_openai(user: str, images: list[tuple[str, bytes]]) -> list:
    """OpenAI Chat Completions image input: text plus image_url data URLs. detail=low bounds tokens."""
    parts: list = [{"type": "text", "text": user}]
    for mime, raw in images:
        if mime not in _VISION_MIME:
            raise EngineError(f"unsupported image type {mime}", retryable=False)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(raw)}", "detail": "low"}})
    return parts


def vision_parts_anthropic(user: str, images: list[tuple[str, bytes]]) -> list:
    """Anthropic Messages image blocks: base64 source, then the text block."""
    parts = []
    for mime, raw in images:
        if mime not in _VISION_MIME:
            raise EngineError(f"unsupported image type {mime}", retryable=False)
        parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": _b64(raw)}})
    parts.append({"type": "text", "text": user})
    return parts


def vision_parts_gemini(user: str, images: list[tuple[str, bytes]]) -> list:
    """Gemini generateContent parts: inlineData (camelCase, as in the REST v1beta text path) then text."""
    parts = []
    for mime, raw in images:
        if mime not in _VISION_MIME:
            raise EngineError(f"unsupported image type {mime}", retryable=False)
        parts.append({"inlineData": {"mimeType": mime, "data": _b64(raw)}})
    parts.append({"text": user})
    return parts


class BaseEngine:
    name = "base"
    offline = False

    def __init__(self, model: str, api_key: str = "", temperature: float = 0.3, base_url: str = "",
                 timeout: float = 300, max_output_tokens: int = 16384):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    async def complete(self, system: str, user: str) -> EngineResult:
        raise NotImplementedError

    async def complete_vision(self, system: str, user: str, images: list[tuple[str, bytes]]) -> EngineResult:
        raise EngineError("이 엔진은 페이지 이미지를 보내는 경로가 없습니다.", retryable=False)

    def _vision_tokens(self) -> int:
        return max(256, min(4096, int(self.max_output_tokens)))

    async def _post(self, url, headers, payload):
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(url, headers=headers, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise EngineError(f"network: {e!r}")
        if r.status_code == 429:
            ra = _retry_after(r)
            daily = "PerDay" in r.text or "per day" in r.text.lower() or (ra or 0) > 900
            if daily:
                raise EngineError(f"API 할당량(일일 한도) 초과 — 할당량이 회복된 후 '재개'하면 이어서 진행합니다. "
                                  f"(retry after ~{int(ra or 0) // 60}분) {r.text[:200]}", retryable=False, retry_after=ra, quota=True)
            raise EngineError(f"HTTP 429 rate limit: {r.text[:200]}", retry_after=ra)
        if r.status_code in (408, 409, 425) or r.status_code >= 500:
            raise EngineError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise EngineError(f"HTTP {r.status_code}: {r.text[:500]}", retryable=False)
        return r.json()


class OpenAIEngine(BaseEngine):
    name = "openai"
    default_base = "https://api.openai.com/v1"

    async def complete(self, system, user):
        url = (self.base_url or self.default_base).rstrip("/") + "/chat/completions"
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "max_tokens": self.max_output_tokens}
        if self.name == "openai":
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        d = await self._post(url, headers, payload)
        ch = d["choices"][0]
        u = d.get("usage", {}) or {}
        return EngineResult(ch["message"].get("content") or "", u.get("prompt_tokens", 0),
                            u.get("completion_tokens", 0), ch.get("finish_reason") == "length")

    async def complete_vision(self, system, user, images):
        # Vision calls do not force response_format. Some image models reject json_object mode.
        url = (self.base_url or self.default_base).rstrip("/") + "/chat/completions"
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": vision_parts_openai(user, images)}],
                   "max_tokens": self._vision_tokens()}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        d = await self._post(url, headers, payload)
        ch = d["choices"][0]
        u = d.get("usage", {}) or {}
        return EngineResult(ch["message"].get("content") or "", u.get("prompt_tokens", 0),
                            u.get("completion_tokens", 0), ch.get("finish_reason") == "length")


class OpenAICompatEngine(OpenAIEngine):
    name = "openai_compatible"
    default_base = "http://localhost:11434/v1"


class AnthropicEngine(BaseEngine):
    name = "anthropic"

    async def complete(self, system, user):
        url = (self.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        payload = {"model": self.model, "max_tokens": self.max_output_tokens, "temperature": self.temperature,
                   "system": system, "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        d = await self._post(url, headers, payload)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return EngineResult(text, u.get("input_tokens", 0), u.get("output_tokens", 0), d.get("stop_reason") == "max_tokens")

    async def complete_vision(self, system, user, images):
        url = (self.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        payload = {"model": self.model, "max_tokens": self._vision_tokens(), "temperature": self.temperature,
                   "system": system, "messages": [{"role": "user", "content": vision_parts_anthropic(user, images)}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        d = await self._post(url, headers, payload)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return EngineResult(text, u.get("input_tokens", 0), u.get("output_tokens", 0), d.get("stop_reason") == "max_tokens")


class GeminiEngine(BaseEngine):
    name = "gemini"

    async def complete(self, system, user):
        base = (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{self.model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens,
                                 "responseMimeType": "application/json"},
            "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")],
        }
        d = await self._post(url, {"x-goog-api-key": self.api_key}, payload)
        cands = d.get("candidates") or []
        if not cands:
            raise EngineError(f"Gemini: no candidates ({json.dumps(d.get('promptFeedback', {}))[:200]})")
        c = cands[0]
        text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
        u = d.get("usageMetadata", {})
        return EngineResult(text, u.get("promptTokenCount", 0),
                            u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                            c.get("finishReason") == "MAX_TOKENS")

    async def complete_vision(self, system, user, images):
        base = (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{self.model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": vision_parts_gemini(user, images)}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self._vision_tokens(),
                                 "responseMimeType": "application/json"},
            "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")],
        }
        d = await self._post(url, {"x-goog-api-key": self.api_key}, payload)
        cands = d.get("candidates") or []
        if not cands:
            raise EngineError(f"Gemini: no candidates ({json.dumps(d.get('promptFeedback', {}))[:200]})")
        c = cands[0]
        text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
        u = d.get("usageMetadata", {})
        return EngineResult(text, u.get("promptTokenCount", 0),
                            u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                            c.get("finishReason") == "MAX_TOKENS")


# ----------------------------------------------------------------- mock engine
def mock_ko(s: str) -> str:
    """Deterministic pseudo-translation: Japanese letters -> Hangul syllables; punctuation converted."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x3041 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            out.append(chr(0xAC00 + (o * 37) % 11172))
        elif ch in "「『":
            out.append("“")
        elif ch in "」』":
            out.append("”")
        elif ch == "。":
            out.append(".")
        elif ch == "、":
            out.append(", ")
        else:
            out.append(ch)
    return "".join(out)


class MockEngine(BaseEngine):
    """Offline engine. Parses the JSON request in the prompt and returns valid JSON.
    Fault injection via env MOCK_FAULTS (comma list): drop_first, truncate_once, japanese_once, fail_once, dup_once."""
    name = "mock"
    _state: dict = {}

    async def complete(self, system, user):
        await asyncio.sleep(0.01)
        m = re.search(r"<<<BLOCKS\n(.*)\nBLOCKS>>>", user, re.S)
        req = json.loads(m.group(1))
        gl = {}
        gm = re.search(r"<<<GLOSSARY\n(.*?)\nGLOSSARY>>>", user, re.S)
        if gm:
            for line in gm.group(1).splitlines():
                if "=>" in line:
                    a, b = line.split("=>", 1)
                    gl[a.strip().lstrip("- ")] = b.split("(")[0].strip()
        faults = set(filter(None, os.environ.get("MOCK_FAULTS", "").split(",")))
        st = MockEngine._state
        if "quota_once" in faults and not st.get("quota") and st.get("calls", 0) >= 2:
            st["quota"] = True
            raise EngineError("mock daily quota exceeded", retryable=False, quota=True)
        st["calls"] = st.get("calls", 0) + 1
        if "fail_once" in faults and not st.get("failed"):
            st["failed"] = True
            raise EngineError("mock transient failure")
        if "truncate_once" in faults and not st.get("trunc") and len(req) > 1:
            st["trunc"] = True
            text = json.dumps({"translations": [{"id": b["id"], "text": mock_ko(b["text"])} for b in req]}, ensure_ascii=False)
            return EngineResult(text[: len(text) // 2], len(user) // 2, len(text) // 4, True)
        items = []
        for b in req:
            t = b["text"]
            for k, v in sorted(gl.items(), key=lambda kv: -len(kv[0])):
                t = t.replace(k, v)
            items.append({"id": b["id"], "text": mock_ko(t)})
        if "drop_first" in faults and not st.get("dropped") and len(items) > 1:
            st["dropped"] = True
            items = items[1:]
        if "japanese_once" in faults and not st.get("jp") and items:
            st["jp"] = True
            items[0]["text"] = req[0]["text"]
        if "dup_once" in faults and not st.get("dup") and len(items) > 1:
            st["dup"] = True
            items[1]["text"] = items[0]["text"]
        names = sorted(set(re.findall(r"[\u4E00-\u9FFF]{2,3}(?=[はがのを、])", " ".join(b["text"] for b in req))))[:3]
        resp = {"translations": items,
                "summary": f"mock summary of {len(req)} blocks",
                "new_terms": [{"source": n, "target": mock_ko(n), "type": "name"} for n in names if n not in gl],
                "characters": []}
        text = json.dumps(resp, ensure_ascii=False)
        return EngineResult(text, len(system + user) // 2, len(text) // 2, False)

    async def complete_vision(self, system, user, images):
        if os.environ.get("MOCK_VISION") != "1":
            raise EngineError("mock 엔진은 페이지 이미지를 보지 않습니다. 고품질 성공으로 표시하지 않습니다.", retryable=False)
        return EngineResult(json.dumps({
            "quality": "mock",
            "readings": [], "groups": [], "translations": [], "proposed": [],
            "summary": "mock vision",
            "review": {"issues": [], "opinion": "mock opinion, not a proof"},
        }, ensure_ascii=False), 1, 1, False)


ENGINES = {"openai": OpenAIEngine, "openai_compatible": OpenAICompatEngine, "anthropic": AnthropicEngine,
           "gemini": GeminiEngine, "mock": MockEngine}


def make_engine(settings: dict) -> BaseEngine:
    kind = settings.get("engine", "mock")
    cls = ENGINES[kind]
    key = settings.get("api_key") or os.environ.get(KEY_ENV.get(kind, ""), "")
    if kind == "gemini" and not key:
        key = os.environ.get("GOOGLE_API_KEY", "")
    if kind not in ("mock", "openai_compatible") and not key:
        raise EngineError(f"{kind} API 키가 설정되지 않았습니다 ({KEY_ENV.get(kind)})", retryable=False)
    base = settings.get("base_url") or (os.environ.get("OPENAI_COMPAT_BASE_URL", "") if kind == "openai_compatible" else "")
    return cls(model=settings.get("model") or DEFAULT_MODELS[kind], api_key=key,
               temperature=float(settings.get("temperature", 0.3)), base_url=base,
               max_output_tokens=int(settings.get("max_output_tokens", 16384)))
