"""Per-block translation checks. A block is only marked 'failed' when structure/semantic checks agree;
length alone only produces a warning which triggers a re-review."""
from __future__ import annotations
import re

from .epub_model import check_tokens
from .textutil import jp_stats, residual_japanese, is_dialogue, strip_tokens, norm

DIGITS = re.compile(r"\d+")


def target_quote_count(t: str) -> int:
    t = strip_tokens(t)
    return t.count("“") + t.count("「") + t.count("『") + t.count("‘") + t.count('"') // 2


LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
LATIN_RUN = re.compile(r"(?:[A-Za-z]+[\u2019'\-,]?\s+){4,}[A-Za-z]+")


def check_block(src: str, dst: str | None, kind: str = "block", tgt_lang: str = "ko") -> tuple[str, list[str]]:
    """Return (severity, flags). severity in ok|warn|error."""
    flags = []
    if dst is None or not strip_tokens(dst).strip():
        return "error", ["missing"]
    flags += check_tokens(src, dst)
    s, d = jp_stats(src), jp_stats(dst)
    if tgt_lang == "ko":
        res = residual_japanese(dst)
        if res:
            flags.append("residual-japanese:" + "".join(res)[:20])
        if s["len"] >= 4 and d["hangul"] == 0 and (s["hira"] + s["kata"] + s["kanji"]) > 0:
            flags.append("no-hangul")
        # Latin-script source (e.g. English): untranslated English left in output
        s_lat = len(LATIN_WORD.findall(src))
        if s_lat >= 2 and d["hangul"] == 0 and "no-hangul" not in flags:
            flags.append("no-hangul")
        elif d["hangul"] > 0 and s_lat >= 4:
            run = LATIN_RUN.search(strip_tokens(dst))
            if run:
                flags.append("residual-latin:" + run.group(0)[:20])
    # length ratio (Korean is usually 0.7x-1.6x of Japanese char count)
    ratio = d["len"] / max(1, s["len"])
    short = s["len"] >= 15 and ratio < 0.35
    long_ = s["len"] >= 10 and ratio > 3.5
    # semantic/structural corroboration for short translations
    sem = []
    if is_dialogue(src) and target_quote_count(dst) == 0:
        sem.append("dialogue-lost")
    sd = DIGITS.findall(norm(src))
    if sd and not set(sd) <= set(DIGITS.findall(norm(dst))):
        sem.append("numbers-mismatch")
    s_sent = len(re.findall(r"[。！？!?]", strip_tokens(src)))
    d_sent = len(re.findall(r"[.。！？!?]", strip_tokens(dst)))
    if s_sent >= 3 and d_sent < s_sent * 0.4:
        sem.append("sentences-missing")
    if short:
        flags.append("too-short")
    if long_:
        flags.append("too-long")
    flags += sem
    errors = {"missing", "tag-mismatch", "tag-nesting", "tag-unclosed", "no-hangul"}
    if any(f in errors or f.startswith(("residual-japanese", "residual-latin")) for f in flags):
        return "error", flags
    # short is an error only when corroborated by a structural/semantic signal
    if short and sem:
        return "error", flags
    if "sentences-missing" in sem and ratio < 0.6:
        return "error", flags
    return ("warn" if flags else "ok"), flags


def find_duplicates(rows: list[dict]) -> dict[str, str]:
    """Detect duplicated translations: two different source blocks with identical (non-trivial) target,
    or a target appearing twice in a row inside a chunk. Returns {block_id: duplicate_of}."""
    dup = {}
    seen = {}
    for r in rows:
        t = norm(r.get("target") or "")
        s = norm(r["source"])
        if len(t) < 12:
            continue
        if t in seen and seen[t][1] != s:
            dup[r["block_id"]] = seen[t][0]
        else:
            seen.setdefault(t, (r["block_id"], s))
    return dup


def repair_tokens(src: str, dst: str | None) -> str | None:
    """Safe local repair: drop placeholder tokens the model invented (numbers not present in the source)."""
    if not dst:
        return dst
    from .epub_model import TOKEN_RE
    allowed = {n for _, n, _ in TOKEN_RE.findall(src)}
    return TOKEN_RE.sub(lambda m: m.group(0) if m.group(2) in allowed else "", dst)


def flatten_tokens(src: str, dst: str | None) -> str | None:
    """Last-resort fallback for blocks whose inline formatting tokens the model could not
    keep in order (word order differs between languages). Only for sources without
    self-closing tokens (images, line breaks, anchors are never dropped). The whole
    translation is wrapped in the first paired token so the block-level styling survives;
    inner emphasis spans are lost and the block is reported as a warning ("tags-flattened")."""
    from .epub_model import TOKEN_RE
    if not dst:
        return None
    toks = TOKEN_RE.findall(src)
    if not toks or any(t[2] for t in toks):
        return None
    plain = TOKEN_RE.sub("", dst).strip()
    if not plain:
        return None
    first = toks[0][1]
    return f"<t{first}>{plain}</t{first}>"
