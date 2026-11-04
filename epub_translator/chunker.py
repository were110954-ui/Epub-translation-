"""Semantic chunking: chunks never split a block, stay inside one chapter file, prefer to break at
scene breaks / headings / narrative paragraphs, and avoid splitting a running dialogue exchange."""
from __future__ import annotations
import hashlib
import re

from .textutil import is_dialogue, strip_tokens

SCENE_BREAK = re.compile(r"^[\s\u3000]*([◇◆□■○●☆★※＊*＃#◎・…―─\-]\s*){1,}[\s\u3000]*$")


def break_score(prev, nxt) -> int:
    """Higher = better place to cut between prev and nxt."""
    p, n = strip_tokens(prev["source"]), strip_tokens(nxt["source"])
    if SCENE_BREAK.match(p) or SCENE_BREAK.match(n):
        return 100
    if nxt.get("kind") == "block" and nxt.get("is_heading"):
        return 90
    if not is_dialogue(p) and not is_dialogue(n):
        return 50
    if is_dialogue(p) and not is_dialogue(n):
        return 30
    if not is_dialogue(p) and is_dialogue(n):
        return 20
    return 0  # inside dialogue run


def make_chunks(blocks: list[dict], chunk_size: int = 2500, max_blocks: int | None = None) -> list[dict]:
    """blocks: dicts with block_id,file,chapter,source,kind. Returns list of {chunk_id,file,chapter,block_ids}."""
    max_blocks = max_blocks or max(60, chunk_size // 20)
    chunks = []
    by_file: dict[str, list] = {}
    order = []
    for b in blocks:
        if b["file"] not in by_file:
            order.append(b["file"])
        by_file.setdefault(b["file"], []).append(b)
    for f in order:
        bl = by_file[f]
        i = 0
        while i < len(bl):
            size = 0
            j = i
            best_cut, best_score = None, -1
            while j < len(bl):
                size += len(bl[j]["source"])
                j += 1
                if j < len(bl) and size >= chunk_size * 0.5:
                    s = break_score(bl[j - 1], bl[j])
                    if s >= best_score:
                        best_cut, best_score = j, s
                if size >= chunk_size or (j - i) >= max_blocks:
                    break
            if j >= len(bl):
                cut = len(bl)
            elif best_cut is not None and best_score > 0:
                cut = best_cut
            else:
                cut = j
            ids = [b["block_id"] for b in bl[i:cut]]
            h = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:10]
            chunks.append({"chunk_id": f"c{len(chunks):05d}-{h}", "file": f, "chapter": bl[i]["chapter"],
                           "block_ids": ids, "chars": sum(len(b["source"]) for b in bl[i:cut])})
            i = cut
    return merge_small(chunks, chunk_size)


def merge_small(chunks, chunk_size):
    """Merge consecutive tiny chunks (title pages, image pages, colophons) to save API calls."""
    out = []
    for c in chunks:
        small = c["chars"] < chunk_size * 0.25
        if out and small and out[-1].get("_small") and out[-1]["chars"] + c["chars"] <= chunk_size * 0.5 \
                and len(out[-1]["block_ids"]) + len(c["block_ids"]) <= 60:
            p = out[-1]
            p["block_ids"] = p["block_ids"] + c["block_ids"]
            p["chars"] += c["chars"]
            h = hashlib.sha1("\n".join(p["block_ids"]).encode()).hexdigest()[:10]
            p["chunk_id"] = p["chunk_id"].split("-")[0] + "-" + h
        else:
            out.append(dict(c, _small=small))
    for i, c in enumerate(out):
        c.pop("_small", None)
        c["chunk_id"] = f"c{i:05d}-" + c["chunk_id"].split("-")[1]
    return out
