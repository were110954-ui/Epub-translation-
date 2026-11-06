"""Prompt construction + long-novel context: rolling summary, glossary, character speech/honorific DB."""
from __future__ import annotations
import json
import re

LANG_NAMES = {"ja": "일본어(Japanese)", "ko": "한국어(Korean)", "en": "영어(English)", "zh": "중국어(Chinese)"}

SYSTEM_PROMPT = """You are a professional literary translator translating a {src} novel into natural, publication-quality {tgt}.
Rules (must follow all):
1. Translate EVERY block completely and faithfully. Never summarize, omit, merge, split or add content. One output item per input id, same ids, same order.
2. Preserve meaning, tone and style. Replace stiff Japanese literal phrasing with natural {tgt} prose (no 번역투).
3. Keep character names, honorifics (e.g. 先輩→선배, さん→씨 or as glossary says) and speech styles (반말/존댓말, dialect, verbal tics) consistent with the glossary and character notes.
4. Distinguish narration style from dialogue. Japanese 「」 dialogue becomes Korean “ ” quotes; 『』 becomes ‘ ’ (or keep 『』 for titles).
5. Translate onomatopoeia/mimetic words and idioms naturally for context.
6. Resolve omitted subjects and pronouns from context, but do not add explicit information the original leaves ambiguous.
7. Inline markup tokens like <t3>...</t3> and <t5/> MUST be kept exactly (same numbers, each exactly once, properly nested) around the corresponding translated words. Do not invent tokens.
8. Furigana hints in the form 漢字(よみ) tell you readings for names; use them to transliterate names into {tgt} (e.g. 香崎(こうざき) → 코자키).
9. Output must contain no untranslated Japanese kana. Do not leave Japanese text except for intentional quoted foreign words.
10. Return ONLY JSON: {{"translations":[{{"id":"...","text":"..."}}], "summary":"2-4 sentence {tgt} summary of what happened in this passage (who, where, what) for continuity", "new_terms":[{{"source":"原語","target":"번역어","type":"name|place|term|honorific"}}], "characters":[{{"name":"原名","korean":"한국어 이름","speech":"말투 설명 (반말/존댓말, 1인칭 등)","honorifics":"호칭 관계"}}]}}
new_terms/characters: include only proper nouns / recurring terms / characters that appear in THIS passage and are not already in the glossary/character list."""


def build_prompt(src_lang, tgt_lang, chunk_blocks, glossary, characters, summaries, prev_pairs, chapter):
    system = SYSTEM_PROMPT.format(src=LANG_NAMES.get(src_lang, src_lang), tgt=LANG_NAMES.get(tgt_lang, tgt_lang))
    if any(b.get("kind") == "balloon" for b in chunk_blocks):
        system += ("\n11. These blocks are comic speech balloons. Translate each balloon on its own, keep the line short "
                   "enough to letter inside the balloon, and keep each character's speech style consistent with the glossary "
                   "and character notes. Do not omit a balloon or merge two balloons into one line.")
    text_all = " ".join(b["source"] for b in chunk_blocks)
    # only relevant glossary entries (appearing in chunk) + top-level names, to bound prompt size
    rel = [g for g in glossary if g["source"] and g["source"] in text_all]
    rel_names = {g["source"] for g in rel}
    chars = [c for c in characters if c["name"] in text_all or (c.get("korean") or "") in text_all]
    parts = [f"Chapter: {chapter}"]
    if summaries:
        parts.append("Story so far (most recent last):\n" + "\n".join(f"- {s}" for s in summaries[-4:]))
    if rel:
        lines = [f"- {g['source']} => {g['target']}" + (f" (reading: {g['reading']})" if g.get("reading") else "")
                 + (f" ({g['type']})" if g.get("type") else "") for g in rel]
        parts.append("<<<GLOSSARY\n" + "\n".join(lines) + "\nGLOSSARY>>>")
    if chars:
        parts.append("Characters (keep consistent):\n" + "\n".join(
            f"- {c['name']} / {c.get('korean') or ''}: 말투={c.get('speech') or ''}; 호칭={c.get('honorifics') or ''}" for c in chars))
    if prev_pairs:
        parts.append("Previous passage (already translated, for continuity; do NOT translate again):\n" +
                     "\n".join(f"JA: {s}\nKO: {t}" for s, t in prev_pairs[-4:]))
    req = []
    for b in chunk_blocks:
        txt = b["source"]
        furi = b.get("furigana") or {}
        item = {"id": b["block_id"], "text": txt}
        hints = {k: v for k, v in furi.items() if k in txt and k not in rel_names}
        if hints:
            item["readings"] = hints
        if b["kind"] in ("ncx", "title", "meta"):
            item["note"] = "table-of-contents/title label"
        elif b.get("kind") == "balloon":
            item["note"] = "comic speech balloon " + str(b.get("xpath") or "")
        req.append(item)
    parts.append(f"Translate these {len(req)} blocks. Return JSON only.\n<<<BLOCKS\n" +
                 json.dumps(req, ensure_ascii=False, indent=0) + "\nBLOCKS>>>")
    return system, "\n\n".join(parts)


def parse_response(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise
