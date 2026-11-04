"""Character-class helpers used by chunking and validation."""
import re
import unicodedata

HIRA = re.compile(r"[\u3041-\u309F]")
KATA = re.compile(r"[\u30A0-\u30FF\u31F0-\u31FF\uFF66-\uFF9F]")
KANJI = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
HANGUL = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")
TAG_TOKEN = re.compile(r"</?t\d+/?>")
JP_PUNCT_ONLY = re.compile(r"^[\s\u3000-\u303F\uFF01-\uFF0F\uFF1A-\uFF20\u2010-\u2027\u2030-\u205E！-～・…―─◇◆□■○●☆★※＊*#\-–—~.,!?\"'()\[\]{}0-9０-９]*$")

def strip_tokens(s: str) -> str:
    return TAG_TOKEN.sub("", s or "")

def is_translatable(text: str) -> bool:
    """True if text contains letters needing translation (not only punctuation/digits/symbols)."""
    t = strip_tokens(text).strip()
    if not t:
        return False
    if JP_PUNCT_ONLY.match(t):
        return False
    return bool(HIRA.search(t) or KATA.search(t) or KANJI.search(t) or re.search(r"[A-Za-z]", t))

def jp_stats(text: str) -> dict:
    t = strip_tokens(text)
    return {
        "hira": len(HIRA.findall(t)),
        "kata": len(KATA.findall(t)),
        "kanji": len(KANJI.findall(t)),
        "hangul": len(HANGUL.findall(t)),
        "len": len(re.sub(r"\s", "", t)),
    }

def kana_only(s: str) -> bool:
    s = s.strip()
    return bool(s) and all(HIRA.match(c) or KATA.match(c) or c in "ー・ 　" for c in s)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", strip_tokens(s))).strip()

def residual_japanese(text: str) -> list[str]:
    """Return Japanese fragments (kana runs, or kanji runs not in parentheses gloss) remaining in a Korean translation."""
    t = strip_tokens(text)
    frags = re.findall(r"[\u3041-\u309F\u30A1-\u30FA\u30FC\u31F0-\u31FF\uFF66-\uFF9F]{2,}", t)
    # Kanji runs of length>=2 outside of parentheses (Korean sometimes glosses hanja in parens)
    no_paren = re.sub(r"[（(][^）)]*[）)]", "", t)
    frags += re.findall(r"[\u4E00-\u9FFF]{2,}", no_paren)
    # single kana mixed (e.g. の) is a strong signal of a leftover
    frags += re.findall(r"[のをがはでにへとも][\u4E00-\u9FFF\u3041-\u309F]", t)
    return frags

DIALOGUE_OPEN = "「『（〈《“\""
def is_dialogue(text: str) -> bool:
    t = strip_tokens(text).lstrip("\u3000 ")
    return bool(t) and t[0] in "「『"

def count_dialogue_marks(text: str) -> int:
    t = strip_tokens(text)
    return t.count("「") + t.count("『")

def count_target_quotes(text: str) -> int:
    t = strip_tokens(text)
    return t.count("“") + t.count("「") + t.count("『") + t.count("‘") + t.count('"') // 2 + t.count("\"") % 2
