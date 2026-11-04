from epub_translator.validator import check_block, find_duplicates
from epub_translator.textutil import residual_japanese


def test_ok():
    assert check_block("「先輩、本当ですか？」", "“선배, 정말이에요?”")[0] == "ok"


def test_missing():
    assert check_block("猫が鳴いた。", None) == ("error", ["missing"])
    assert check_block("猫が鳴いた。", "  ")[0] == "error"


def test_residual_japanese():
    sev, flags = check_block("彼は走った。", "그는 走ったした.")
    assert sev == "error" and any(f.startswith("residual-japanese") for f in flags)
    assert residual_japanese("그는 달렸다.") == []
    assert residual_japanese("신기루(蜃気楼)") == []    # hanja gloss in parentheses tolerated


def test_short_alone_is_warning_not_error():
    src = "　その日、僕はずっと窓の外の雨を眺めながら、彼女のことを考えていた。"
    sev, flags = check_block(src, "그날 비를 봤다.")
    assert sev == "warn" and "too-short" in flags


def test_short_with_semantic_signal_is_error():
    src = "「明日の朝、七時に駅前で待っているから、絶対に遅れないでね」"
    sev, flags = check_block(src, "내일 봐.")
    assert sev == "error" and "dialogue-lost" in flags


def test_tag_mismatch_error():
    assert check_block("彼は<t1>走った</t1>。", "그는 달렸다.")[0] == "error"


def test_duplicates():
    rows = [{"block_id": "a", "source": "一つ目の文章です。とても長い。", "target": "같은 번역 문장이 여기에 있습니다."},
            {"block_id": "b", "source": "二つ目の全く別の文章です。", "target": "같은 번역 문장이 여기에 있습니다."}]
    assert find_duplicates(rows) == {"b": "a"}


def test_repair_tokens_drops_invented_tokens():
    from epub_translator.validator import repair_tokens
    assert repair_tokens("花城は笑った。", "<t3>하나시로</t3>는 웃었다.") == "하나시로는 웃었다."
    assert repair_tokens("a<t1>b</t1>", "x<t1>y</t1><t9/>") == "x<t1>y</t1>"


def test_english_source_checks():
    from epub_translator.validator import check_block
    assert check_block("“Did you miss me?”", "“내가 보고 싶었어?”")[0] == "ok"
    assert "no-hangul" in check_block("“Did you miss me?”", "“Did you miss me?”")[1]
    assert any(f.startswith("residual-latin") for f in
               check_block("He said she was not here at all.", "그는 she was not here at all 라고 했다.")[1])


def test_flatten_tokens_fallback():
    from epub_translator.validator import flatten_tokens
    assert flatten_tokens("<t1>a </t1><t2>b</t2>", "<t2>나</t1> 가") == "<t1>나 가</t1>"
    assert flatten_tokens("<t1>a</t1><t2/>", "가") is None  # never drop images/line breaks
