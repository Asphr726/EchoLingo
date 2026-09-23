from echolingo.session_context import build_asr_prompt, parse_session_context


def test_parses_topic_terms_and_glossary_pairs() -> None:
    parsed = parse_session_context(
        "CS180 Intro to Computer Vision and Computational Photography: color and image sensors\n"
        "Terms: Bayer mosaic, Foveon stacked sensor, CIE Lab\n"
        "demosaicing\n"
        "pre-attentive vision = 前注意视觉\n"
        "# a comment line\n"
        "saccade → 扫视\n",
        "Béla Julesz = 朱莱斯\ndemosaicing\n",
    )
    assert parsed.topic.startswith("CS180 Intro to Computer Vision")
    assert "Bayer mosaic" in parsed.hint_terms and "CIE Lab" in parsed.hint_terms
    assert "demosaicing" in parsed.hint_terms
    # Duplicates across both inputs collapse (case-insensitively).
    assert sum(1 for term in parsed.hint_terms if term.lower() == "demosaicing") == 1
    pairs = {(item.source, item.target) for item in parsed.glossary}
    assert ("pre-attentive vision", "前注意视觉") in pairs
    assert ("saccade", "扫视") in pairs
    assert ("Béla Julesz", "朱莱斯") in pairs
    # Glossary sources are recognition hints too.
    assert "saccade" in parsed.hint_terms
    assert "comment" not in parsed.asr_prompt
    assert parsed.asr_prompt.startswith("CS180") and "\nTerms: " in parsed.asr_prompt
    assert len(parsed.asr_prompt) <= 1000


def test_handles_chinese_and_japanese_lines() -> None:
    parsed = parse_session_context("计算机视觉课程：纹理感知与视觉搜索\n术语：纹理基元、前注意视觉、扫视\n注意 = attention")
    assert parsed.topic.startswith("计算机视觉课程")
    assert {"纹理基元", "前注意视觉", "扫视"} <= set(parsed.hint_terms)
    assert [(item.source, item.target) for item in parsed.glossary] == [("注意", "attention")]
    japanese = parse_session_context("画像処理の講義です。\nガウシアン\nエッジ検出")
    assert japanese.topic == "画像処理の講義です。"
    assert japanese.hint_terms == ("ガウシアン", "エッジ検出")


def test_sentences_with_colons_are_topic_not_glossary() -> None:
    parsed = parse_session_context("Today: we look at how texture perception works in the visual system.")
    assert parsed.glossary == ()
    assert parsed.topic.startswith("Today:")


def test_empty_input_and_prompt_budget() -> None:
    assert parse_session_context("", "").empty
    terms = [f"term{index}" for index in range(400)]
    prompt = build_asr_prompt("topic " * 200, terms)
    assert len(prompt) <= 1000 and prompt.count("Terms:") == 1
