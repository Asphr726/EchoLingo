import asyncio

from echolingo.session_context import (
    TOPIC_PROMPT_MAX_CHARS,
    asr_scripts,
    build_asr_prompt,
    parse_session_context,
)


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


# ---------------------------------------------------------------------------
# Script filter: context the source language cannot use stays out of the ASR
# prompt and reaches translation only.
# ---------------------------------------------------------------------------

TITLE = "CS180 Intro to Computer Vision and Computational Photography: color"


def test_english_session_keeps_a_chinese_context_out_of_the_asr_prompt() -> None:
    parsed = parse_session_context("机器学习与神经网络", source_language="en")
    assert parsed.asr_prompt == ""
    assert parsed.topic == "" and parsed.hint_terms == ()
    assert parsed.translation_domain == "机器学习与神经网络"
    assert parsed.domain == "机器学习与神经网络"
    assert not parsed.empty
    # A same-script context is untouched.
    same = parse_session_context("image-to-image translation", source_language="en")
    assert same.asr_prompt == "Terms: image-to-image translation"
    assert same.translation_domain == ""


def test_no_source_language_or_auto_behaves_as_before() -> None:
    text = f"机器学习与神经网络\n{TITLE}\nTerms: saccade, 扫视"
    baseline = parse_session_context(text)
    assert parse_session_context(text, source_language=None) == baseline
    assert parse_session_context(text, source_language="auto") == baseline
    assert baseline.translation_domain == ""
    assert baseline.hint_terms == ("机器学习与神经网络", "saccade", "扫视")
    assert "机器学习与神经网络" in baseline.asr_prompt
    assert baseline.domain == baseline.topic == TITLE
    assert asr_scripts(None) is None and asr_scripts("") is None and asr_scripts("AUTO") is None


def test_chinese_session_keeps_han_and_latin_terms() -> None:
    parsed = parse_session_context(
        "计算机视觉课程：纹理感知与视觉搜索\n术语：纹理基元、CNN、ガウシアン、ResNet",
        source_language="zh-CN",
    )
    assert parsed.topic == "计算机视觉课程：纹理感知与视觉搜索"
    assert parsed.hint_terms == ("纹理基元", "CNN", "ResNet")
    assert parsed.translation_domain == "ガウシアン"
    assert "CNN" in parsed.asr_prompt and "ガウシアン" not in parsed.asr_prompt


def test_japanese_and_korean_sessions_keep_their_scripts() -> None:
    japanese = parse_session_context(
        "画像処理の講義です。\nガウシアン\nＣＮＮ\n색 공간", source_language="ja"
    )
    assert japanese.topic == "画像処理の講義です。"
    # Kana, Han and full-width Latin stay; Hangul moves.
    assert japanese.hint_terms == ("ガウシアン", "ＣＮＮ")
    assert japanese.translation_domain == "색 공간"
    korean = parse_session_context("색 공간\n色空間\nガウシアン", source_language="ko")
    assert korean.hint_terms == ("색 공간", "色空間")
    assert korean.translation_domain == "ガウシアン"


def test_mixed_lines_follow_the_majority_of_their_letters() -> None:
    parsed = parse_session_context(
        f"{TITLE}\n"
        "机器学习课程：纹理感知与视觉搜索的基本原理介绍\n"
        "Terms: Béla Julesz, Transformer 模型, 神经网络 CNN, 3.14\n"
        "扫视 = saccade",
        source_language="en",
    )
    assert parsed.topic == TITLE
    # A Latin majority (11 of 13 letters) stays, a Han majority (4 of 7)
    # moves, and a term without letters stays.
    assert parsed.hint_terms == ("Béla Julesz", "Transformer 模型", "3.14")
    assert parsed.translation_domain == "机器学习课程：纹理感知与视觉搜索的基本原理介绍 神经网络 CNN"
    # The glossary is unchanged; its Chinese source already reaches
    # translation as a pair and is not repeated in the domain.
    assert [(item.source, item.target) for item in parsed.glossary] == [("扫视", "saccade")]
    assert "扫视" not in parsed.asr_prompt
    assert parsed.domain == f"{TITLE} {parsed.translation_domain}"


def test_translation_domain_is_bounded_like_the_topic() -> None:
    lines = "\n".join(f"第{index}讲：纹理感知与视觉搜索的基本原理与实验方法。" for index in range(8))
    parsed = parse_session_context(f"{lines}\n{TITLE}", source_language="en")
    assert parsed.topic == TITLE
    assert parsed.translation_domain.startswith("第0讲")
    domain = parsed.domain
    assert len(domain) <= TOPIC_PROMPT_MAX_CHARS
    assert domain.startswith("CS180") and domain.endswith(parsed.translation_domain)
    many = "\n".join(f"第{index}讲：纹理感知与视觉搜索的基本原理与实验方法。" for index in range(50))
    parsed = parse_session_context(f"{many}\n{TITLE}", source_language="en")
    assert 0 < len(parsed.translation_domain) <= TOPIC_PROMPT_MAX_CHARS
    assert len(parsed.domain) <= TOPIC_PROMPT_MAX_CHARS


async def test_desktop_session_sends_off_script_context_to_translation_only(monkeypatch) -> None:
    from echolingo.service import session as session_module

    captured: dict = {}
    coordinator = session_module.StreamingTranslationCoordinator

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return coordinator(*args, **kwargs)

    monkeypatch.setattr(session_module, "StreamingTranslationCoordinator", capture)
    events: asyncio.Queue = asyncio.Queue()
    session = await session_module.DesktopInferenceSession.create(
        {
            "session_id": "script-filter",
            "source_language": "en",
            "target_language": "zh",
            "sample_rate_hz": 16_000,
            "channels": 1,
            "audio_profile": "raw",
            "inference_mode": "auto",
            "asr_provider": "mock",
            "translation_provider": "mock",
            "alignment_enabled": False,
            "session_context": f"机器学习与神经网络\n{TITLE}",
            "privacy": {"audio_upload_allowed": False, "transcript_upload_allowed": False},
        },
        events,
    )
    try:
        context = session.pipeline.session_context
        assert context.asr_prompt == TITLE
        assert context.translation_domain == "机器学习与神经网络"
        assert captured["domain"] == f"{TITLE} 机器学习与神经网络"
    finally:
        await session.close()
