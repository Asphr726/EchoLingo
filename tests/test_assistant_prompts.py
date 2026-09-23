"""Prompt rendering for notes, titles and context terms."""

from __future__ import annotations

import pytest

from echolingo.assistant import prompts
from echolingo.assistant.prompts import (
    MaterialText,
    SessionInfo,
    TranscriptLine,
    background_block,
    format_timestamp,
    format_transcript,
    language_name,
)


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("zh", "Simplified Chinese (简体中文)"),
        ("zh-CN", "Simplified Chinese (简体中文)"),
        ("zh_TW", "Traditional Chinese (繁體中文)"),
        ("en", "English"),
        ("en-US", "English"),
        ("ja", "Japanese (日本語)"),
        ("ko", "Korean (한국어)"),
        ("fr", 'the language with the code "fr"'),
        ("", "English"),
        (None, "English"),
    ],
)
def test_output_language_names(code, name) -> None:
    assert language_name(code) == name


def test_timestamps_are_minutes_and_seconds_beyond_an_hour() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(59_999) == "00:59"
    assert format_timestamp(75 * 60_000 + 4_000) == "75:04"
    assert format_timestamp(-5) == "00:00"
    lines = [
        TranscriptLine(61_000, 64_000, "  So   what makes textures\npop out? "),
        TranscriptLine(64_500, 66_000, "   "),
        TranscriptLine(3_605_000, 3_609_000, "Julesz called them textons."),
    ]
    assert format_transcript(lines) == (
        "[01:01] So what makes textures pop out?\n[60:05] Julesz called them textons."
    )


def test_notes_system_prompt_keeps_every_content_requirement() -> None:
    prompt = prompts.notes_system_prompt("zh")
    for required in (
        "expert note-taker",
        "lectures, talks, seminars, meetings",
        "fillers, false starts, repetitions",
        "self-corrections, digressions",
        "streaming speech recognizer",
        "punctuation is unreliable",
        '"psychotic movements" for "saccadic movements"',
        '"pre-ten division"',
        "gaps",
        "[mm:ss]",
        "NOT to compress",
        "Completeness beats brevity",
        "question posed to the audience",
        "assignment or",
        '"## "',
        "(mm:ss–mm:ss)",
        "Bold key terms",
        "chronological",
        '"(?)"',
        "Never invent content",
        "Do not summarize parts of the materials that were not",
        "$...$",
        "$$...$$",
        "Markdown tables only for tabular content",
        "Write everything in Simplified Chinese (简体中文)",
        "translation\nin parentheses at first mention".replace("\n", " "),
        "no preamble",
        "an overview of 2–4 sentences",
        '"## 关键术语"',
        '"## 问题与待办"',
        "session background",
        "glossary",
    ):
        assert required in prompt.replace("\n", " "), required


def test_notes_prompt_names_headings_for_unlisted_languages() -> None:
    prompt = prompts.notes_system_prompt("de")
    assert 'meaning "Key terms" in the language with the code "de"' in prompt
    assert '"## Key terms"' in prompts.notes_system_prompt("en")
    assert '"## 重要用語"' in prompts.notes_system_prompt("ja")
    assert '"## 핵심 용어"' in prompts.notes_system_prompt("ko")


def test_background_block_includes_context_and_glossary_without_comments() -> None:
    block = background_block(
        "CS180 lecture 5: texture perception",
        "# standing terms\nsaccade = 扫视\n\nJulesz\n",
    )
    assert "<session_background>" in block
    assert "CS180 lecture 5: texture perception" in block
    assert "<glossary>" in block
    assert "saccade = 扫视\nJulesz" in block
    assert "standing terms" not in block
    assert background_block("", "") == ""
    assert background_block(None, "# only a comment") == ""


def test_notes_user_message_orders_background_materials_and_transcript() -> None:
    session = SessionInfo(
        source_language="en",
        target_language="zh",
        duration_ms=3_845_000,
        context="Texture perception",
        glossary="texton = 纹理基元",
    )
    message = prompts.notes_user_message(
        session,
        "[00:01] hello",
        [MaterialText('slides "v2".pdf', "--- page 1 ---\nTextons")],
        "zh",
    )
    assert message.startswith("Session (spoken language: en, duration: 64:05).")
    assert message.index("<session_background>") < message.index("<reference_materials>")
    assert message.index("<reference_materials>") < message.index("<transcript>")
    assert '<material name="slides &quot;v2&quot;.pdf">' in message
    assert message.rstrip().endswith("Write the complete notes in Simplified Chinese (简体中文), following the rules.")


def test_window_message_carries_headings_and_continuation_rules() -> None:
    message = prompts.notes_window_message(
        SessionInfo(context="Vision"),
        "[14:00] next part",
        [],
        "en",
        index=2,
        total=5,
        start_ms=840_000,
        end_ms=1_710_000,
        headings_so_far=["Texture segmentation (00:00–06:10)", "Textons (06:10–13:55)"],
    )
    assert "part 2 of 5" in message
    assert "14:00–28:30" in message
    assert "- Texture segmentation (00:00–06:10)\n- Textons (06:10–13:55)" in message
    assert "Do not write the document title, the overview" in message
    assert '<transcript part="2/5">' in message
    first = prompts.notes_window_message(
        SessionInfo(), "[00:00] start", [], "en", index=1, total=3, start_ms=0, end_ms=1, headings_so_far=[]
    )
    assert "none yet" in first


def test_finishing_prompt_writes_only_title_overview_and_closing_sections() -> None:
    prompt = prompts.finishing_system_prompt("zh")
    assert '"# "' in prompt and "2–4 sentences" in prompt
    assert '"## 关键术语"' in prompt
    assert "Do not repeat or rewrite the sections" in prompt
    message = prompts.finishing_user_message(SessionInfo(), "## A (00:00–01:00)\n- x")
    assert "<sections>\n## A (00:00–01:00)\n- x\n</sections>" in message


@pytest.mark.parametrize(
    ("code", "limit"),
    [
        ("zh", "at most 20 characters"),
        ("ja", "at most 20 characters"),
        ("ko", "at most 25 characters"),
        ("en", "at most 8 words"),
        ("fr", "at most 8 words"),
    ],
)
def test_title_prompt_language_and_length(code, limit) -> None:
    prompt = prompts.title_system_prompt(code)
    assert limit in prompt
    assert language_name(code) in prompt
    assert "Output only the title" in prompt
    assert "no quotes" in prompt and "no trailing punctuation" in prompt
    assert "evenly sampled" in prompt


def test_title_user_message_includes_background_and_excerpts() -> None:
    message = prompts.title_user_message("[00:10] textons", "Lecture on texture", "saccade = 扫视")
    assert "Lecture on texture" in message
    assert "saccade = 扫视" in message
    assert "<transcript_excerpts>\n[00:10] textons\n</transcript_excerpts>" in message


def test_terms_prompt_asks_for_term_translation_lines() -> None:
    prompt = prompts.terms_system_prompt("zh")
    assert '"term = translation"' in prompt
    assert "at most 60" in prompt
    assert "Simplified Chinese" in prompt
    message = prompts.terms_user_message([MaterialText("deck.pptx", "Textons")], "en")
    assert message.startswith("The session will be spoken in English.")
    assert '<material name="deck.pptx">' in message


def test_prompt_version_is_recorded() -> None:
    assert prompts.PROMPT_VERSION.startswith("notes-v")
    assert prompts.PROBE_MESSAGES == [{"role": "user", "content": "Reply with OK"}]
