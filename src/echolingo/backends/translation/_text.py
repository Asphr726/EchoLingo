"""Text hygiene shared by LLM-based translation adapters."""

from __future__ import annotations

import re

# Full language names as the Hy-MT2 model card requires: Chinese names inside
# Chinese prompts, English names inside English prompts.
LANGUAGE_NAMES = {
    "zh": ("中文", "Chinese"),
    "en": ("英语", "English"),
    "ja": ("日语", "Japanese"),
    "ko": ("韩语", "Korean"),
    "fr": ("法语", "French"),
    "de": ("德语", "German"),
    "es": ("西班牙语", "Spanish"),
    "ru": ("俄语", "Russian"),
}

_ECHO_PREFIXES = (
    re.compile(r"^(好的[，,]?\s*)?(这是|以下是)?(翻译|译文)(结果|内容)?[：:]\s*", re.IGNORECASE),
    re.compile(r"^(sure[,!]?\s*)?(here('s| is) (the |your )?translation)[:：]\s*", re.IGNORECASE),
    re.compile(r"^(translation|translated (result|text))[:：]\s*", re.IGNORECASE),
)
_ECHO_FRAGMENTS = (
    "仅返回翻译内容",
    "只需要输出翻译后的结果",
    "不要额外解释",
    "注意只需要输出翻译后的结果",
    "only output the translated result",
    "without any additional explanation",
)


def strip_instruction_echo(text: str) -> str:
    """Remove chatter a small model sometimes copies from the instruction."""
    stripped = text.lstrip()
    for pattern in _ECHO_PREFIXES:
        stripped = pattern.sub("", stripped, count=1)
    for fragment in _ECHO_FRAGMENTS:
        index = stripped.lower().find(fragment.lower())
        if index >= 0:
            stripped = (stripped[:index] + stripped[index + len(fragment):]).strip("：:。. \n")
    return stripped


def repeated_tail(text: str, *, repeats: int = 4, max_unit: int = 48) -> str | None:
    """Return the unit that repeats at the tail of ``text`` at least ``repeats`` times."""
    stripped = text.rstrip()
    for size in range(2, max_unit + 1):
        if len(stripped) < size * repeats:
            break
        unit = stripped[-size:]
        if stripped.endswith(unit * repeats):
            return unit
    return None


def language_name(code: str, *, chinese_prompt: bool) -> str:
    names = LANGUAGE_NAMES.get(code.lower())
    if names is None:
        return code
    return names[0] if chinese_prompt else names[1]


def strip_wrapping_quotes(text: str) -> str:
    """Chat models sometimes quote the whole translation; drop matching quotes."""
    stripped = text.strip()
    pairs = (('"', '"'), ("“", "”"), ("「", "」"), ("'", "'"))
    for left, right in pairs:
        if len(stripped) > 2 and stripped.startswith(left) and stripped.endswith(right):
            inner = stripped[1:-1]
            if left not in inner and right not in inner:
                return inner.strip()
    return stripped
