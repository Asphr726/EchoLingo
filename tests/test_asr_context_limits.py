"""Session-context bounds shared by the cloud ASR adapters."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

from echolingo.backends.asr import assemblyai, deepgram
from echolingo.backends.asr._context import CONTEXT_MAX_CHARS, bounded_context, hint_terms
from echolingo.models import AsrSessionConfig


def test_bounded_context_prefers_a_word_boundary() -> None:
    assert bounded_context(None) == ""
    assert bounded_context("  Topic: vision \n") == "Topic: vision"
    text = "saccade " * 200
    cut = bounded_context(text)
    assert len(cut) <= CONTEXT_MAX_CHARS and cut.endswith("saccade")
    # No boundary in the second half: a hard cut at the limit.
    assert bounded_context("x" * 1500) == "x" * CONTEXT_MAX_CHARS


def test_hint_terms_normalise_dedupe_and_skip() -> None:
    terms = ["  Béla   Julesz ", "béla julesz", "", None, "x" * 11, "texton", 42]
    assert hint_terms(terms, max_terms=10, max_chars=10) == ["texton", "42"]
    assert hint_terms(terms, max_terms=10, max_chars=20) == [
        "Béla Julesz",
        "x" * 11,
        "texton",
        "42",
    ]
    assert hint_terms(terms, max_terms=1, max_chars=20) == ["Béla Julesz"]
    assert hint_terms(None, max_terms=5, max_chars=5) == []


def test_hint_terms_total_byte_budget_keeps_order_and_later_short_terms() -> None:
    terms = ["aaaa", "視覚心理学", "bb", "cc"]  # 4, 15, 2, 2 UTF-8 bytes
    assert hint_terms(terms, max_terms=10, max_chars=50, max_total_bytes=8) == ["aaaa", "bb", "cc"]
    assert hint_terms(terms, max_terms=10, max_chars=50, max_total_bytes=19) == [
        "aaaa",
        "視覚心理学",
    ]
    # A skipped duplicate never consumes budget.
    assert hint_terms(["ab", "AB", "cd"], max_terms=10, max_chars=5, max_total_bytes=4) == [
        "ab",
        "cd",
    ]


def test_deepgram_keyterms_fit_the_url_and_token_budget() -> None:
    backend = deepgram.DeepgramAsrBackend(api_key="k", model="nova-3", language="ja")
    long_terms = tuple(f"視覚心理学の用語{index:02d}" for index in range(50))  # 26 bytes each
    backend.config = AsrSessionConfig("s", "ja", terms=long_terms)
    terms = parse_qs(urlsplit(backend.endpoint).query)["keyterm"]
    assert sum(len(term.encode("utf-8")) for term in terms) <= deepgram.MAX_KEYTERM_TOTAL_BYTES
    assert terms == list(long_terms[: len(terms)]) and 30 <= len(terms) < 50
    assert len(backend.endpoint) < 12_000


def test_assemblyai_keyterms_prompt_fits_the_url_budget() -> None:
    backend = assemblyai.AssemblyAiAsrBackend(api_key="k")
    long_terms = tuple(f"pre-attentive texture segregation term {index:03d}" for index in range(100))
    backend.config = AsrSessionConfig("s", "en", terms=long_terms)
    prompt = json.loads(parse_qs(urlsplit(backend.build_url()).query)["keyterms_prompt"][0])
    assert sum(len(term.encode("utf-8")) for term in prompt) <= assemblyai.MAX_KEYTERM_TOTAL_BYTES
    assert prompt == list(long_terms[: len(prompt)]) and 40 <= len(prompt) < 100
    assert len(backend.build_url()) < 8_000
