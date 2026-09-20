from echolingo.models import TranscriptKind
from echolingo.streaming import (
    LectureSpeechPolicy,
    SentenceUnitSegmenter,
    UnitClosureRules,
    WlkEventMapper,
    parse_timestamp_ms,
    sanitize_committed_text,
)


def test_lecture_policy_uses_low_threshold_and_long_release() -> None:
    policy = LectureSpeechPolicy()
    assert not any(policy.observe(0.3, 10) for _ in range(9))
    assert policy.observe(0.3, 10)
    assert all(policy.observe(0.0, 10) for _ in range(99))
    assert not policy.observe(0.0, 10)


def test_segmenter_closes_units_at_sentence_boundaries_only_when_long_enough() -> None:
    segmenter = SentenceUnitSegmenter("en")
    assert segmenter.append("We propose a", start_ms=0, end_ms=500) == []
    units = segmenter.append("new framework.", start_ms=500, end_ms=1_200)
    assert len(units) == 1
    assert units[0].text == "We propose a new framework."
    assert units[0].start_ms == 0
    assert units[0].end_ms == 1_200
    assert segmenter.open_text == ""

    # "Yes." alone is too short to be a unit; it waits for more text.
    assert segmenter.append("Yes.", start_ms=1_200, end_ms=1_400) == []
    units = segmenter.append("It uses tactile feedback. And", start_ms=1_400, end_ms=3_000)
    assert [unit.text for unit in units] == ["Yes. It uses tactile feedback."]
    assert segmenter.open_text == "And"
    assert segmenter.flush().text == "And"
    assert segmenter.flush() is None


def test_segmenter_ignores_abbreviations_and_decimals() -> None:
    segmenter = SentenceUnitSegmenter("en")
    assert segmenter.append("Dr. Smith measured 3.5 volts", start_ms=0, end_ms=1_000) == []
    assert segmenter.append("on the e.g. probe. Next", start_ms=1_000, end_ms=2_000)[0].text == (
        "Dr. Smith measured 3.5 volts on the e.g. probe."
    )


def test_segmenter_falls_back_to_clause_length_and_duration_caps() -> None:
    segmenter = SentenceUnitSegmenter(
        "en", UnitClosureRules(sentence_min_chars=12, clause_min_chars=30, max_chars=60, max_duration_ms=5_000)
    )
    words = "alpha beta gamma delta epsilon zeta, eta theta iota kappa lambda mu nu xi"
    units = segmenter.append(words, start_ms=0, end_ms=1_000)
    assert [unit.text for unit in units] == ["alpha beta gamma delta epsilon zeta,"]
    assert segmenter.open_text == "eta theta iota kappa lambda mu nu xi"

    slow = SentenceUnitSegmenter("en", UnitClosureRules(max_duration_ms=5_000))
    assert slow.append("no punctuation here", start_ms=0, end_ms=1_000) == []
    units = slow.append("still talking", start_ms=1_000, end_ms=5_500)
    assert [unit.text for unit in units] == ["no punctuation here still talking"]

    # A cap prefers the last clause boundary over a mid-phrase cut.
    capped = SentenceUnitSegmenter("en", UnitClosureRules(max_duration_ms=5_000))
    assert capped.append("of blue, and then purples, and it takes", start_ms=0, end_ms=2_000) == []
    units = capped.append("back the", start_ms=2_000, end_ms=5_500)
    assert [unit.text for unit in units] == ["of blue, and then purples,"]
    assert capped.open_text == "and it takes back the"


def test_segmenter_joins_cjk_without_spaces() -> None:
    segmenter = SentenceUnitSegmenter("zh")
    assert segmenter.append("我们提出了一个", start_ms=0, end_ms=800) == []
    units = segmenter.append("新的框架。然后", start_ms=800, end_ms=1_600)
    assert [unit.text for unit in units] == ["我们提出了一个新的框架。"]
    assert segmenter.open_text == "然后"


def test_wlk_partial_carries_open_and_unstable_text_and_commits_units() -> None:
    clock = {"now_ns": 1_250_000_000}
    mapper = WlkEventMapper(
        "session", "en", "qwen", "bounded_recompute", clock_ns=lambda: clock["now_ns"]
    )
    mapper.note_audio_cursor(0, 0)
    first = mapper.map_message({"lines": [], "buffer_transcription": "hello"}, 500)
    assert [event.kind for event in first] == [TranscriptKind.PARTIAL]
    assert first[0].revision_id == 1
    assert first[0].stable_text == "" and first[0].unstable_text == "hello"

    update = mapper.map_message(
        {
            "lines": [{"text": "hello world", "start": "0:00:00", "end": "0:00:01"}],
            "buffer_transcription": "next",
        },
        1250,
    )
    # "hello world" has no sentence boundary, so it stays open (dark) rather
    # than becoming its own row.
    assert [event.kind for event in update] == [TranscriptKind.PARTIAL]
    assert update[0].stable_text == "hello world"
    assert update[0].unstable_text == "next"
    assert update[0].text == "hello world next"
    assert update[0].committed_text == ""

    repeated = mapper.map_message(
        {
            "lines": [{"text": "hello world", "start": "0:00:00", "end": "0:00:01"}],
            "buffer_transcription": "next",
        },
        1300,
    )
    assert repeated == []

    clock["now_ns"] = 2_250_000_000
    closed = mapper.map_message(
        {
            "lines": [{"text": "hello world, this is fine.", "start": "0:00:00", "end": "0:00:02"}],
            "buffer_transcription": "then",
        },
        2250,
    )
    assert [event.kind for event in closed] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]
    assert closed[0].text == "hello world, this is fine."
    assert closed[0].committed_text == "hello world, this is fine."
    assert closed[0].start_ms == 0 and closed[0].end_ms == 2_000
    assert closed[0].commit_latency_ms == 250
    assert closed[1].stable_text == "" and closed[1].unstable_text == "then"


def test_wlk_reports_first_token_latency_only_once() -> None:
    now_ns = 1_000_000_000
    mapper = WlkEventMapper(
        "session", "en", "qwen", "bounded_recompute", clock_ns=lambda: now_ns
    )
    mapper.note_speech_onset(500_000_000)

    first = mapper.map_message({"lines": [], "buffer_transcription": "hello"}, 500)
    assert first[0].first_token_latency_ms == 500

    now_ns = 2_000_000_000
    second = mapper.map_message(
        {"lines": [], "buffer_transcription": "hello world"}, 1_500
    )
    assert second[0].first_token_latency_ms is None
    assert 0.0 < second[0].stability < 1.0


def test_wlk_new_upstream_line_closes_the_open_unit() -> None:
    mapper = WlkEventMapper("session", "en", "qwen", "streaming")
    events = mapper.map_message(
        {
            "lines": [
                {"text": "first fragment", "start": 0, "end": 1},
                {"text": "second fragment", "start": 3, "end": 4},
            ],
            "buffer_transcription": "",
        },
        4_000,
    )
    assert [event.kind for event in events] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]
    assert events[0].text == "first fragment"
    assert events[0].revision_id == 1
    assert events[1].stable_text == "second fragment"


def test_wlk_grows_an_existing_line_and_flushes_the_tail_at_finish() -> None:
    """WLK grows its last full-state line instead of appending a new line."""
    mapper = WlkEventMapper("session", "en", "qwen", "streaming")
    first = mapper.map_message(
        {
            "lines": [{"text": "The first committed sentence.", "start": 0, "end": 2}],
            "buffer_transcription": "The editable tail",
        },
        2_500,
    )
    assert [event.kind for event in first] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]

    grown = mapper.map_message(
        {
            "lines": [
                {
                    "text": "The first committed sentence. The second committed sentence.",
                    "start": 0,
                    "end": 5,
                }
            ],
            "buffer_transcription": "A new editable tail",
        },
        5_500,
    )
    assert [event.kind for event in grown] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]
    assert grown[0].text == "The second committed sentence."
    assert grown[0].committed_text == (
        "The first committed sentence. The second committed sentence."
    )
    assert grown[0].start_ms == 2_000 and grown[0].end_ms == 5_000

    flushed = mapper.flush_events(6_000)
    assert [event.kind for event in flushed] == [TranscriptKind.STABLE, TranscriptKind.FINAL]
    assert flushed[0].text == "A new editable tail"
    assert flushed[1].text.endswith("A new editable tail")
    assert flushed[1].committed_text == flushed[1].text


def test_wlk_reanchors_after_an_incompatible_rewrite() -> None:
    mapper = WlkEventMapper("session", "en", "qwen", "streaming")
    mapper.map_message(
        {"lines": [{"text": "we went to the store.", "start": 0, "end": 1}], "buffer_transcription": ""},
        1_000,
    )
    rewritten = mapper.map_message(
        {
            "lines": [{"text": "we want to the store. Then we went home.", "start": 0, "end": 2}],
            "buffer_transcription": "",
        },
        2_000,
    )
    assert mapper.dropped_rewrites == 1
    assert [event.kind for event in rewritten] == [TranscriptKind.STABLE]
    assert rewritten[0].text == "Then we went home."
    assert rewritten[0].committed_text == "we went to the store. Then we went home."


def test_wlk_promotes_agreed_cjk_sentences_before_upstream_commits() -> None:
    """Upstream cannot commit CJK mid-segment; a finished, agreed sentence can."""
    mapper = WlkEventMapper("session", "zh", "qwen", "streaming")
    first = mapper.map_message({"lines": [], "buffer_transcription": "我们提出了一个新的框架。它使用"}, 1_000)
    assert [event.kind for event in first] == [TranscriptKind.PARTIAL]
    assert first[0].stable_text == ""

    # Agreement covers the whole previous buffer (15 chars) but the sentence
    # end sits inside the 8-char hold-back, so nothing is promoted yet.
    second = mapper.map_message(
        {"lines": [], "buffer_transcription": "我们提出了一个新的框架。它使用视觉触觉反馈"}, 2_000
    )
    assert [event.kind for event in second] == [TranscriptKind.PARTIAL]
    assert second[0].stable_text == ""
    assert second[0].unstable_text == "我们提出了一个新的框架。它使用视觉触觉反馈"

    third = mapper.map_message(
        {"lines": [], "buffer_transcription": "我们提出了一个新的框架。它使用视觉触觉反馈来操作物体"}, 3_000
    )
    # Agreement 21 chars - 8 hold-back = 13 -> the finished sentence closes as a unit.
    assert [event.kind for event in third] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]
    assert third[0].text == "我们提出了一个新的框架。"
    assert third[0].committed_text == "我们提出了一个新的框架。"
    assert third[1].stable_text == ""
    assert third[1].unstable_text == "它使用视觉触觉反馈来操作物体"

    # The upstream segment roll finally commits the whole hypothesis; only the
    # remainder beyond the promoted prefix is appended.
    rolled = mapper.map_message(
        {
            "lines": [{"text": "我们提出了一个新的框架。它使用视觉触觉反馈来操作物体。", "start": 0, "end": 4}],
            "buffer_transcription": "然后",
        },
        4_500,
    )
    assert [event.kind for event in rolled] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]
    assert rolled[0].text == "它使用视觉触觉反馈来操作物体。"
    assert rolled[0].committed_text == "我们提出了一个新的框架。它使用视觉触觉反馈来操作物体。"
    assert rolled[1].stable_text == "" and rolled[1].unstable_text == "然后"


def test_parse_wlk_timestamp() -> None:
    assert parse_timestamp_ms("1:02:03.5") == 3_723_500


def test_sanitize_drops_symbol_runs_and_wrong_script_hallucinations() -> None:
    assert sanitize_committed_text("今年は # # # # # # # の中", "ja") == "今年は  の中"
    english = "Human Rights Watch is a non-profit organization that works to protect people."
    assert sanitize_committed_text(english, "ja") == ""
    assert sanitize_committed_text(" " + english, "ja") == " "
    assert sanitize_committed_text("今年は戦後八十年です。" + english + "と楽", "ja").replace(" ", "") == "今年は戦後八十年です。と楽"
    assert sanitize_committed_text("これまでの日本の歩みを振り返り", "ja") == "これまでの日本の歩みを振り返り"
    # Short foreign fragments (names, acronyms) are kept.
    assert sanitize_committed_text("RGBの値", "ja") == "RGBの値"
    assert sanitize_committed_text("我们使用 CCD 传感器。", "zh") == "我们使用 CCD 传感器。"
    assert sanitize_committed_text("这是一个完全错误的中文幻觉句子。", "en") == ""
    assert sanitize_committed_text("Look at that line.", "en") == "Look at that line."


def test_wlk_mapper_never_commits_wrong_script_hallucination() -> None:
    mapper = WlkEventMapper("session", "ja", "qwen", "streaming", character_hold_back=0)
    events = mapper.map_message(
        {
            "lines": [{"text": "今年は戦後八十年です。Human Rights Watch is a non-profit organization that works.", "start": 0, "end": 6}],
            "buffer_transcription": "",
        },
        6_000,
    )
    assert [event.kind for event in events] == [TranscriptKind.PARTIAL]
    assert events[0].stable_text == "今年は戦後八十年です。"
    flushed = mapper.flush_events(6_000)
    assert [event.text for event in flushed if event.kind == TranscriptKind.STABLE] == ["今年は戦後八十年です。"]
    assert "Human" not in flushed[-1].text
