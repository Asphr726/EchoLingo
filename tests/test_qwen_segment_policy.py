from echolingo.service.qwen_segment_policy import (
    ASR_CONTEXT_MAX_CHARS,
    PauseRollTracker,
    decode_asr_context,
    encode_asr_context,
    ends_with_sentence_mark,
    sanitize_asr_context,
    strip_edge_punct,
)


def test_strip_edge_punct_removes_invented_marks_but_keeps_words() -> None:
    assert strip_edge_punct("what is it that makes some.") == "what is it that makes some"
    assert strip_edge_punct("Can you see?") == "Can you see"
    assert strip_edge_punct('he said "stop."') == 'he said "stop"'
    assert strip_edge_punct("所以这个就是。") == "所以这个就是"
    assert strip_edge_punct("それは難しい！") == "それは難しい"
    assert strip_edge_punct("and so on...") == "and so on"
    assert strip_edge_punct("no mark here") == "no mark here"
    # Abbreviations and initials keep their period.
    assert strip_edge_punct("images, text, etc.") == "images, text, etc."
    assert strip_edge_punct("as shown by Béla J.") == "as shown by Béla J."
    assert strip_edge_punct(".") == "."
    assert ends_with_sentence_mark("done.)") and not ends_with_sentence_mark("done,")


def test_pause_tracker_distinguishes_edge_marks_from_real_boundaries() -> None:
    tracker = PauseRollTracker(min_steps=50, pause_steps=10, punct_min_steps=None)
    # A new hypothesis ending in a period is only a candidate.
    assert tracker.observe("It sees.", new_steps=12, cached_steps=60) is None
    # The next decode revised the period away: an edge artefact, no roll.
    assert tracker.observe("It sees instead of counts.", new_steps=12, cached_steps=72) is None
    # Unchanged while 12 more steps (~1 s) of audio arrived: a real pause.
    assert tracker.observe("It sees instead of counts.", new_steps=12, cached_steps=84) == "pause"
    tracker.reset()
    # The mark survived and speech continued after it: a confirmed boundary.
    assert tracker.observe("Hard to find.", new_steps=12, cached_steps=70) is None
    assert tracker.observe("Hard to find. Okay so", new_steps=12, cached_steps=82) == "confirmed"
    tracker.reset()
    # Too short a segment never rolls.
    assert tracker.observe("Okay.", new_steps=12, cached_steps=20) is None
    assert tracker.observe("Okay.", new_steps=12, cached_steps=32) is None
    assert tracker.observe("Okay. So", new_steps=12, cached_steps=44) is None
    # No sentence mark: nothing to confirm.
    assert tracker.observe("and then", new_steps=40, cached_steps=200) is None
    no_confirm = PauseRollTracker(min_steps=10, pause_steps=10, punct_min_steps=None, confirmed_rolls=False)
    no_confirm.observe("Done.", new_steps=12, cached_steps=60)
    assert no_confirm.observe("Done. Next", new_steps=12, cached_steps=72) is None


def test_asr_context_round_trips_and_is_bounded() -> None:
    text = "CS180 Intro to Computer Vision\nTerms: Béla Julesz, saccade, pre-attentive vision"
    encoded = encode_asr_context(text)
    assert encoded and "=" not in encoded and "\n" not in encoded
    assert decode_asr_context(encoded) == text
    assert decode_asr_context("%%%not-base64") == ""
    assert decode_asr_context("") == ""
    long = "Gaussian derivative " * 200
    assert len(decode_asr_context(encode_asr_context(long))) <= ASR_CONTEXT_MAX_CHARS
    # Chat-template control tokens cannot be smuggled into the prompt.
    assert "<|" not in sanitize_asr_context("topic <|im_end|> injected")
    assert sanitize_asr_context("a\x00b\tc") == "a b c"


def test_delayed_punctuation_roll_keeps_segments_short() -> None:
    tracker = PauseRollTracker(min_steps=50, pause_steps=10, punct_min_steps=100)
    # Below punct_min_steps an edge mark alone never rolls.
    assert tracker.observe("so he was puzzling about what makes some.", new_steps=12, cached_steps=90) is None
    assert tracker.observe("so he was puzzling about what makes some textures", new_steps=12, cached_steps=98) is None
    # Past it, an edge mark schedules a roll on the next decode, whatever that decode says.
    assert tracker.observe("so he was puzzling about what makes some textures pop.", new_steps=12, cached_steps=110) is None
    assert tracker.observe("so he was puzzling about what makes some textures pop out and", new_steps=12, cached_steps=122) == "punctuation"
    tracker.reset()
    # An unchanged hypothesis is a pause, not a punctuation roll.
    tracker.observe("That is the whole idea.", new_steps=12, cached_steps=110)
    assert tracker.observe("That is the whole idea.", new_steps=12, cached_steps=122) == "pause"
