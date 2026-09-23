"""Assistant orchestration with a fake model: notes (single call and
windowed), titles, context import, probe, consent gate and cancellation."""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from echolingo.assistant import output, service, terms, windows
from echolingo.assistant.attachments import AttachmentSegment, ExtractedAttachment
from echolingo.assistant.llm import AssistantError, ChatDelta, ChatResult
from echolingo.assistant.prompts import PROMPT_VERSION, TranscriptLine
from echolingo.assistant.service import AssistantService, TaskEvents
from echolingo.errors import AuthenticationError
from echolingo.session_context import parse_session_context

CONSENT = {"transcript_upload_allowed": True}
LLM = {"group": "dashscope", "model": "qwen-plus"}


class FakeClient:
    def __init__(
        self,
        *,
        streams: list[list[str]] | None = None,
        completions: list[str] | None = None,
        error: Exception | None = None,
        block: asyncio.Event | None = None,
    ) -> None:
        self.streams = list(streams or [])
        self.completions = list(completions or [])
        self.error = error
        self.block = block
        self.calls: list[tuple[str, list[dict[str, str]], dict[str, Any]]] = []
        self.provider = "dashscope"
        self.model = "qwen-plus"
        self.endpoint = SimpleNamespace(display_name="Qwen (Alibaba Model Studio)")
        self.closed = False

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True

    async def stream_text(self, messages, **kwargs):
        self.calls.append(("stream", messages, kwargs))
        if self.error is not None:
            raise self.error
        chunks = self.streams.pop(0)
        for chunk in chunks:
            if self.block is not None:
                await self.block.wait()
            yield ChatDelta(text=chunk)
        yield ChatDelta(finish_reason="stop", usage={"prompt_tokens": 1000, "completion_tokens": 200})

    async def complete(self, messages, **kwargs) -> ChatResult:
        self.calls.append(("complete", messages, kwargs))
        if self.error is not None:
            raise self.error
        return ChatResult(
            text=self.completions.pop(0),
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )

    async def complete_streamed(self, messages, **kwargs) -> ChatResult:
        self.calls.append(("complete_streamed", messages, kwargs))
        if self.error is not None:
            raise self.error
        return ChatResult(
            text=self.completions.pop(0),
            finish_reason="stop",
            usage={"prompt_tokens": 300, "completion_tokens": 100},
        )


def make_service(client: FakeClient) -> tuple[AssistantService, list[Any]]:
    seen: list[Any] = []

    def factory(llm, environ):
        seen.append(llm)
        return client

    return AssistantService(environ={}, client_factory=factory), seen


async def run_task(assistant: AssistantService, task: str, payload: dict[str, Any]) -> list[dict]:
    events: list[dict] = []
    await assistant.handle(
        {"request_id": "req-1", "task": task, "payload": payload}, events.append
    )
    return events


def result_of(events: list[dict]) -> dict:
    results = [event for event in events if event["type"] == "assistant_result"]
    assert len(results) == 1, events
    assert results[0]["payload"]["request_id"] == "req-1"
    return results[0]["payload"]


def units(count: int, *, step_ms: int = 30_000, words: str = "the lecturer explains textons") -> list[dict]:
    return [
        {"start_ms": index * step_ms, "end_ms": index * step_ms + step_ms - 500, "text": f"{words} {index}."}
        for index in range(count)
    ]


def notes_payload(transcript: list[dict], **extra) -> dict:
    payload = {
        "llm": LLM,
        "consent": CONSENT,
        "output_language": "zh",
        "session": {
            "id": "s1",
            "title": "en → zh lecture",
            "source_language": "en",
            "target_language": "zh",
            "started_at": "2026-09-22T10:00:00Z",
            "duration_ms": transcript[-1]["end_ms"] if transcript else 0,
            "context": "Topic: texture perception",
        },
        "transcript": transcript,
        "attachments": [],
    }
    payload.update(extra)
    return payload


def rebuild_like_the_shell(events: list[dict]) -> str:
    """Apply deltas as the desktop shell does: append-only, in seq order."""
    text = ""
    last_seq = 0
    for event in events:
        payload = event["payload"]
        if event["type"] == "assistant_delta":
            assert payload["seq"] == last_seq + 1
            last_seq = payload["seq"]
            text += payload["text"]
    return text


# ------------------------------------------------------------------- notes


async def test_short_session_notes_stream_in_one_call() -> None:
    client = FakeClient(
        streams=[["Sure, here are your notes.\n\n", "# 纹理感知\n\n概述。\n\n", "## 纹理基元 (00:00–05:00)\n- 要点\n"]]
    )
    assistant, seen = make_service(client)
    events = await run_task(assistant, "notes", notes_payload(units(10)))
    result = result_of(events)
    assert result["ok"] is True, result
    notes = result["result"]
    assert notes["markdown"] == "# 纹理感知\n\n概述。\n\n## 纹理基元 (00:00–05:00)\n- 要点"
    assert notes["title"] == "纹理感知"
    assert notes["provider"] == "dashscope" and notes["model"] == "qwen-plus"
    assert notes["usage"] == {"prompt_tokens": 1000, "completion_tokens": 200}
    assert notes["attachments"] == []
    assert notes["prompt_version"] == PROMPT_VERSION
    assert notes["source_chars"] == sum(len(unit["text"]) for unit in units(10))
    assert seen == [LLM]
    assert client.closed

    stages = [event["payload"]["stage"] for event in events if event["type"] == "assistant_progress"]
    assert stages == ["reading", "writing"]
    # The preamble never reaches the stream; the shell's text equals the result.
    assert rebuild_like_the_shell(events).strip() == notes["markdown"]

    kind, messages, kwargs = client.calls[0]
    assert kind == "stream"
    assert kwargs == {"max_tokens": 8192, "temperature": 0.3}
    assert "Simplified Chinese" in messages[0]["content"]
    user = messages[1]["content"]
    assert "[00:00] the lecturer explains textons 0." in user
    assert "[04:30] the lecturer explains textons 9." in user
    assert "Topic: texture perception" in user


async def test_long_session_notes_are_written_window_by_window() -> None:
    transcript = units(128, step_ms=30_000)  # 64 minutes
    plan = windows.plan_windows(service.parse_transcript(transcript))
    total = len(plan)
    assert total == 5
    streams = [
        [f"# Stray title\n\nStray overview.\n\n## Part {index} (00:00–01:00)\n", f"- point {index}\n"]
        for index in range(1, total + 1)
    ]
    finishing = (
        "# Texture Perception and Textons\n\nA lecture about textons.\n\n"
        "## 关键术语\n- **texton** — 纹理基元\n\n"
        "## Part 2 (00:00–01:00)\n- repeated section that must be dropped\n"
    )
    client = FakeClient(streams=streams, completions=[finishing])
    assistant, _seen = make_service(client)
    events = await run_task(assistant, "notes", notes_payload(transcript))
    result = result_of(events)
    assert result["ok"] is True, result
    markdown = result["result"]["markdown"]
    assert markdown.startswith("# Texture Perception and Textons\n\nA lecture about textons.\n\n## Part 1")
    positions = [markdown.index(f"## Part {index} ") for index in range(1, total + 1)]
    assert positions == sorted(positions)
    assert markdown.count("## Part 2 ") == 1
    assert "Stray title" not in markdown and "Stray overview" not in markdown
    assert markdown.rstrip().endswith("- **texton** — 纹理基元")
    assert result["result"]["title"] == "Texture Perception and Textons"
    assert result["result"]["usage"] == {
        "prompt_tokens": 1000 * total + 300,
        "completion_tokens": 200 * total + 100,
    }

    sections = [
        event["payload"]["detail"]
        for event in events
        if event["type"] == "assistant_progress" and event["payload"]["stage"] == "section"
    ]
    assert [detail["index"] for detail in sections] == list(range(1, total + 1))
    assert all(detail["total"] == total for detail in sections)
    assert sections[0]["start_ms"] == 0
    assert sections[-1]["end_ms"] == transcript[-1]["end_ms"]
    for previous, current in zip(sections, sections[1:]):
        assert previous["end_ms"] <= current["start_ms"]
        assert 12 * 60_000 <= current["start_ms"] - previous["start_ms"] <= 15.5 * 60_000

    # Each part sees the headings written before it and is told to continue.
    window_calls = [call for call in client.calls if call[0] == "stream"]
    assert len(window_calls) == total
    third_user = window_calls[2][1][1]["content"]
    assert "- Part 1 (00:00–01:00)\n- Part 2 (00:00–01:00)" in third_user
    assert "part 3 of 5" in third_user
    finishing_call = client.calls[-1]
    assert finishing_call[0] == "complete_streamed"
    assert "## Part 5" in finishing_call[1][1]["content"]

    # Deltas stay append-only: the streamed text is the sections plus the
    # closing sections; the header reaches the shell through the result.
    progress = [event for event in events if event["type"] == "assistant_progress"]
    assert progress[-1]["payload"] == {"request_id": "req-1", "stage": "finishing", "detail": {}}
    streamed = rebuild_like_the_shell(events)
    assert streamed.startswith("## Part 1")
    assert "# Texture Perception and Textons" not in streamed
    header = "# Texture Perception and Textons\n\nA lecture about textons.\n\n"
    assert header + streamed.strip() == markdown


async def test_fenced_answers_lose_their_closing_fence() -> None:
    # A single-call answer wrapped in a code fence after a preamble.
    client = FakeClient(
        streams=[["Here are the notes:\n\n```mark", "down\n# 纹理感知\n\n概述。\n\n", "## A (00:00–05:00)\n- x\n`", "``\n"]]
    )
    assistant, _seen = make_service(client)
    events = await run_task(assistant, "notes", notes_payload(units(10)))
    markdown = result_of(events)["result"]["markdown"]
    assert markdown == "# 纹理感知\n\n概述。\n\n## A (00:00–05:00)\n- x"
    assert "```" not in rebuild_like_the_shell(events)

    # In a long session a part's closing fence would swallow every later part.
    transcript = units(128, step_ms=30_000)
    total = len(windows.plan_windows(service.parse_transcript(transcript)))
    streams = [
        ["```markdown\n", f"## Part {index} (00:00–01:00)\n```python\nx = {index}\n```\n", "- done\n```"]
        for index in range(1, total + 1)
    ]
    finishing = "Sure!\n```markdown\n# Long Session\n\nOverview.\n\n## 关键术语\n- **x** — y\n```\n"
    client = FakeClient(streams=streams, completions=[finishing])
    assistant, _seen = make_service(client)
    markdown = result_of(await run_task(assistant, "notes", notes_payload(transcript)))["result"]["markdown"]
    assert markdown.startswith("# Long Session\n\nOverview.\n\n## Part 1 (00:00–01:00)\n```python\nx = 1\n```\n- done")
    # Only the fenced code inside each part remains: two fence lines per part.
    assert markdown.count("```") == 2 * total
    assert markdown.rstrip().endswith("- **x** — y")


async def test_failed_finishing_call_keeps_the_written_sections() -> None:
    transcript = units(128, step_ms=30_000)
    total = len(windows.plan_windows(service.parse_transcript(transcript)))

    class FinishingFails(FakeClient):
        async def complete_streamed(self, messages, **kwargs) -> ChatResult:
            self.calls.append(("complete_streamed", messages, kwargs))
            raise AssistantError("provider_timeout", "Qwen stopped responding.")

    client = FinishingFails(
        streams=[[f"## Part {index} (00:00–01:00)\n- point {index}\n"] for index in range(1, total + 1)]
    )
    assistant, _seen = make_service(client)
    result = result_of(await run_task(assistant, "notes", notes_payload(transcript)))
    assert result["ok"] is True, result
    notes = result["result"]
    assert notes["markdown"].startswith("## Part 1 (00:00–01:00)")
    assert notes["markdown"].count("## Part ") == total
    assert notes["title"] == ""


async def test_context_import_with_auto_source_language_names_no_language(tmp_path) -> None:
    deck = make_deck(tmp_path / "deck.pptx")
    client = FakeClient(completions=["saccade = 扫视"])
    assistant, _seen = make_service(client)
    payload = {
        "attachments": [{"id": "d1", "path": str(deck), "name": "deck.pptx"}],
        "source_language": "auto",
        "target_language": "zh",
        "use_llm": True,
        "llm": LLM,
        "consent": CONSENT,
    }
    result = result_of(await run_task(assistant, "context", payload))
    assert result["ok"] is True, result
    user = client.calls[0][1][1]["content"]
    assert "auto" not in user
    assert "Simplified Chinese" in client.calls[0][1][0]["content"]


async def test_notes_with_attachments_report_extraction(tmp_path) -> None:
    handout = tmp_path / "handout.txt"
    handout.write_text("Textons were proposed by Bela Julesz.", encoding="utf-8")
    client = FakeClient(streams=[["# Notes\n\nOverview.\n"]])
    assistant, _seen = make_service(client)
    payload = notes_payload(
        units(4),
        attachments=[{"id": "att-1", "path": str(handout), "name": "handout.txt"}],
    )
    events = await run_task(assistant, "notes", payload)
    result = result_of(events)["result"]
    assert result["attachments"] == [
        {"id": "att-1", "name": "handout.txt", "pages": None, "chars": 37, "truncated": False, "warning": None}
    ]
    assert events[0]["payload"]["stage"] == "extracting"
    user = client.calls[0][1][1]["content"]
    assert '<material name="handout.txt">' in user and "Bela Julesz" in user


@pytest.mark.parametrize("task", ["notes", "title"])
@pytest.mark.parametrize(
    "consent", [None, {}, {"transcript_upload_allowed": False}, {"transcript_upload_allowed": "yes"}]
)
async def test_transcript_tasks_require_explicit_consent(task, consent) -> None:
    client = FakeClient(streams=[["# x"]], completions=["x"])
    assistant, seen = make_service(client)
    payload = notes_payload(units(3))
    if consent is None:
        payload.pop("consent")
    else:
        payload["consent"] = consent
    result = result_of(await run_task(assistant, task, payload))
    assert result["ok"] is False
    assert result["result"]["code"] == "privacy_policy_denied"
    assert seen == [] and client.calls == []


@pytest.mark.parametrize(
    ("command", "code"),
    [
        ({"request_id": "req-1", "task": "summarize", "payload": {}}, "invalid_request"),
        ({"request_id": "req-1", "task": "notes", "payload": "nope"}, "invalid_request"),
        ({"request_id": "req-1", "task": "notes", "payload": {"consent": CONSENT, "transcript": "text"}}, "invalid_request"),
        (
            {"request_id": "req-1", "task": "notes", "payload": {"consent": CONSENT, "transcript": [{"text": "x", "start_ms": "0"}]}},
            "invalid_request",
        ),
        (
            {"request_id": "req-1", "task": "notes", "payload": {"consent": CONSENT, "transcript": [], "llm": LLM}},
            "empty_transcript",
        ),
        (
            {"request_id": "req-1", "task": "notes", "payload": {"consent": CONSENT, "output_language": "中文", "transcript": units(2)}},
            "invalid_request",
        ),
        (
            {"request_id": "req-1", "task": "notes", "payload": {"consent": CONSENT, "transcript": units(2), "attachments": [{"id": "1"}]}},
            "invalid_request",
        ),
    ],
)
async def test_malformed_requests_end_in_one_failed_result(command, code) -> None:
    assistant, _seen = make_service(FakeClient())
    events: list[dict] = []
    await assistant.handle(command, events.append)
    result = result_of(events)
    assert result["ok"] is False
    assert result["result"]["code"] == code
    assert result["result"]["message"]


async def test_request_without_an_id_still_gets_a_result() -> None:
    assistant, _seen = make_service(FakeClient())
    events: list[dict] = []
    await assistant.handle({"task": "probe"}, events.append)
    await assistant.handle(["not", "an", "object"], events.append)
    assert [event["payload"]["ok"] for event in events] == [False, False]
    assert all(event["payload"]["request_id"] == "" for event in events)


async def test_provider_errors_map_to_codes_without_raising() -> None:
    client = FakeClient(error=AuthenticationError("Qwen rejected the API key (HTTP 401)."))
    assistant, _seen = make_service(client)
    result = result_of(await run_task(assistant, "notes", notes_payload(units(3))))
    assert result == {
        "request_id": "req-1",
        "ok": False,
        "result": {"code": "authentication_failed", "message": "Qwen rejected the API key (HTTP 401)."},
    }
    assert client.closed

    unexpected = FakeClient(error=RuntimeError("secret transcript text"))
    assistant, _seen = make_service(unexpected)
    result = result_of(await run_task(assistant, "notes", notes_payload(units(3))))
    assert result["result"]["code"] == "internal_error"
    assert "secret transcript text" not in result["result"]["message"]


async def test_not_configured_provider_fails_cleanly() -> None:
    def factory(llm, environ):
        raise AssistantError("not_configured", "No AI assistant provider is configured.")

    assistant = AssistantService(environ={}, client_factory=factory)
    result = result_of(await run_task(assistant, "notes", notes_payload(units(3))))
    assert result["result"]["code"] == "not_configured"


async def test_cancellation_emits_a_cancelled_result() -> None:
    gate = asyncio.Event()
    client = FakeClient(streams=[["# never"]], block=gate)
    assistant, _seen = make_service(client)
    events: list[dict] = []
    task = asyncio.create_task(
        assistant.handle(
            {"request_id": "req-1", "task": "notes", "payload": notes_payload(units(3))},
            events.append,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if client.calls:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = result_of(events)
    assert result["ok"] is False and result["result"]["code"] == "cancelled"
    assert client.closed
    assert assistant.running == 0


async def test_concurrency_limit_answers_busy() -> None:
    assistant, _seen = make_service(FakeClient())
    assistant.running = assistant.max_concurrent
    result = result_of(await run_task(assistant, "probe", {"llm": LLM}))
    assert result["result"]["code"] == "busy"


# ------------------------------------------------------------------- title


async def test_title_uses_even_samples_and_sanitizes() -> None:
    client = FakeClient(completions=['**Title:** "Texture Perception and Pop-Out."'])
    assistant, _seen = make_service(client)
    transcript = units(600, words="a fairly long sentence about pre-attentive texture segmentation")
    payload = notes_payload(transcript, output_language="en")
    result = result_of(await run_task(assistant, "title", payload))
    assert result["result"] == {
        "title": "Texture Perception and Pop-Out",
        "provider": "dashscope",
        "model": "qwen-plus",
    }
    kind, messages, kwargs = client.calls[0]
    assert kind == "complete"
    assert "at most 8 words" in messages[0]["content"]
    user = messages[1]["content"]
    excerpts = user.split("<transcript_excerpts>\n", 1)[1].split("\n</transcript_excerpts>", 1)[0]
    assert len(excerpts) <= 6000
    assert "segmentation 0." in excerpts  # the start …
    assert "segmentation 525." in excerpts  # … the start of the last eighth
    assert "segmentation 300." in excerpts
    assert "Topic: texture perception" in user


async def test_empty_title_is_an_error() -> None:
    client = FakeClient(completions=['""'])
    assistant, _seen = make_service(client)
    result = result_of(await run_task(assistant, "title", notes_payload(units(5))))
    assert result["result"]["code"] == "empty_response"


@pytest.mark.parametrize(
    ("raw", "title"),
    [
        ("# 纹理感知与视觉搜索。", "纹理感知与视觉搜索"),
        ("标题：「预注意视觉」", "预注意视觉"),
        ("“Feature Integration Theory”", "Feature Integration Theory"),
        ("\n\n  - Textons and pop-out!  \nsecond line", "Textons and pop-out"),
        ("《红楼梦》导读", "《红楼梦》导读"),
        ("`Saccades`", "Saccades"),
        ("```\nVision\n```", "Vision"),
        ("", ""),
    ],
)
def test_sanitize_title(raw, title) -> None:
    assert output.sanitize_title(raw) == title


def test_sanitize_title_limits_length() -> None:
    long_title = "Texture " * 20
    cut = output.sanitize_title(long_title)
    assert len(cut) <= 60 and not cut.endswith(" ")
    assert len(output.sanitize_title("视" * 100)) == 60


def test_sample_excerpts_covers_the_session_within_budget() -> None:
    lines = [TranscriptLine(index * 1000, index * 1000 + 900, f"sentence number {index}") for index in range(2000)]
    sample = windows.sample_excerpts(lines, 6000)
    assert len(sample) <= 6000
    assert "sentence number 0" in sample
    assert "sentence number 1750" in sample or "sentence number 1751" in sample
    short = windows.sample_excerpts(lines[:3], 6000)
    assert short == "[00:00] sentence number 0\n[00:01] sentence number 1\n[00:02] sentence number 2"


# ------------------------------------------------------------------- probe


async def test_probe_needs_no_consent_and_sends_no_transcript() -> None:
    client = FakeClient(completions=["OK"])
    assistant, seen = make_service(client)
    result = result_of(await run_task(assistant, "probe", {"llm": LLM}))
    assert result["ok"] is True
    assert result["result"]["provider"] == "dashscope"
    assert result["result"]["model"] == "qwen-plus"
    assert isinstance(result["result"]["latency_ms"], int)
    assert client.calls[0][1] == [{"role": "user", "content": "Reply with OK"}]
    assert seen == [LLM]


# ----------------------------------------------------------------- context


def make_deck(path: Path) -> Path:
    a_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"

    def slide(*paragraphs: str) -> str:
        body = "".join(f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs)
        return f'<p:sld xmlns:a="{a_ns}" xmlns:p="p"><p:txBody>{body}</p:txBody></p:sld>'

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide("Texture Perception", "Béla Julesz and textons"))
        archive.writestr("ppt/slides/slide2.xml", slide("Outline", "Feature Integration Theory"))
        archive.writestr(
            "ppt/slides/slide3.xml",
            slide("Pre-attentive Vision", "Anne Treisman proposed Feature Integration Theory", "Julesz: textons"),
        )
        archive.writestr(
            "ppt/slides/slide4.xml",
            slide("Visual Search", "Anne Treisman and Béla Julesz disagreed", "Pop-out in V1 and V1 cells"),
        )
    return path


async def test_context_import_without_llm_is_local(tmp_path) -> None:
    deck = make_deck(tmp_path / "deck.pptx")
    client = FakeClient()
    assistant, seen = make_service(client)
    payload = {
        "attachments": [{"id": "d1", "path": str(deck), "name": "deck.pptx"}],
        "source_language": "en",
        "target_language": "zh",
        "use_llm": False,
    }
    events = await run_task(assistant, "context", payload)
    result = result_of(events)
    assert result["ok"] is True, result
    context = result["result"]["context"]
    assert context.startswith("Topic: Texture Perception; Pre-attentive Vision; Visual Search")
    assert "Outline" not in context
    terms_line = next(line for line in context.splitlines() if line.startswith("Terms: "))
    for term in ("Béla Julesz", "Anne Treisman", "Feature Integration Theory", "V1"):
        assert term in terms_line
    assert len(context) <= 2000
    assert "Béla Julesz" in result["result"]["terms"]
    assert result["result"]["glossary"] == []
    assert result["result"]["attachments"][0]["pages"] == 4
    assert result["result"]["warnings"] == []
    assert seen == [] and client.calls == []


async def test_context_import_with_llm_merges_term_pairs(tmp_path) -> None:
    deck = make_deck(tmp_path / "deck.pptx")
    client = FakeClient(
        completions=["Here you go:\n- saccade = 扫视\n2. **texton** = 纹理基元\nBéla Julesz = 贝拉·尤尔斯\nnot a pair"]
    )
    assistant, seen = make_service(client)
    payload = {
        "attachments": [{"id": "d1", "path": str(deck), "name": "deck.pptx"}],
        "source_language": "en",
        "target_language": "zh",
        "use_llm": True,
        "llm": LLM,
        "consent": CONSENT,
    }
    result = result_of(await run_task(assistant, "context", payload))["result"]
    assert {"source": "saccade", "target": "扫视"} in result["glossary"]
    assert {"source": "texton", "target": "纹理基元"} in result["glossary"]
    assert "saccade = 扫视" in result["context"]
    # A term with a pair is not repeated on the Terms line.
    terms_line = next(line for line in result["context"].splitlines() if line.startswith("Terms: "))
    assert "Béla Julesz" not in terms_line
    system = client.calls[0][1][0]["content"]
    assert "Simplified Chinese" in system
    assert seen == [LLM]


async def test_context_import_with_llm_requires_consent(tmp_path) -> None:
    deck = make_deck(tmp_path / "deck.pptx")
    assistant, seen = make_service(FakeClient())
    payload = {
        "attachments": [{"id": "d1", "path": str(deck), "name": "deck.pptx"}],
        "use_llm": True,
        "llm": LLM,
    }
    result = result_of(await run_task(assistant, "context", payload))
    assert result["result"]["code"] == "privacy_policy_denied"
    assert seen == []


async def test_context_import_keeps_local_result_when_the_model_fails(tmp_path) -> None:
    deck = make_deck(tmp_path / "deck.pptx")
    client = FakeClient(error=AuthenticationError("OpenAI rejected the API key (HTTP 401)."))
    assistant, _seen = make_service(client)
    payload = {
        "attachments": [{"id": "d1", "path": str(deck), "name": "deck.pptx"}],
        "target_language": "zh",
        "use_llm": True,
        "llm": LLM,
        "consent": CONSENT,
    }
    result = result_of(await run_task(assistant, "context", payload))
    assert result["ok"] is True
    assert result["result"]["context"].startswith("Topic: ")
    assert result["result"]["warnings"] == [
        "AI term extraction failed: OpenAI rejected the API key (HTTP 401)."
    ]


def test_composed_context_parses_as_session_context() -> None:
    context = terms.compose_context(
        ["Texture Perception", "Visual Search"],
        ["Béla Julesz", "V1", "pre-attentive", "bad, term", "a = b"],
        [("saccade", "扫视"), ("texton", "纹理基元")],
    )
    parsed = parse_session_context(context)
    assert "Texture Perception" in parsed.topic
    assert {"Béla Julesz", "V1", "pre-attentive", "saccade", "texton"} <= set(parsed.hint_terms)
    assert [(term.source, term.target) for term in parsed.glossary] == [
        ("saccade", "扫视"),
        ("texton", "纹理基元"),
    ]
    assert "bad, term" not in context


def test_compose_context_respects_the_limit() -> None:
    titles = [f"Slide title number {index}" for index in range(100)]
    many_terms = [f"Term{index}X" for index in range(500)]
    pairs = [(f"source{index}", f"目标{index}") for index in range(60)]
    context = terms.compose_context(titles, many_terms, pairs)
    assert len(context) <= 2000
    topic = context.splitlines()[0]
    assert len(topic) <= 400
    assert "source0 = 目标0" in context


def test_derive_terms_finds_names_acronyms_and_compounds() -> None:
    text = (
        "Pre-attentive Vision\n"
        "Béla Julesz proposed textons; Julesz argued pre-attentive vision is parallel.\n"
        "Feature Integration Theory (FIT) by Anne Treisman.\n"
        "Anne Treisman revised FIT. Feature Integration Theory explains pop-out.\n"
        "The CNN uses ImageNet. ImageNet and CNN again. Texture matters; texture is local.\n"
        "Béla Julesz, again.\n"
    )
    found = terms.derive_terms([text])
    for term in ("Béla Julesz", "Anne Treisman", "Feature Integration Theory", "FIT", "CNN", "ImageNet", "pre-attentive"):
        assert term in found, (term, found)
    assert "Texture" not in found  # ordinary word at a sentence start
    assert "Julesz" not in found  # part of the full name
    assert terms.derive_terms(["Single Mention only once"]) == []


def test_parse_term_lines_accepts_bullets_and_skips_noise() -> None:
    pairs = terms.parse_term_lines(
        "# Terms\n- saccade = 扫视\n1. **texton** = 纹理基元\nsaccade = 重复\nno pair here\nx =\n"
    )
    assert pairs == [("saccade", "扫视"), ("texton", "纹理基元")]


# ----------------------------------------------------------- windows/materials


def lines_every(step_ms: int, minutes: int, text: str = "word " * 6) -> list[TranscriptLine]:
    count = minutes * 60_000 // step_ms
    return [TranscriptLine(index * step_ms, index * step_ms + step_ms - 100, text.strip()) for index in range(count)]


@pytest.mark.parametrize(("minutes", "expected"), [(26, 2), (40, 3), (64, 5), (90, 7)])
def test_plan_windows_uses_twelve_to_fifteen_minute_windows(minutes, expected) -> None:
    lines = lines_every(10_000, minutes)
    plan = windows.plan_windows(lines)
    assert len(plan) == expected
    assert [line for window in plan for line in window] == lines  # unit boundaries, order kept
    for window in plan[:-1]:
        span = window[-1].end_ms - window[0].start_ms
        assert 11 * 60_000 <= span <= 15.5 * 60_000


def test_plan_windows_splits_dense_speech_by_characters() -> None:
    dense = lines_every(5_000, 20, text="x" * 400)  # ~96k chars in 20 minutes
    plan = windows.plan_windows(dense)
    assert len(plan) >= 4
    assert all(sum(len(line.text) + 9 for line in window) <= 30_000 for window in plan)


def test_plan_windows_without_timing_splits_by_characters() -> None:
    untimed = [TranscriptLine(0, 0, "y" * 1000) for _ in range(60)]
    plan = windows.plan_windows(untimed)
    assert len(plan) == 3
    assert sum(len(window) for window in plan) == 60


def test_short_final_window_joins_the_previous_one() -> None:
    lines = lines_every(10_000, 29)  # round(29 / 13.5) = 2 windows of 14.5 min
    assert len(windows.plan_windows(lines)) == 2
    tail = lines + [TranscriptLine(60 * 60_000, 60 * 60_000 + 1000, "late question")]
    plan = windows.plan_windows(tail)
    assert plan[-1][-1].text == "late question"


def test_single_call_threshold_uses_size_and_duration() -> None:
    assert windows.fits_single_call(lines_every(10_000, 20))
    assert not windows.fits_single_call(lines_every(10_000, 30))
    assert not windows.fits_single_call(lines_every(1_000, 10, text="z" * 50))


def attachment(name: str, pages: list[str]) -> ExtractedAttachment:
    return ExtractedAttachment(
        id=name,
        name=name,
        kind="pdf",
        pages=len(pages),
        segments=[AttachmentSegment(f"page {index + 1}", index + 1, text) for index, text in enumerate(pages)],
    )


def test_select_materials_prefers_relevant_pages_in_document_order() -> None:
    filler = ["General remarks about the course organisation and grading policy " * 8] * 10
    pages = filler[:3] + ["Saccadic eye movements and fixation duration " * 6] + filler[3:] + [
        "Textons: Julesz texture discrimination conjecture " * 6
    ]
    deck = attachment("deck.pdf", pages)
    materials = windows.select_materials(
        [deck],
        "[10:00] so saccadic movements move the eye between fixation points; julesz textons",
        1200,
        window_start=0.0,
        window_end=0.2,
    )
    assert len(materials) == 1
    text = materials[0].text
    assert "Saccadic eye movements" in text
    assert len(text) <= 1200
    if "Textons" in text:
        assert text.index("--- page 4 ---") < text.index("--- page 12 ---")


def test_budget_materials_shares_the_budget_and_flags_cuts() -> None:
    small = attachment("small.pdf", ["short page"])
    large = attachment("large.pdf", ["L" * 3000, "M" * 3000])
    materials = windows.budget_materials([small, large], 2000)
    assert materials[0].text == small.text
    assert len(materials[1].text) <= 2000 - len(small.text)
    assert large.truncated is True and small.truncated is False
    assert large.warning and large.warning.startswith("Only the first")


# ------------------------------------------------------------------- events


def test_task_events_coalesce_deltas_and_emit_one_result() -> None:
    now = [0.0]
    sent: list[dict] = []
    events = TaskEvents("r", sent.append, clock=lambda: now[0])
    events.delta("a")
    events.delta("b")
    assert sent == []
    now[0] = 0.1
    events.delta("c")
    assert [event["payload"]["text"] for event in sent] == ["abc"]
    events.delta("d")
    events.progress("section", {"index": 1})
    assert [event["type"] for event in sent][-2:] == ["assistant_delta", "assistant_progress"]
    assert [event["payload"]["seq"] for event in sent if event["type"] == "assistant_delta"] == [1, 2]
    events.delta("e")
    events.result(True, {"markdown": "x"})
    events.result(False, {"code": "late"})
    events.delta("ignored")
    assert sent[-2]["payload"] == {"request_id": "r", "seq": 3, "text": "e"}
    assert sent[-1]["payload"] == {"request_id": "r", "ok": True, "result": {"markdown": "x"}}


def test_preamble_gate_and_header_helpers() -> None:
    gate = output.PreambleGate(sections_only=True, limit=50)
    assert gate.feed("# Title\n\nOverview text.\n\n#") == ""
    assert gate.feed("# Section (00:00–01:00)\n") == "## Section (00:00–01:00)\n"
    assert gate.feed("- more") == "- more"
    fallback = output.PreambleGate(sections_only=False, limit=10)
    assert fallback.feed("```markdown\nplain text without headings") == "plain text without headings"
    head, rest = output.split_header("```markdown\n# T\n\nOverview.\n\n## Key terms\n- a\n```")
    assert head == "# T\n\nOverview." and rest == "## Key terms\n- a"
    assert output.drop_repeated_sections("## A (00:00–01:00)\nx\n\n## B\ny", ["A (05:00–06:00)"]) == "## B\ny"
    assert output.extract_title("intro\n# **Vision**  \n## x") == "Vision"


def test_preamble_gate_drops_the_fence_that_wrapped_the_answer() -> None:
    def run(chunks: list[str], *, sections_only: bool = False, limit: int = 800) -> str:
        gate = output.PreambleGate(sections_only=sections_only, limit=limit)
        return "".join(gate.feed(chunk) for chunk in chunks) + gate.finish()

    # Closing fence split across chunks, with blank lines around it.
    assert run(["```markdown\n# T\n\n- a\n", "\n`", "``", "\n\n"]).rstrip() == "# T\n\n- a"
    # An inner code block survives; only the outer closing fence goes.
    assert run(["```md\n## S\n```py\nx\n```\n", "```"], sections_only=True) == "## S\n```py\nx\n```\n"
    # Unwrapped answers pass through unchanged, including their own fences.
    assert run(["# T\n```\ncode\n```"]) == "# T\n```\ncode\n```"
    # No heading within the limit: the opening fence is stripped at once and
    # the closing one when it arrives.
    assert run(["```markdown\nplain text", " more\n```\n"], limit=5) == "plain text more\n"
    # A fence opened and closed inside the preamble is not a wrapper.
    assert run(["Example:\n```\nx\n```\n# T\n- a\n```"]) == "# T\n- a\n```"
    assert output.trim_preamble("Sure!\n```markdown\n# T\n\nOverview.\n```\n") == "# T\n\nOverview."
    assert output.trim_preamble("Overview only.\n\n## Key terms\n- a") == "Overview only.\n\n## Key terms\n- a"


async def test_logs_carry_no_transcript_model_text_or_key(caplog) -> None:
    """End to end over a mock HTTP transport (with a continuation and an
    authentication failure): INFO-and-above logs hold sizes, codes and the
    provider only."""
    import json as json_module
    import logging

    import httpx

    from echolingo.assistant.llm import ChatClient, resolve_endpoint

    secret = "sk-log-check-secret-0123456789"
    spoken = "zygomorphic flurbination"
    written = "quixotic-marmalade"
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        body = json_module.loads(request.content)
        if body.get("stream") is False:
            return httpx.Response(401, json={"error": {"message": f"bad key {secret}"}})
        finish = "length" if calls["count"] == 1 else "stop"
        chunk = {"choices": [{"delta": {"content": f"# {written}\n\n## A (00:00–01:00)\n- {written}\n"}, "finish_reason": finish}]}
        return httpx.Response(
            200,
            content=f"data: {json_module.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    environ = {"DASHSCOPE_API_KEY": secret}

    def factory(llm_field, env):
        return ChatClient(resolve_endpoint(llm_field, env), client=httpx.AsyncClient(transport=transport))

    assistant = AssistantService(environ=environ, client_factory=factory)
    caplog.set_level(logging.INFO)
    events = await run_task(assistant, "notes", notes_payload(units(6, words=spoken)))
    assert result_of(events)["ok"] is True
    assert calls["count"] == 2  # one continuation
    failed = result_of(await run_task(assistant, "title", notes_payload(units(6, words=spoken))))
    assert failed["result"]["code"] == "authentication_failed"
    assert secret not in json_module.dumps(failed)

    logged = "\n".join(record.getMessage() for record in caplog.records if record.levelno >= logging.INFO)
    assert "assistant notes" in logged and "continuing" in logged
    for forbidden in (secret, "zygomorphic", "flurbination", written, "texture perception"):
        assert forbidden not in logged
