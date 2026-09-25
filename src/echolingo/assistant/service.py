"""Assistant task orchestration: notes, titles, context terms and probes.

The sidecar command ``assistant_request {request_id, task, payload}``
runs :meth:`AssistantService.handle` as a background task.
Everything is validated inside that task, and every outcome, including
malformed input, missing credentials, provider errors and cancellation,
ends in exactly one ``assistant_result`` event. Nothing raises into the
sidecar's command loop.

Privacy: transcripts and attachment text are sent to the model only when the
task payload carries ``consent.transcript_upload_allowed == true``; the
probe sends one fixed prompt and the local context import sends nothing.
INFO logs carry sizes, counts, provider and model, never transcript,
attachment or model text.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..errors import BackendError
from . import prompts, terms as terms_module
from .attachments import ExtractedAttachment, extract_attachments, parse_attachment_items
from .llm import AssistantError, ChatClient, add_usage, build_client, strip_reasoning
from .output import (
    PreambleGate,
    clean_markdown,
    drop_repeated_sections,
    extract_title,
    sanitize_title,
    section_headings,
    split_header,
    trim_preamble,
)
from .prompts import MaterialText, SessionInfo, TranscriptLine
from .windows import (
    budget_materials,
    fits_single_call,
    plan_windows,
    sample_excerpts,
    select_materials,
    transcript_chars,
)

logger = logging.getLogger("echolingo.assistant")

TASKS = ("notes", "title", "context", "probe")
MAX_CONCURRENT_TASKS = 4

# notes
NOTES_MAX_TOKENS = 8192
NOTES_TEMPERATURE = 0.3
FINISHING_MAX_TOKENS = 2048
MATERIALS_BUDGET_SINGLE = 40_000
MATERIALS_BUDGET_WINDOW = 12_000
PREAMBLE_LIMIT_DOCUMENT = 800
PREAMBLE_LIMIT_SECTION = 1500
# title
TITLE_SAMPLE_CHARS = 6000
TITLE_MAX_TOKENS = 200
# One retry when a reasoning model used the whole budget before the title.
TITLE_RETRY_MAX_TOKENS = 2048
# The desktop shell stops a title job after 90 s. The retry gives up this long
# after the task started, so the reason reaches the user rather than a
# generic timeout; a slow reasoning model can take minutes for 2048 tokens.
TITLE_DEADLINE_S = 75.0
# context
CONTEXT_MATERIALS_BUDGET = 30_000
CONTEXT_MAX_TOKENS = 2048
# probe
PROBE_MAX_TOKENS = 16
# input limits (the shell validates too)
MAX_TRANSCRIPT_UNITS = 50_000
MAX_TRANSCRIPT_CHARS = 1_500_000
MAX_CONTEXT_CHARS = 4000
MAX_GLOSSARY_CHARS = 4000
REQUEST_ID_MAX_CHARS = 128

# delta coalescing: one event per ~80 ms or 1 KiB of text
DELTA_FLUSH_S = 0.08
DELTA_FLUSH_CHARS = 1024

_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8}){0,2}$")

PutEvent = Callable[[dict[str, Any]], Any]
ClientFactory = Callable[[Any, Mapping[str, str]], ChatClient]


def _invalid(message: str) -> AssistantError:
    return AssistantError("invalid_request", message)


# ------------------------------------------------------------------- events


class TaskEvents:
    """Progress, coalesced text deltas and the single result of one request."""

    def __init__(
        self,
        request_id: str,
        put: PutEvent,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.request_id = request_id
        self._put = put
        self._clock = clock
        self._seq = 0
        self._buffer: list[str] = []
        self._buffered = 0
        self._last_flush = clock()
        self.finished = False

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._put({"type": event_type, "payload": {"request_id": self.request_id, **payload}})

    def progress(self, stage: str, detail: Mapping[str, Any] | None = None) -> None:
        if self.finished:
            return
        self.flush()
        self._emit("assistant_progress", {"stage": stage, "detail": dict(detail or {})})

    def delta(self, text: str, *, immediate: bool = False) -> None:
        if not text or self.finished:
            return
        self._buffer.append(text)
        self._buffered += len(text)
        if (
            immediate
            or self._buffered >= DELTA_FLUSH_CHARS
            or self._clock() - self._last_flush >= DELTA_FLUSH_S
        ):
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        self._buffered = 0
        self._seq += 1
        self._last_flush = self._clock()
        self._emit("assistant_delta", {"seq": self._seq, "text": text})

    def result(self, ok: bool, result: Mapping[str, Any]) -> None:
        if self.finished:
            return
        if ok:
            self.flush()
        else:
            self._buffer.clear()
        self.finished = True
        self._emit("assistant_result", {"ok": ok, "result": dict(result)})


def error_result(error: BaseException) -> dict[str, str]:
    """``{code, message}`` for an ``ok: false`` result; never echoes secrets."""
    if isinstance(error, BackendError):
        return {"code": error.error_code, "message": str(error)}
    return {
        "code": "internal_error",
        "message": f"The assistant failed unexpectedly ({type(error).__name__}).",
    }


def request_id_of(command_payload: Any) -> str:
    if isinstance(command_payload, Mapping):
        value = command_payload.get("request_id")
        if isinstance(value, str) and 0 < len(value) <= REQUEST_ID_MAX_CHARS:
            return value
    return ""


# ------------------------------------------------------------------ parsing


def _language(value: Any, field: str, *, required: bool = False) -> str:
    if value is None or value == "":
        if required:
            raise _invalid(f"{field} is required.")
        return ""
    if not isinstance(value, str) or not (value == "auto" or _LANGUAGE_RE.match(value)):
        raise _invalid(f"{field} must be a language code such as 'zh' or 'en'.")
    return value


def _text(value: Any, field: str, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _invalid(f"{field} must be a string.")
    return value[:limit]


def _milliseconds(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"transcript {field} must be a number of milliseconds.")
    if not math.isfinite(value) or value < 0:
        raise _invalid(f"transcript {field} must be a non-negative number.")
    return int(value)


def parse_transcript(value: Any) -> list[TranscriptLine]:
    """Committed source units ``[{start_ms, end_ms, text}]`` in time order."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise _invalid("transcript must be a list of {start_ms, end_ms, text}.")
    if len(value) > MAX_TRANSCRIPT_UNITS:
        raise AssistantError("transcript_too_long", "The transcript is too long for the assistant.")
    lines: list[TranscriptLine] = []
    total = 0
    for item in value:
        if not isinstance(item, Mapping):
            raise _invalid("each transcript unit must be an object.")
        text = item.get("text")
        if not isinstance(text, str):
            raise _invalid("each transcript unit needs a text string.")
        text = " ".join(text.split())
        if not text:
            continue
        start = _milliseconds(item.get("start_ms"), "start_ms")
        end_value = item.get("end_ms")
        end = start if end_value is None else max(start, _milliseconds(end_value, "end_ms"))
        total += len(text)
        if total > MAX_TRANSCRIPT_CHARS:
            raise AssistantError(
                "transcript_too_long", "The transcript is too long for the assistant."
            )
        lines.append(TranscriptLine(start, end, text))
    lines.sort(key=lambda line: line.start_ms)
    return lines


def parse_session(payload: Mapping[str, Any]) -> SessionInfo:
    session = payload.get("session")
    if session is None:
        session = {}
    if not isinstance(session, Mapping):
        raise _invalid("session must be an object.")
    duration = session.get("duration_ms")
    duration_ms = (
        int(duration)
        if isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration > 0
        else 0
    )
    source = _language(session.get("source_language"), "session.source_language")
    glossary = payload.get("glossary")
    if glossary is None:
        glossary = session.get("glossary")
    return SessionInfo(
        source_language="" if source == "auto" else source,
        target_language=_language(session.get("target_language"), "session.target_language"),
        duration_ms=duration_ms,
        context=_text(session.get("context"), "session.context", MAX_CONTEXT_CHARS),
        glossary=_text(glossary, "glossary", MAX_GLOSSARY_CHARS),
    )


def output_language(payload: Mapping[str, Any], session: SessionInfo | None = None) -> str:
    value = _language(payload.get("output_language"), "output_language")
    if value and value != "auto":
        return value
    if session is not None and session.target_language and session.target_language != "auto":
        return session.target_language
    return "en"


def require_consent(payload: Mapping[str, Any]) -> None:
    consent = payload.get("consent")
    if not (isinstance(consent, Mapping) and consent.get("transcript_upload_allowed") is True):
        raise AssistantError(
            "privacy_policy_denied",
            "Sending transcripts and attachments to the AI assistant is not allowed. "
            "Turn on the consent in Settings → AI assistant first.",
        )


def _attachment_items(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    try:
        return parse_attachment_items(payload.get("attachments"))
    except ValueError as error:
        raise _invalid(str(error)) from None


def _usage_result(usage: Mapping[str, int]) -> dict[str, int | None]:
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _display_name(client: Any) -> str:
    endpoint = getattr(client, "endpoint", None)
    return getattr(endpoint, "display_name", None) or getattr(client, "provider", "The model")


def _reasoning_title_error(client: Any) -> AssistantError:
    return AssistantError(
        "empty_response",
        f"{_display_name(client)} spent its reply budget on reasoning and returned no "
        "title. Try again, or choose a model without reasoning in Settings → AI assistant.",
    )


# ------------------------------------------------------------------ service


class AssistantService:
    """Runs assistant tasks against the configured OpenAI-compatible model."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
        max_concurrent: int = MAX_CONCURRENT_TASKS,
    ) -> None:
        self._environ = environ
        self._client_factory = client_factory or build_client
        self.max_concurrent = max_concurrent
        self.running = 0

    @property
    def environ(self) -> Mapping[str, str]:
        # The shell fills keys and provider settings into the sidecar's
        # environment from the OS secure store.
        return self._environ if self._environ is not None else os.environ

    def _client(self, payload: Mapping[str, Any]) -> ChatClient:
        return self._client_factory(payload.get("llm"), self.environ)

    async def handle(self, command_payload: Any, put: PutEvent) -> None:
        """Run one ``assistant_request`` and emit its events; never raises
        except to propagate cancellation after the result was emitted."""
        request_id = request_id_of(command_payload)
        events = TaskEvents(request_id, put)
        if not request_id:
            events.result(False, error_result(_invalid("assistant_request needs a request_id.")))
            return
        if self.running >= self.max_concurrent:
            events.result(
                False,
                error_result(
                    AssistantError(
                        "busy", "Other assistant tasks are still running; try again shortly."
                    )
                ),
            )
            return
        self.running += 1
        started = time.monotonic()
        task = "?"
        try:
            task = command_payload.get("task")
            if not isinstance(task, str) or task not in TASKS:
                task = "?"
                raise _invalid("Unknown assistant task.")
            payload = command_payload.get("payload")
            if payload is None:
                payload = {}
            if not isinstance(payload, Mapping):
                raise _invalid("The assistant task payload must be an object.")
            logger.info("assistant %s started (request %s)", task, request_id)
            result = await self.run(task, payload, events)
        except asyncio.CancelledError:
            logger.info("assistant %s cancelled (request %s)", task, request_id)
            events.result(
                False, {"code": "cancelled", "message": "The assistant task was cancelled."}
            )
            raise
        except BackendError as error:
            logger.info(
                "assistant %s failed (request %s): %s", task, request_id, error.error_code
            )
            events.result(False, error_result(error))
        except Exception as error:  # defensive: parser/provider surprises
            logger.warning(
                "assistant %s failed unexpectedly (request %s): %s",
                task,
                request_id,
                type(error).__name__,
            )
            logger.debug("assistant failure details", exc_info=True)
            events.result(False, error_result(error))
        else:
            logger.info(
                "assistant %s finished (request %s) in %.1f s",
                task,
                request_id,
                time.monotonic() - started,
            )
            events.result(True, result)
        finally:
            self.running -= 1

    async def run(
        self, task: str, payload: Mapping[str, Any], events: TaskEvents
    ) -> dict[str, Any]:
        if task == "probe":
            return await self.probe(payload)
        if task == "context":
            return await self.context(payload, events)
        require_consent(payload)
        if task == "title":
            return await self.title(payload, events)
        if task == "notes":
            return await self.notes(payload, events)
        raise _invalid("Unknown assistant task.")

    # ------------------------------------------------------------------ probe

    async def probe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        async with self._client(payload) as client:
            started = time.monotonic_ns()
            await client.complete(
                [dict(message) for message in prompts.PROBE_MESSAGES],
                max_tokens=PROBE_MAX_TOKENS,
                temperature=0.0,
            )
            latency_ms = int(round((time.monotonic_ns() - started) / 1_000_000))
            return {"provider": client.provider, "model": client.model, "latency_ms": latency_ms}

    # ------------------------------------------------------------------ title

    async def title(self, payload: Mapping[str, Any], events: TaskEvents) -> dict[str, Any]:
        session = parse_session(payload)
        language = output_language(payload, session)
        lines = parse_transcript(payload.get("transcript"))
        if not lines:
            raise AssistantError("empty_transcript", "The session has no transcript yet.")
        excerpts = sample_excerpts(lines, TITLE_SAMPLE_CHARS)
        messages = _messages(
            prompts.title_system_prompt(language),
            prompts.title_user_message(excerpts, session.context, session.glossary),
        )
        deadline = asyncio.get_running_loop().time() + TITLE_DEADLINE_S
        async with self._client(payload) as client:
            events.progress("writing")
            answer = await client.complete(
                messages, max_tokens=TITLE_MAX_TOKENS, temperature=0.3
            )
            title = sanitize_title(answer.text)
            if not title and answer.finish_reason == "length":
                # A reasoning model can stop at the output limit before it
                # writes anything.
                logger.info("assistant title: no text before the output limit; retrying once")
                try:
                    async with asyncio.timeout_at(deadline):
                        answer = await client.complete(
                            messages, max_tokens=TITLE_RETRY_MAX_TOKENS, temperature=0.3
                        )
                except TimeoutError:
                    logger.info("assistant title: the retry ran out of time")
                    raise _reasoning_title_error(client) from None
                title = sanitize_title(answer.text)
            if not title:
                if answer.finish_reason == "length":
                    raise _reasoning_title_error(client)
                raise AssistantError(
                    "empty_response", f"{_display_name(client)} returned no title."
                )
            return {"title": title, "provider": client.provider, "model": client.model}

    # ------------------------------------------------------------------ notes

    async def notes(self, payload: Mapping[str, Any], events: TaskEvents) -> dict[str, Any]:
        session = parse_session(payload)
        language = output_language(payload, session)
        lines = parse_transcript(payload.get("transcript"))
        if not lines:
            raise AssistantError("empty_transcript", "The session has no transcript yet.")
        items = _attachment_items(payload)
        # Resolve the provider before reading files so a missing key fails fast.
        async with self._client(payload) as client:
            extracted: list[ExtractedAttachment] = []
            if items:
                events.progress("extracting", {"total": len(items)})
                extracted = await extract_attachments(items)
            usage: dict[str, int] = {}
            single = fits_single_call(lines)
            logger.info(
                "assistant notes: %d units, %d chars, %d attachment(s), %s",
                len(lines),
                transcript_chars(lines),
                len(extracted),
                "one call" if single else "windowed",
            )
            if single:
                markdown = await self._notes_single(
                    client, session, language, lines, extracted, events, usage
                )
            else:
                markdown = await self._notes_windowed(
                    client, session, language, lines, extracted, events, usage
                )
            markdown = clean_markdown(markdown)
            if not markdown:
                raise AssistantError(
                    "empty_response", f"{_display_name(client)} returned no notes."
                )
            return {
                "markdown": markdown,
                "title": extract_title(markdown),
                "provider": client.provider,
                "model": client.model,
                "usage": _usage_result(usage),
                "attachments": [item.report() for item in extracted],
                "prompt_version": prompts.PROMPT_VERSION,
                "source_chars": transcript_chars(lines),
            }

    async def _stream(
        self,
        client: ChatClient,
        messages: list[dict[str, str]],
        gate: PreambleGate,
        emit: Callable[[str], None],
        usage: dict[str, int],
        on_first_text: Callable[[], None] | None = None,
    ) -> str:
        parts: list[str] = []
        first = True
        async for delta in client.stream_text(
            messages, max_tokens=NOTES_MAX_TOKENS, temperature=NOTES_TEMPERATURE
        ):
            if delta.text:
                text = gate.feed(delta.text)
                if text:
                    if first and on_first_text is not None:
                        on_first_text()
                    first = False
                    parts.append(text)
                    emit(text)
            else:
                add_usage(usage, delta.usage)
        tail = gate.finish()
        if tail:
            parts.append(tail)
            emit(tail)
        return "".join(parts)

    async def _notes_single(
        self,
        client: ChatClient,
        session: SessionInfo,
        language: str,
        lines: Sequence[TranscriptLine],
        extracted: Sequence[ExtractedAttachment],
        events: TaskEvents,
        usage: dict[str, int],
    ) -> str:
        materials = budget_materials(extracted, MATERIALS_BUDGET_SINGLE)
        messages = _messages(
            prompts.notes_system_prompt(language),
            prompts.notes_user_message(
                session, prompts.format_transcript(lines), materials, language
            ),
        )
        events.progress("reading")
        text = await self._stream(
            client,
            messages,
            PreambleGate(sections_only=False, limit=PREAMBLE_LIMIT_DOCUMENT),
            events.delta,
            usage,
            on_first_text=lambda: events.progress("writing"),
        )
        events.flush()
        return text

    async def _notes_windowed(
        self,
        client: ChatClient,
        session: SessionInfo,
        language: str,
        lines: Sequence[TranscriptLine],
        extracted: Sequence[ExtractedAttachment],
        events: TaskEvents,
        usage: dict[str, int],
    ) -> str:
        windows = plan_windows(lines)
        total = len(windows)
        logger.info("assistant notes: %d windows", total)
        joiner = _PartJoiner(events.delta)
        headings: list[str] = []
        system = prompts.notes_system_prompt(language)
        for index, window in enumerate(windows, 1):
            start_ms = window[0].start_ms
            end_ms = max(line.end_ms for line in window)
            events.progress(
                "section",
                {"index": index, "total": total, "start_ms": start_ms, "end_ms": end_ms},
            )
            transcript = prompts.format_transcript(window)
            materials: list[MaterialText] = select_materials(
                extracted,
                transcript,
                MATERIALS_BUDGET_WINDOW,
                window_start=(index - 1) / total,
                window_end=index / total,
            )
            messages = _messages(
                system,
                prompts.notes_window_message(
                    session,
                    transcript,
                    materials,
                    language,
                    index=index,
                    total=total,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    headings_so_far=headings,
                ),
            )
            joiner.start_part()
            section = await self._stream(
                client,
                messages,
                PreambleGate(sections_only=True, limit=PREAMBLE_LIMIT_SECTION),
                joiner,
                usage,
            )
            headings.extend(section_headings(section))
        events.flush()
        sections = joiner.text
        if not sections.strip():
            raise AssistantError("empty_response", f"{_display_name(client)} returned no notes.")

        # Title, overview (placed at the top) and closing sections (appended).
        events.progress("finishing")
        try:
            finishing = await client.complete_streamed(
                _messages(
                    prompts.finishing_system_prompt(language),
                    prompts.finishing_user_message(session, clean_markdown(sections)),
                ),
                max_tokens=FINISHING_MAX_TOKENS,
                temperature=NOTES_TEMPERATURE,
                max_continuations=1,
            )
        except BackendError as error:
            # Every section is already written; a failed summary call (rate
            # limit, timeout) must not throw them away. The notes are saved
            # without the title and overview.
            logger.info(
                "assistant notes: finishing call failed (%s); keeping the sections",
                error.error_code,
            )
            return sections
        add_usage(usage, finishing.usage)
        head, rest = split_header(trim_preamble(finishing.text))
        closing = drop_repeated_sections(rest, headings)
        # Deltas are append-only: the header, which belongs at
        # the top of the document, reaches the shell only through the result.
        header = f"{head}\n\n" if head else ""
        tail = ""
        if closing:
            tail = _separator(sections) + closing + "\n"
            events.delta(tail)
        events.flush()
        return header + sections + tail

    # ---------------------------------------------------------------- context

    async def context(self, payload: Mapping[str, Any], events: TaskEvents) -> dict[str, Any]:
        use_llm = payload.get("use_llm") is True
        if use_llm:
            require_consent(payload)
        items = _attachment_items(payload)
        if not items:
            raise _invalid("Choose at least one file to import.")
        # "auto" (detect the spoken language) names no language for the prompt.
        source_language = _language(payload.get("source_language"), "source_language")
        source_language = "" if source_language == "auto" else source_language
        target_language = _language(payload.get("target_language"), "target_language")
        target_language = "" if target_language == "auto" else target_language
        events.progress("extracting", {"total": len(items)})
        extracted = await extract_attachments(items)
        warnings = [f"{item.name}: {item.warning}" for item in extracted if item.warning]
        titles = terms_module.derive_titles(extracted)
        # Up to 300k characters of regex work: off the event loop, which also
        # serves a live session's audio.
        heuristic_terms = await asyncio.to_thread(
            terms_module.derive_terms,
            [segment.text for item in extracted for segment in item.segments],
        )
        pairs: list[tuple[str, str]] = []
        if use_llm and any(item.text.strip() for item in extracted):
            events.progress("reading")
            try:
                pairs = await self._llm_terms(
                    payload, extracted, source_language, target_language or "en"
                )
            except BackendError as error:
                # The local import still succeeds; the user sees why the AI
                # part is missing.
                warnings.append(f"AI term extraction failed: {error}")
        context = terms_module.compose_context(titles, heuristic_terms, pairs)
        from ..session_context import parse_session_context

        parsed = parse_session_context(context)
        return {
            "context": context,
            "terms": list(parsed.hint_terms),
            "glossary": [{"source": term.source, "target": term.target} for term in parsed.glossary],
            "attachments": [item.report() for item in extracted],
            "warnings": warnings,
        }

    async def _llm_terms(
        self,
        payload: Mapping[str, Any],
        extracted: Sequence[ExtractedAttachment],
        source_language: str,
        target_language: str,
    ) -> list[tuple[str, str]]:
        materials = budget_materials(
            [_copy_for_prompt(item) for item in extracted], CONTEXT_MATERIALS_BUDGET
        )
        async with self._client(payload) as client:
            # A short list: reasoning would only spend the budget.
            answer = await client.complete_streamed(
                _messages(
                    prompts.terms_system_prompt(target_language),
                    prompts.terms_user_message(materials, source_language),
                ),
                max_tokens=CONTEXT_MAX_TOKENS,
                temperature=0.2,
                max_continuations=0,
                disable_thinking=True,
            )
        return terms_module.parse_term_lines(strip_reasoning(answer.text))


def _separator(written: str) -> str:
    """What keeps exactly one blank line after ``written``."""
    if not written.strip() or written.endswith("\n\n"):
        return ""
    return "\n" if written.endswith("\n") else "\n\n"


class _PartJoiner:
    """Appends the parts of a long document, one blank line between parts."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._parts: list[str] = []
        self._part_started = False

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def start_part(self) -> None:
        self._part_started = False

    def __call__(self, text: str) -> None:
        if not text:
            return
        if not self._part_started:
            self._part_started = True
            separator = _separator(self.text)
            if separator:
                self._parts.append(separator)
                self._emit(separator)
        self._parts.append(text)
        self._emit(text)


def _copy_for_prompt(item: ExtractedAttachment) -> ExtractedAttachment:
    """A copy whose truncation flags the prompt budget may change freely
    (the reports returned to the shell describe the extraction only)."""
    return ExtractedAttachment(
        id=item.id,
        name=item.name,
        kind=item.kind,
        pages=item.pages,
        truncated=item.truncated,
        warning=item.warning,
        segments=list(item.segments),
    )
