"""Streaming OpenAI-compatible chat client for the AI assistant.

One client serves every assistant preset in ``backends/registry.py``
(DashScope compatible mode, OpenAI, DeepSeek, Gemini's OpenAI layer, Groq,
OpenRouter, SiliconFlow and self-hosted servers). It is deliberately separate
from the translation adapters: notes need long outputs, continuation after
``finish_reason == "length"`` and minute-scale read timeouts, while live
translation needs second-scale deadlines.

Error texts carry the provider name, HTTP status and (for DashScope) the
region/host that was tried. They never include the API key, request bodies or
provider response bodies.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..backends import dashscope as dashscope_module
from ..backends import registry
from ..errors import AuthenticationError, BackendError, RateLimitError

logger = logging.getLogger("echolingo.assistant.llm")

CONNECT_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 120.0
WRITE_TIMEOUT_S = 60.0
MAX_CONTINUATIONS = 2
CONTINUE_PROMPT = "Continue exactly where you stopped, without repeating."
# Some endpoints (SiliconFlow, older vLLM) cap max_tokens at 4096 and answer
# HTTP 400 above it; the request is retried with this value.
FALLBACK_MAX_TOKENS = 4096
# One request plus retries that each drop or rename one rejected field.
MAX_REQUEST_ATTEMPTS = 4

# OpenRouter attribution headers (https://openrouter.ai/docs/api-reference/overview).
OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/echolingo",
    "X-Title": "EchoLingo",
}

_CONTEXT_WORDING = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "too many tokens",
    "too long",
    "reduce the length",
    "input length",
    "prompt is too long",
    "range of input length",
    "exceeds the model",
    "token limit",
)
_UNSUPPORTED_WORDING = ("unsupported", "not supported", "not support", "only the default")
_CONTENT_FILTER_WORDING = ("data_inspection_failed", "content_filter", "inappropriate content")

# The request field that switches a provider's reasoning ("thinking") phase
# off. Qwen3-family models on DashScope otherwise spend the budget on hidden
# reasoning (and refuse non-streaming calls with thinking enabled); DeepSeek's
# current models think by default and can spend a short budget before the
# answer starts.
_THINKING_OFF: dict[str, tuple[str, Any]] = {
    "dashscope": ("enable_thinking", False),
    "deepseek": ("thinking", {"type": "disabled"}),
}
# Providers whose every call turns thinking off; the others only for the
# short tasks that ask for it (titles, the probe, term lists).
_THINKING_OFF_ALWAYS = frozenset({"dashscope"})

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class AssistantError(BackendError):
    """An assistant failure with a stable machine-readable ``code``."""

    recoverable = True

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class LlmEndpoint:
    provider: str
    display_name: str
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    key_required: bool = True
    stream_usage: bool = False
    dashscope_endpoint: dashscope_module.DashScopeEndpoint | None = None

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or ""


@dataclass(slots=True)
class ChatDelta:
    """A streamed text delta; the last item of a stream carries the totals."""

    text: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    continuations: int = 0


@dataclass(slots=True)
class ChatResult:
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    continuations: int = 0


def is_loopback_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").strip("[]").lower()
    if host in {"localhost", "0.0.0.0"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_endpoint(llm: Any, environ: Mapping[str, str]) -> LlmEndpoint:
    """Validate the ``llm`` task field against the registry presets."""
    if not isinstance(llm, Mapping):
        raise AssistantError("not_configured", "No AI assistant provider is configured.")
    group = llm.get("group")
    model = llm.get("model")
    if group is None or group == "":
        raise AssistantError("not_configured", "No AI assistant provider is configured.")
    if not isinstance(group, str) or group not in registry.ASSISTANT_PRESETS:
        raise AssistantError("invalid_request", "Unknown AI assistant provider.")
    if model is not None and not isinstance(model, str):
        raise AssistantError("invalid_request", "The assistant model must be a string.")
    values = registry.assistant_preset_values(group, model, environ)
    display = values["display_name"]
    if not values["base_url"]:
        raise AssistantError(
            "not_configured", f"Set the base URL for the {display} in Settings."
        )
    if urlsplit(values["base_url"]).scheme not in {"http", "https"}:
        raise AssistantError("not_configured", f"The {display} base URL must start with http(s)://.")
    if not values["model"]:
        raise AssistantError("not_configured", f"Choose a model for {display} in Settings.")
    if values["key_required"] and not values["api_key"]:
        raise AssistantError(
            "not_configured", f"{display} has no API key. Add it in Settings → Cloud providers."
        )
    return LlmEndpoint(
        provider=group,
        display_name=display,
        base_url=values["base_url"].rstrip("/"),
        model=values["model"],
        api_key=values["api_key"] or None,
        key_required=values["key_required"],
        stream_usage=values["stream_usage"],
        dashscope_endpoint=values["dashscope_endpoint"],
    )


def _usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens"):
        number = value.get(key)
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            result[key] = int(number)
    return result or None


def add_usage(total: dict[str, int], usage: Mapping[str, int] | None) -> None:
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens"):
        if key in usage:
            total[key] = total.get(key, 0) + int(usage[key])


def strip_reasoning(text: str) -> str:
    """Drop the ``<think>…</think>`` reasoning that self-hosted reasoning
    models (Ollama and similar) put in the answer text.

    A reply cut off inside a leading, unterminated block has no answer and
    becomes empty; a stray closing tag (chat templates that open the block in
    the prompt) drops everything before it.
    """
    lowered = text.lower()
    if _THINK_OPEN not in lowered and _THINK_CLOSE not in lowered:
        return text
    text = _THINK_BLOCK.sub("", text)
    lowered = text.lower()
    close = lowered.rfind(_THINK_CLOSE)
    if close >= 0:
        text = text[close + len(_THINK_CLOSE) :]
    elif lowered.lstrip().startswith(_THINK_OPEN):
        return ""
    return text.lstrip()


def trim_overlap(previous: str, head: str, *, min_overlap: int = 16, window: int = 400) -> str:
    """Drop a repeated prefix of a continuation that restates the previous tail."""
    tail = previous[-window:]
    for size in range(min(len(tail), len(head)), min_overlap - 1, -1):
        if tail.endswith(head[:size]):
            return head[size:]
    return head


class ChatClient:
    """Chat-completions client for one resolved endpoint."""

    def __init__(
        self,
        endpoint: LlmEndpoint,
        *,
        client: httpx.AsyncClient | None = None,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
        read_timeout_s: float = READ_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint
        self.read_timeout_s = read_timeout_s
        self._owns_client = client is None
        # httpx honours HTTPS_PROXY / NO_PROXY; a server on this machine never
        # goes through the proxy.
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=read_timeout_s,
                write=WRITE_TIMEOUT_S,
                pool=connect_timeout_s,
            ),
            trust_env=not is_loopback_url(endpoint.base_url),
        )
        # Optional request fields; each is dropped for the rest of this
        # client's life once the endpoint rejects it with HTTP 400/422.
        self._stream_usage = endpoint.stream_usage
        self._thinking_off = _THINKING_OFF.get(endpoint.provider)
        self._thinking_off_always = endpoint.provider in _THINKING_OFF_ALWAYS
        self._send_temperature = True
        # OpenAI reasoning models accept only ``max_completion_tokens``.
        self._max_tokens_field = "max_tokens"

    @property
    def provider(self) -> str:
        return self.endpoint.provider

    @property
    def model(self) -> str:
        return self.endpoint.model

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- request

    def chat_url(self) -> str:
        return f"{self.endpoint.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        if self.endpoint.provider == "openrouter":
            headers.update(OPENROUTER_HEADERS)
        return headers

    def payload(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        disable_thinking: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            self._max_tokens_field: int(max_tokens),
        }
        if self._send_temperature:
            body["temperature"] = float(temperature)
        if stream and self._stream_usage:
            body["stream_options"] = {"include_usage": True}
        if self._thinking_off is not None and (disable_thinking or self._thinking_off_always):
            name, value = self._thinking_off
            body[name] = value
        return body

    # ----------------------------------------------------------------- errors

    def _auth_message(self, status: int) -> str:
        endpoint = self.endpoint
        if endpoint.dashscope_endpoint is not None:
            if status == 403:
                return dashscope_module.forbidden_message(
                    endpoint.display_name, endpoint.dashscope_endpoint
                )
            return dashscope_module.unauthorized_message(
                endpoint.display_name, endpoint.dashscope_endpoint
            )
        return (
            f"{endpoint.display_name} rejected the API key (HTTP {status}). "
            f"Verify that the key is active and that model {endpoint.model!r} is enabled for it."
        )

    def _status_error(self, status: int, body: str, model: str) -> BackendError:
        display = self.endpoint.display_name
        lowered = body.lower()
        if status in (401, 403):
            return AuthenticationError(self._auth_message(status))
        if status == 400 and "api key" in lowered and ("valid" in lowered or "invalid" in lowered):
            # Gemini's OpenAI endpoint answers 400 (not 401) for a bad key.
            return AuthenticationError(self._auth_message(400))
        if status == 429:
            return RateLimitError(
                f"{display} rate limit or quota exceeded (HTTP 429). Wait a moment and retry, "
                "or check the account's quota."
            )
        if status in (400, 413, 422) and any(word in lowered for word in _CONTEXT_WORDING):
            return AssistantError(
                "context_too_long",
                f"{display} rejected the request as too long for model {model!r} "
                f"(HTTP {status}). Choose a model with a longer context window or fewer "
                "attachments.",
            )
        if status == 400 and any(word in lowered for word in _CONTENT_FILTER_WORDING):
            return AssistantError(
                "content_filtered",
                f"{display} declined the request because of its content policy (HTTP 400).",
            )
        if status == 404:
            return AssistantError(
                "provider_error",
                f"{display} returned HTTP 404; check that model {model!r} exists for this "
                "endpoint.",
            )
        return AssistantError("provider_error", f"{display} returned HTTP {status}.")

    def _transport_error(self, error: httpx.HTTPError) -> AssistantError:
        display = self.endpoint.display_name
        if isinstance(error, httpx.TimeoutException) and not isinstance(error, httpx.ConnectTimeout):
            return AssistantError(
                "provider_timeout",
                f"{display} stopped responding (no data for {self.read_timeout_s:.0f} s).",
            )
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
            return AssistantError(
                "network_error",
                f"Cannot reach {display} ({self.endpoint.host}). Check the network "
                "connection or proxy settings.",
            )
        return AssistantError(
            "network_error", f"The connection to {display} failed ({type(error).__name__})."
        )

    @staticmethod
    async def _read_error_body(response: httpx.Response) -> str:
        # Classification only; never surfaced or logged.
        try:
            return (await response.aread()).decode("utf-8", "replace")[:4000]
        except Exception:  # pragma: no cover - diagnostics only
            return ""

    def _adjust_for_rejection(
        self, status: int, body: str, max_tokens: int, sent: Mapping[str, Any]
    ) -> int | None:
        """Relax the request after an HTTP 400/422 that names an optional field.

        ``sent`` is the rejected request body. Returns the ``max_tokens`` to
        retry with, or ``None`` when the error is not about a field this client
        can drop.
        """
        if status not in (400, 422):
            return None
        lowered = body.lower()
        unsupported = any(word in lowered for word in _UNSUPPORTED_WORDING)
        if "stream_options" in sent and "stream_options" in lowered:
            self._stream_usage = False
            return max_tokens
        thinking = self._thinking_off
        if thinking is not None and thinking[0] in sent and thinking[0] in lowered:
            self._thinking_off = None
            return max_tokens
        if (
            self._max_tokens_field == "max_tokens"
            and "max_completion_tokens" in lowered
            and unsupported
        ):
            self._max_tokens_field = "max_completion_tokens"
            return max_tokens
        if self._send_temperature and "temperature" in lowered and unsupported:
            self._send_temperature = False
            return max_tokens
        if max_tokens > FALLBACK_MAX_TOKENS and (
            "max_tokens" in lowered or "max_completion_tokens" in lowered
        ):
            # Some endpoints (SiliconFlow, older vLLM) cap the output size.
            return FALLBACK_MAX_TOKENS
        return None

    # -------------------------------------------------------------- streaming

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        disable_thinking: bool = False,
    ) -> AsyncIterator[ChatDelta]:
        """Stream one completion.

        Yields text deltas; the final item has empty text and carries
        ``finish_reason`` and ``usage``. Cancelling the consuming task closes
        the HTTP response. ``disable_thinking`` turns the model's reasoning
        phase off where the provider has a field for it.
        """
        model = model or self.endpoint.model
        finish_reason: str | None = None
        usage: dict[str, int] | None = None
        attempt_max_tokens = int(max_tokens)
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            request = self.payload(
                messages,
                model=model,
                max_tokens=attempt_max_tokens,
                temperature=temperature,
                stream=True,
                disable_thinking=disable_thinking,
            )
            try:
                async with self.client.stream(
                    "POST", self.chat_url(), headers=self.headers(), json=request
                ) as response:
                    if response.status_code >= 400:
                        body = await self._read_error_body(response)
                        retry = (
                            self._adjust_for_rejection(
                                response.status_code, body, attempt_max_tokens, request
                            )
                            if attempt + 1 < MAX_REQUEST_ATTEMPTS
                            else None
                        )
                        if retry is not None:
                            attempt_max_tokens = retry
                            continue
                        raise self._status_error(response.status_code, body, model)
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        if chunk.get("error"):
                            raise AssistantError(
                                "provider_error",
                                f"{self.endpoint.display_name} reported an error while "
                                "generating.",
                            )
                        chunk_usage = _usage(chunk.get("usage"))
                        if chunk_usage:
                            usage = chunk_usage
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            continue
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        delta = choice.get("delta")
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            yield ChatDelta(text=content)
                break
            except httpx.HTTPError as error:
                raise self._transport_error(error) from None
        yield ChatDelta(finish_reason=finish_reason, usage=usage)

    async def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        max_continuations: int = MAX_CONTINUATIONS,
        disable_thinking: bool = False,
    ) -> AsyncIterator[ChatDelta]:
        """Stream a completion, continuing up to ``max_continuations`` times
        when the model stops at its output limit.

        The final item carries the last finish reason, the usage summed over
        every call and the number of continuation calls made.
        """
        history = list(messages)
        total: dict[str, int] = {}
        have_usage = False
        text_so_far = ""
        finish: str | None = None
        continuations = 0
        for attempt in range(max_continuations + 1):
            segment: list[str] = []
            pending = ""
            dedupe = attempt > 0
            finish = None
            async for delta in self.stream_chat(
                history,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                disable_thinking=disable_thinking,
            ):
                if delta.text:
                    if dedupe:
                        # Hold back the head of a continuation until it can be
                        # compared with the tail of what was already written.
                        pending += delta.text
                        if len(pending) < 400:
                            continue
                        text = trim_overlap(text_so_far, pending)
                        dedupe = False
                        pending = ""
                    else:
                        text = delta.text
                    if text:
                        segment.append(text)
                        yield ChatDelta(text=text)
                    continue
                if delta.finish_reason:
                    finish = delta.finish_reason
                if delta.usage:
                    have_usage = True
                    add_usage(total, delta.usage)
            if pending:
                text = trim_overlap(text_so_far, pending)
                if text:
                    segment.append(text)
                    yield ChatDelta(text=text)
            written = "".join(segment)
            text_so_far += written
            if finish != "length" or attempt == max_continuations or not written.strip():
                break
            continuations += 1
            logger.info(
                "%s stopped at the output limit; continuing (%d/%d)",
                self.endpoint.display_name,
                continuations,
                max_continuations,
            )
            history = history + [
                {"role": "assistant", "content": written},
                {"role": "user", "content": CONTINUE_PROMPT},
            ]
        yield ChatDelta(
            finish_reason=finish,
            usage=total if have_usage else None,
            continuations=continuations,
        )

    async def complete_streamed(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        max_continuations: int = MAX_CONTINUATIONS,
        disable_thinking: bool = False,
    ) -> ChatResult:
        parts: list[str] = []
        final = ChatDelta()
        async for delta in self.stream_text(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_continuations=max_continuations,
            disable_thinking=disable_thinking,
        ):
            if delta.text:
                parts.append(delta.text)
            else:
                final = delta
        return ChatResult(
            text="".join(parts),
            finish_reason=final.finish_reason,
            usage=final.usage or {},
            continuations=final.continuations,
        )

    # ------------------------------------------------------------- one-shot

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> ChatResult:
        """One non-streaming completion for a short task (a title, the probe).

        The model's reasoning phase is turned off where the provider allows
        it, and ``<think>`` blocks are removed from the text; a reasoning
        model that used the whole budget returns empty text with
        ``finish_reason == "length"``.
        """
        model = model or self.endpoint.model
        attempt_max_tokens = int(max_tokens)
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            request = self.payload(
                messages,
                model=model,
                max_tokens=attempt_max_tokens,
                temperature=temperature,
                stream=False,
                disable_thinking=True,
            )
            try:
                response = await self.client.post(
                    self.chat_url(), headers=self.headers(), json=request
                )
            except httpx.HTTPError as error:
                raise self._transport_error(error) from None
            if response.status_code >= 400:
                body = await self._read_error_body(response)
                retry = (
                    self._adjust_for_rejection(
                        response.status_code, body, attempt_max_tokens, request
                    )
                    if attempt + 1 < MAX_REQUEST_ATTEMPTS
                    else None
                )
                if retry is not None:
                    attempt_max_tokens = retry
                    continue
                raise self._status_error(response.status_code, body, model)
            break
        try:
            body = response.json()
        except ValueError:
            raise AssistantError(
                "provider_error", f"{self.endpoint.display_name} returned a malformed response."
            ) from None
        if not isinstance(body, dict):
            raise AssistantError(
                "provider_error", f"{self.endpoint.display_name} returned a malformed response."
            )
        if body.get("error"):
            raise AssistantError(
                "provider_error", f"{self.endpoint.display_name} reported an error."
            )
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, dict):
            choice = {}
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        return ChatResult(
            text=strip_reasoning(content) if isinstance(content, str) else "",
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") else None,
            usage=_usage(body.get("usage")) or {},
        )


def build_client(llm: Any, environ: Mapping[str, str]) -> ChatClient:
    """Default factory used by the assistant service."""
    return ChatClient(resolve_endpoint(llm, environ))


def elapsed_ms(started_ns: int) -> float:
    return round((time.monotonic_ns() - started_ns) / 1_000_000.0, 1)
