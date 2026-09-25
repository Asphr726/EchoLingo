"""Translation over any OpenAI-compatible chat completions endpoint.

One adapter serves the presets registered in ``backends/registry.py``
(``openai_chat``, ``deepseek_chat``, ``gemini_chat``, ``groq_chat``,
``openrouter_chat``, ``siliconflow_chat``) and ``custom_chat`` for self-hosted
servers such as Ollama, LM Studio or vLLM. The registry factory resolves the
base URL, model and key; this module only decides what the request looks like
and how the answer is cleaned up.

Prompt lessons carried over from the local Hy-MT adapter: earlier *source*
spans give a chat model useful context, but earlier *target* text in the
prompt gets copied back as the "translation", so it is never included.
Only the transcript text leaves the machine, and only after the explicit
transcript-upload flag is set.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx

from ...errors import AuthenticationError, ConfigurationError, PolicyDeniedError
from ...models import BackendDescriptor, CanonicalTranslationEvent, TranslationRequest
from ...translation.policy import TranslationRequestError
from ._text import LANGUAGE_NAMES, strip_instruction_echo, strip_wrapping_quotes
from .chat_base import COMMIT_TIMEOUT_S, OpenAiCompatibleChatTranslation

logger = logging.getLogger(__name__)

CUSTOM_PROVIDER = "custom_chat"

DISPLAY_NAMES: dict[str, str] = {
    "openai_chat": "OpenAI",
    "deepseek_chat": "DeepSeek",
    "gemini_chat": "Google Gemini",
    "groq_chat": "Groq",
    "openrouter_chat": "OpenRouter",
    "siliconflow_chat": "SiliconFlow",
    CUSTOM_PROVIDER: "Custom endpoint",
}

# Endpoints known to accept ``stream_options.include_usage``. Gemini's OpenAI
# compatibility layer and unknown self-hosted servers may reject the field with
# HTTP 400, so it is only sent where it is known to work.
STREAM_USAGE_PROVIDERS: frozenset[str] = frozenset(
    {"openai_chat", "deepseek_chat", "groq_chat", "openrouter_chat", "siliconflow_chat"}
)

# Presets whose models reason ("think") before answering unless the request
# turns it off, with the field that does. Reasoning would spend the small
# caption budget before any translation is written. An endpoint that rejects
# the field (HTTP 400/422 naming it) gets requests without it from then on.
THINKING_OFF_FIELDS: dict[str, tuple[str, object]] = {
    "deepseek_chat": ("thinking", {"type": "disabled"}),
}

# OpenRouter attribution headers (https://openrouter.ai/docs/api-reference/overview).
OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/echolingo",
    "X-Title": "EchoLingo",
}

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

# "Translation: ..." style labels a chat model sometimes prepends despite the
# instruction; ``strip_instruction_echo`` handles the longer preambles.
_LABEL_PREFIX = re.compile(
    r"^\s*\**\s*(?:translation|translated text|target text|译文|翻译|翻译结果)"
    r"\s*\**\s*[:：]\s*\**\s*",
    re.IGNORECASE,
)


class _ThinkingFieldRejected(TranslationRequestError):
    """The endpoint refused the thinking-off field the request carried; the
    request is sent once more without it."""


def language_label(code: str) -> str:
    """English language name for the prompt, falling back to the code."""
    names = LANGUAGE_NAMES.get((code or "").lower())
    return names[1] if names else code


class OpenAiChatTranslation(OpenAiCompatibleChatTranslation):
    """Chat-completions translation shared by every OpenAI-compatible preset."""

    provider_id = "openai_chat"
    display_name = "OpenAI"

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transcript_upload_allowed: bool = False,
        timeout_s: float = COMMIT_TIMEOUT_S,
        temperature: float = 0.1,
        max_output_tokens: int = 400,
        background_spans: int = 2,
        background_max_chars: int = 600,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Instance attributes shadow the class defaults so the base class's
        # descriptor, events and error messages carry the preset identity.
        self.provider = provider
        self.provider_id = provider
        self.display_name = DISPLAY_NAMES.get(provider, provider)
        # Dropped for the rest of this backend's life once the endpoint
        # rejects it.
        self._thinking_off = THINKING_OFF_FIELDS.get(provider)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.background_spans = int(background_spans)
        self.background_max_chars = int(background_max_chars)
        base_url = (base_url or "").strip()
        owns_client = client is None
        if client is None and _is_loopback(base_url):
            # A self-hosted server on this machine never goes through HTTPS_PROXY.
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s, connect=10.0), trust_env=False
            )
        super().__init__(
            base_url=base_url,
            model=(model or "").strip(),
            api_key=api_key,
            transcript_upload_allowed=transcript_upload_allowed,
            timeout_s=timeout_s,
            client=client,
        )
        self._owns_client = owns_client
        self.descriptor = BackendDescriptor(
            self.provider_id,
            self.model,
            self.locality,
            transcript_upload_required=self.transcript_upload_required,
        )

    # ---------------------------------------------------------------- policy

    @property
    def is_custom(self) -> bool:
        return self.provider == CUSTOM_PROVIDER

    def credentials_present(self) -> bool:
        if self.is_custom:
            # Local servers (Ollama, LM Studio, llama-server) accept no key.
            return bool(self.base_url)
        return bool(self.api_key)

    def authorize(self) -> None:
        if not self.transcript_upload_allowed:
            raise PolicyDeniedError("Cloud translation requires explicit transcript upload consent")
        if not self.base_url:
            if self.is_custom:
                raise ConfigurationError("Set the base URL for the custom endpoint")
            raise ConfigurationError(f"{self.display_name} has no base URL configured")
        if not self.credentials_present():
            raise AuthenticationError(f"{self.display_name} requires an API key.")

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter_chat":
            headers.update(OPENROUTER_HEADERS)
        return headers

    # ---------------------------------------------------------------- prompt

    def _background(self, request: TranslationRequest) -> list[str]:
        """Previous SOURCE sentences only: target text in the prompt gets copied."""
        if self.background_spans <= 0:
            return []
        spans: list[str] = []
        total = 0
        for item in reversed(request.context[-self.background_spans :]):
            source = item.source.strip()
            if not source:
                continue
            if total + len(source) > self.background_max_chars:
                break
            spans.append(source)
            total += len(source)
        spans.reverse()
        return spans

    def system_prompt(self, request: TranslationRequest) -> str:
        target = language_label(request.target_lang)
        source_code = (request.source_lang or "").strip().lower()
        if source_code in {"", "auto"}:
            direction = f"Translate the speaker's language into {target}."
        else:
            direction = f"Translate from {language_label(request.source_lang)} into {target}."
        lines = [
            "You are a professional simultaneous interpreter for lecture transcripts.",
            direction,
            "Output ONLY the translation: no explanations, no quotes, no preface.",
            "Keep numbers, names, code identifiers and technical terms accurate.",
        ]
        topic = " ".join((request.domain or "").split())[:400]
        if topic:
            lines.append(f"Session topic (background, do not translate): {topic}")
        terms = request.terms or self.glossary
        if terms:
            lines.append("")
            lines.append("Glossary:")
            lines.extend(f"{item.source} => {item.target}" for item in terms)
        background = self._background(request)
        if background:
            lines.append("")
            lines.append("Earlier context (do not translate):")
            lines.extend(background)
        if not request.source_committed:
            lines.append("")
            lines.append(
                "The text may be an unfinished sentence; translate what is present "
                "without completing it."
            )
        return "\n".join(lines)

    def max_tokens(self, request: TranslationRequest) -> int:
        return min(self.max_output_tokens, max(24, 3 * len(request.source_text) + 16))

    def build_payload(self, request: TranslationRequest, *, model: str, stream: bool) -> dict:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt(request)},
                {"role": "user", "content": request.source_text.strip()},
            ],
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens(request),
        }
        if stream and self.provider in STREAM_USAGE_PROVIDERS:
            payload["stream_options"] = {"include_usage": True}
        if self._thinking_off is not None:
            name, value = self._thinking_off
            payload[name] = value
        return payload

    # ------------------------------------------------------------- requests

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        start = super().translate_incremental

        async def generate():
            events = start(request)
            try:
                # The status is checked before the first event, so a request
                # whose thinking field was rejected has emitted nothing yet.
                try:
                    first = await anext(events)
                except _ThinkingFieldRejected:
                    events = start(request)
                    first = await anext(events)
                yield first
                async for event in events:
                    yield event
            finally:
                await events.aclose()

        return generate()

    async def retranslate_window(self, request: TranslationRequest) -> CanonicalTranslationEvent:
        try:
            return await super().retranslate_window(request)
        except _ThinkingFieldRejected:
            return await super().retranslate_window(request)

    # --------------------------------------------------------------- answers

    def postprocess(self, text: str, request: TranslationRequest) -> str:
        text = strip_instruction_echo(text)
        text = _LABEL_PREFIX.sub("", text, count=1)
        cleaned = strip_wrapping_quotes(text)
        if cleaned == text.strip():
            # No quotes removed: keep trailing whitespace so the next streamed
            # delta ("Hello " + "world") is not fused into one word.
            return text.lstrip()
        return cleaned

    def _thinking_rejected(self, response: httpx.Response, body: str) -> bool:
        """Whether an HTTP 400/422 names the thinking-off field that the
        rejected request carried. The field is then left out of every later
        request."""
        field = THINKING_OFF_FIELDS.get(self.provider)
        if field is None or field[0] not in body.lower():
            return False
        if not _request_has_field(response, field[0]):
            return False
        if self._thinking_off is not None:
            self._thinking_off = None
            logger.info(
                "%s rejected the %r request field (HTTP %d); sending requests without it",
                self.display_name,
                field[0],
                response.status_code,
            )
        return True

    async def raise_status(self, response: httpx.Response, request: TranslationRequest) -> None:
        if response.status_code in (400, 422):
            body = ""
            try:
                body = (await response.aread()).decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - diagnostics only
                body = ""
            if self._thinking_rejected(response, body):
                raise _ThinkingFieldRejected(
                    "bad_request",
                    f"{self.display_name} rejected the request (HTTP {response.status_code})",
                    retry_without_context=False,
                )
        if response.status_code == 400:
            lowered = body.lower()
            if "api key" in lowered and ("valid" in lowered or "invalid" in lowered):
                # Gemini's OpenAI endpoint answers 400 (not 401) for a bad key.
                raise AuthenticationError(
                    f"{self.display_name} rejected the API key (HTTP 400). "
                    "Verify that the key is active and the model is enabled for it."
                )
            if "stream_options" in body:
                hint = (
                    " Use the custom endpoint preset for this server."
                    if self.provider in STREAM_USAGE_PROVIDERS
                    else ""
                )
                raise TranslationRequestError(
                    "bad_request",
                    f"{self.display_name} rejected the request (HTTP 400): the endpoint does "
                    f"not accept stream_options.{hint}",
                    retry_without_context=False,
                )
        await super().raise_status(response, request)


def _request_has_field(response: httpx.Response, name: str) -> bool:
    """Whether the request that got ``response`` had a top-level ``name``."""
    try:
        payload = json.loads(response.request.content)
    except (RuntimeError, ValueError):
        return False
    return isinstance(payload, dict) and name in payload


def _is_loopback(base_url: str) -> bool:
    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


__all__ = [
    "DISPLAY_NAMES",
    "OPENROUTER_HEADERS",
    "OpenAiChatTranslation",
    "STREAM_USAGE_PROVIDERS",
    "THINKING_OFF_FIELDS",
    "language_label",
]
