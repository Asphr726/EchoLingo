"""DeepL API v2 translation adapter.

DeepL is a request/response translator: one HTTPS ``POST /v2/translate`` per
stable source unit, one FINAL event per request (``RestTranslationBase``).
The registry marks the provider ``streaming_partials=False`` so the scheduler
never sends the provisional live tail, which keeps the API Free quota
(500,000 characters per month) for text that is actually committed.

Host selection
    ``tier="free"`` uses ``api-free.deepl.com``; ``tier="pro"`` uses
    ``api.deepl.com``. DeepL API Free keys always end with ``":fx"`` and are
    rejected by the Pro host with HTTP 403, so a key with that suffix is sent
    to the Free host regardless of the configured tier.

What leaves the machine
    The source text of the unit being translated, the previous source spans as
    DeepL ``context`` (never any target text, so nothing is copied back), the
    language codes and the model type. No audio, no target-side history, no
    glossary terms (see below). The key travels only in the
    ``Authorization`` header, never in the URL, log lines or error messages.

Glossary
    DeepL only applies glossaries that were created beforehand through its
    glossary endpoint and referenced by ``glossary_id``; it has no per-request
    term list. ``request.terms`` and ``set_glossary`` are therefore ignored by
    this adapter (a single warning names the count, never the terms).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ...errors import AuthenticationError, NetworkError, RateLimitError
from ...models import TranslationRequest
from ...translation.policy import TranslationRequestError
from .rest_base import RestTranslationBase

logger = logging.getLogger(__name__)

FREE_HOST = "https://api-free.deepl.com"
PRO_HOST = "https://api.deepl.com"
TRANSLATE_PATH = "/v2/translate"
FREE_KEY_SUFFIX = ":fx"
TIERS = ("free", "pro")
DEFAULT_MODEL_TYPE = "latency_optimized"
# Documented DeepL values; others are passed through so a new DeepL option
# does not need an EchoLingo release.
MODEL_TYPES = ("latency_optimized", "quality_optimized", "prefer_quality_optimized")
USER_AGENT = "EchoLingo"

# DeepL source languages are bare ISO 639-1 codes (no region).
_SOURCE_LANGS = {"en": "EN", "zh": "ZH", "ja": "JA", "ko": "KO"}
# Target languages need a variant for English and Chinese.
_TARGET_LANGS = {
    "en": "EN-US",
    "en-us": "EN-US",
    "en-gb": "EN-GB",
    "zh": "ZH-HANS",
    "zh-cn": "ZH-HANS",
    "zh-hans": "ZH-HANS",
    "zh-sg": "ZH-HANS",
    "zh-tw": "ZH-HANT",
    "zh-hk": "ZH-HANT",
    "zh-hant": "ZH-HANT",
    "ja": "JA",
    "ko": "KO",
    "pt": "PT-BR",
}
_AUTO_SOURCE = {"", "auto", "und"}


def _normalise_lang(code: str | None) -> str:
    return (code or "").strip().replace("_", "-").lower()


def source_lang_code(code: str | None) -> str | None:
    """DeepL ``source_lang`` for an EchoLingo code, ``None`` for auto-detect."""
    value = _normalise_lang(code)
    if value in _AUTO_SOURCE:
        return None
    base = value.split("-", 1)[0]
    return _SOURCE_LANGS.get(base, base.upper())


def target_lang_code(code: str) -> str:
    """DeepL ``target_lang`` for an EchoLingo code (EN and ZH need a variant)."""
    value = _normalise_lang(code)
    if value in _TARGET_LANGS:
        return _TARGET_LANGS[value]
    base = value.split("-", 1)[0]
    if base in _TARGET_LANGS:
        return _TARGET_LANGS[base]
    return value.upper()


class DeepLTranslation(RestTranslationBase):
    provider_id = "deepl"
    display_name = "DeepL"

    def __init__(
        self,
        *,
        api_key: str | None,
        tier: str = "free",
        model_type: str = DEFAULT_MODEL_TYPE,
        transcript_upload_allowed: bool = False,
        timeout_s: float = 20.0,
        client: httpx.AsyncClient | None = None,
        context_spans: int = 2,
        context_max_chars: int = 600,
    ) -> None:
        tier = (tier or "free").strip().lower()
        if tier not in TIERS:
            raise ValueError("DeepL tier must be free or pro")
        self.tier = tier
        self.model_type = (model_type or DEFAULT_MODEL_TYPE).strip().lower() or DEFAULT_MODEL_TYPE
        self.model = f"deepl-{self.model_type}"
        self.context_spans = max(0, int(context_spans))
        self.context_max_chars = max(0, int(context_max_chars))
        self.last_detected_source_language: str | None = None
        self._glossary_warned = False
        super().__init__(
            api_key=api_key,
            transcript_upload_allowed=transcript_upload_allowed,
            timeout_s=timeout_s,
            client=client,
        )

    # ------------------------------------------------------------- endpoint

    @property
    def uses_free_host(self) -> bool:
        """Free keys (``:fx`` suffix) always go to the Free host."""
        if self.api_key and self.api_key.endswith(FREE_KEY_SUFFIX):
            return True
        return self.tier == "free"

    @property
    def host(self) -> str:
        return FREE_HOST if self.uses_free_host else PRO_HOST

    def translate_url(self) -> str:
        return f"{self.host}{TRANSLATE_PATH}"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"DeepL-Auth-Key {self.api_key}"
        return headers

    def describe_endpoint(self) -> dict[str, object]:
        """Non-secret diagnostics for doctor output and tests."""
        return {
            "host": self.host,
            "tier": self.tier,
            "free_key": bool(self.api_key and self.api_key.endswith(FREE_KEY_SUFFIX)),
            "model_type": self.model_type,
        }

    # -------------------------------------------------------------- payload

    def _context(self, request: TranslationRequest) -> str:
        """The last ``context_spans`` non-empty previous *source* spans.

        Target text is never sent as context: DeepL would treat it as more
        source and the translation drifts toward copying it.
        """
        if not self.context_spans:
            return ""
        spans: list[str] = []
        total = 0
        for item in reversed(request.context):
            source = item.source.strip()
            if not source:
                continue
            if total + len(source) > self.context_max_chars:
                break
            spans.append(source)
            total += len(source)
            if len(spans) >= self.context_spans:
                break
        return "\n".join(reversed(spans))

    def build_payload(self, request: TranslationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": [request.source_text],
            "target_lang": target_lang_code(request.target_lang),
        }
        source = source_lang_code(request.source_lang)
        if source is not None:
            payload["source_lang"] = source
        context = self._context(request)
        if context:
            payload["context"] = context
        payload["model_type"] = self.model_type
        payload["preserve_formatting"] = True
        payload["split_sentences"] = "nonewlines"
        return payload

    def _warn_glossary(self, request: TranslationRequest) -> None:
        terms = request.terms or self.glossary
        if terms and not self._glossary_warned:
            self._glossary_warned = True
            logger.warning(
                "DeepL ignores %d glossary term(s): it only applies glossaries created "
                "in advance through its glossary endpoint (glossary_id)",
                len(terms),
            )

    # --------------------------------------------------------------- errors

    def _redact(self, text: str) -> str:
        if self.api_key and self.api_key in text:
            text = text.replace(self.api_key, "[redacted]")
        return text

    @staticmethod
    def _provider_message(response: httpx.Response) -> str:
        """DeepL's ``{"message": ...}`` body, if any, truncated for display."""
        try:
            body = response.json()
        except Exception:
            return ""
        if isinstance(body, dict):
            message = body.get("message") or body.get("detail") or ""
            return str(message)[:200].strip()
        return ""

    def raise_status(
        self, response: httpx.Response, request: TranslationRequest | None = None
    ) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = self._redact(self._provider_message(response))
        suffix = f" DeepL said: {detail}" if detail else ""
        has_context = bool(request is not None and self._context(request))
        if status in (401, 403):
            raise AuthenticationError(
                f"DeepL rejected the authentication key (HTTP {status}). "
                "Free keys end with ':fx' and must use the API Free plan." + suffix
            )
        if status == 456:
            raise RateLimitError(
                "DeepL character quota exceeded for this billing period (HTTP 456)." + suffix
            )
        if status in (429, 529):
            raise RateLimitError(f"DeepL rate limit exceeded (HTTP {status}); retry later." + suffix)
        if status == 400:
            lowered = detail.lower()
            unsupported = "not supported" in lowered or "unsupported" in lowered
            if unsupported and "lang" in lowered:
                raise TranslationRequestError(
                    "bad_request",
                    "DeepL does not support the requested language pair (HTTP 400)." + suffix,
                )
            raise TranslationRequestError(
                "bad_request",
                "DeepL rejected the request (HTTP 400)." + suffix,
                retry_without_context=has_context,
            )
        if status == 413:
            raise TranslationRequestError(
                "request_too_large",
                "DeepL rejected the request as too large (HTTP 413).",
                retry_without_context=has_context,
            )
        if status == 404:
            raise TranslationRequestError(
                "not_found", f"DeepL endpoint not found on {self.host} (HTTP 404)."
            )
        raise TranslationRequestError(
            f"http_{status}", f"DeepL returned HTTP {status} from {self.host}." + suffix
        )

    # ------------------------------------------------------------ translate

    async def translate_text(self, request: TranslationRequest) -> tuple[str, str]:
        if not request.source_text.strip():
            return "", self.model
        self._warn_glossary(request)
        try:
            response = await self.client.post(
                self.translate_url(),
                headers=self.headers(),
                json=self.build_payload(request),
                timeout=self.request_timeout(request),
            )
        except httpx.TimeoutException:
            raise  # the base maps timeouts to ``provider_timeout``
        except httpx.TransportError as error:
            # httpx messages can carry the URL; the key is never in it, but
            # keep the message to the error class anyway.
            raise NetworkError(
                f"DeepL request to {self.host} failed: {type(error).__name__}"
            ) from error
        self.raise_status(response, request)
        try:
            body = response.json()
        except ValueError as error:
            raise TranslationRequestError(
                "invalid_response", "DeepL returned a non-JSON response"
            ) from error
        translations = body.get("translations") if isinstance(body, dict) else None
        if not translations or not isinstance(translations[0], dict):
            raise TranslationRequestError("empty_response", "DeepL returned no translation")
        first = translations[0]
        detected = first.get("detected_source_language")
        if detected:
            self.last_detected_source_language = str(detected)
        used = str(first.get("model_type_used") or "").strip().lower()
        model = f"deepl-{used}" if used else self.model
        return str(first.get("text") or ""), model
