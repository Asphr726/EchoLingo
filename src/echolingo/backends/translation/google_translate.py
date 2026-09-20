"""Google Cloud Translation (Basic, v2) adapter.

Cloud Translation Basic is a request/response translator: one HTTPS
``POST /language/translate/v2`` per stable source unit, one FINAL event per
request (``RestTranslationBase``). The registry marks the provider
``streaming_partials=False`` so the scheduler never sends the provisional
live tail; the free tier (500,000 characters per month) is spent only on
text that is actually committed.

Authentication
    The API key travels in the ``X-goog-api-key`` request header, never as
    the ``?key=`` query parameter Google also accepts: a key in the URL would
    end up in proxy logs, ``httpx`` exception text and crash reports. Error
    messages never include the request URL, and any provider text quoted in a
    message has the key redacted.

What leaves the machine
    The source text of the unit being translated, the target language code
    and, unless the source is ``auto``, the source language code. No audio,
    no session history, no target text, no glossary terms (v2 has no
    glossary; ``request.terms`` and ``set_glossary`` are ignored, with one
    warning that names the count, never the terms).

Response handling
    v2 answers ``{"data": {"translations": [{"translatedText": ...}]}}``.
    Even with ``format: "text"`` the service occasionally HTML-escapes
    characters (``&#39;``, ``&amp;``), so the text is passed through
    ``html.unescape`` before it reaches the caption.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

import httpx

from ...errors import AuthenticationError, NetworkError, RateLimitError
from ...models import TranslationRequest
from ...translation.policy import TranslationRequestError
from .rest_base import RestTranslationBase

logger = logging.getLogger(__name__)

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
API_KEY_HEADER = "X-goog-api-key"
USER_AGENT = "EchoLingo"
MODEL = "nmt"

# Cloud Translation v2 uses ISO 639-1 codes; Chinese needs a script/region.
_LANGS = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-sg": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hk": "zh-TW",
    "zh-hant": "zh-TW",
}
_AUTO_SOURCE = {"", "auto", "und"}
# Google API keys look like "AIza" + 35 URL-safe characters. Redacting that
# shape covers keys that differ from the configured one (a rotated key echoed
# by a proxy, for example).
_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_INVALID_KEY_MARKERS = ("api key not valid", "api_key_invalid", "invalid api key")


def _normalise_lang(code: str | None) -> str:
    return (code or "").strip().replace("_", "-").lower()


def lang_code(code: str | None) -> str | None:
    """Cloud Translation v2 language code for an EchoLingo code.

    ``None`` means "let Google detect" and is only meaningful for the source.
    Unknown codes are passed through lower-cased (``de`` → ``de``).
    """
    value = _normalise_lang(code)
    if value in _AUTO_SOURCE:
        return None
    if value in _LANGS:
        return _LANGS[value]
    base = value.split("-", 1)[0]
    if base in _LANGS:
        return _LANGS[base]
    return base


class GoogleTranslateV2(RestTranslationBase):
    provider_id = "google_translate"
    display_name = "Google Cloud Translation"
    model = MODEL

    def __init__(
        self,
        *,
        api_key: str | None,
        transcript_upload_allowed: bool = False,
        timeout_s: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.last_detected_source_language: str | None = None
        self._glossary_warned = False
        super().__init__(
            api_key=api_key,
            transcript_upload_allowed=transcript_upload_allowed,
            timeout_s=timeout_s,
            client=client,
        )

    # ------------------------------------------------------------- endpoint

    def translate_url(self) -> str:
        """The v2 endpoint; the key is never part of it."""
        return TRANSLATE_URL

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers[API_KEY_HEADER] = self.api_key
        return headers

    def describe_endpoint(self) -> dict[str, object]:
        """Non-secret diagnostics for doctor output and tests."""
        return {"host": "translation.googleapis.com", "api_version": "v2", "model": self.model}

    # -------------------------------------------------------------- payload

    def build_payload(self, request: TranslationRequest) -> dict[str, Any]:
        target = lang_code(request.target_lang)
        if target is None:
            raise TranslationRequestError(
                "bad_request", "Google Cloud Translation needs an explicit target language."
            )
        payload: dict[str, Any] = {
            "q": [request.source_text],
            "target": target,
            "format": "text",
        }
        source = lang_code(request.source_lang)
        if source is not None:
            payload["source"] = source
        return payload

    def _warn_glossary(self, request: TranslationRequest) -> None:
        terms = request.terms or self.glossary
        if terms and not self._glossary_warned:
            self._glossary_warned = True
            logger.warning(
                "Google Cloud Translation v2 ignores %d glossary term(s): glossaries "
                "need the Advanced (v3) API",
                len(terms),
            )

    # --------------------------------------------------------------- errors

    def _redact(self, text: str) -> str:
        if self.api_key and self.api_key in text:
            text = text.replace(self.api_key, "[redacted]")
        return _KEY_SHAPE.sub("[redacted]", text)

    @staticmethod
    def _provider_message(response: httpx.Response) -> str:
        """Google's ``{"error": {"message": ...}}`` text, if any, truncated."""
        try:
            body = response.json()
        except Exception:
            return ""
        if not isinstance(body, dict):
            return ""
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status") or ""
        else:
            message = error or body.get("message") or ""
        return str(message)[:200].strip()

    @staticmethod
    def _body_says_invalid_key(response: httpx.Response) -> bool:
        """Google answers HTTP 400 (not 401) for a malformed or unknown key."""
        try:
            body = response.text[:4000].lower()
        except Exception:
            return False
        return any(marker in body for marker in _INVALID_KEY_MARKERS)

    def raise_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = self._redact(self._provider_message(response))
        suffix = f" Google said: {detail}" if detail else ""
        if status == 400 and self._body_says_invalid_key(response):
            raise AuthenticationError(
                "Google rejected the API key as not valid (HTTP 400). "
                "Re-copy the key from the Cloud console." + suffix
            )
        if status in (401, 403):
            raise AuthenticationError(
                "Google rejected the API key or the Cloud Translation API is not "
                f"enabled for the project (HTTP {status})." + suffix
            )
        if status == 429:
            raise RateLimitError(
                "Google Cloud Translation quota or rate limit exceeded (HTTP 429); "
                "retry later." + suffix
            )
        if status == 400:
            raise TranslationRequestError(
                "bad_request", "Google Cloud Translation rejected the request (HTTP 400)." + suffix
            )
        if status == 413:
            raise TranslationRequestError(
                "request_too_large",
                "Google Cloud Translation rejected the request as too large (HTTP 413).",
            )
        raise TranslationRequestError(
            f"http_{status}", f"Google Cloud Translation returned HTTP {status}." + suffix
        )

    # ------------------------------------------------------------ translate

    async def translate_text(self, request: TranslationRequest) -> tuple[str, str]:
        if not request.source_text.strip():
            return "", self.model
        self._warn_glossary(request)
        payload = self.build_payload(request)
        try:
            response = await self.client.post(
                self.translate_url(),
                headers=self.headers(),
                json=payload,
                timeout=self.request_timeout(request),
            )
        except httpx.TimeoutException:
            raise  # the base maps timeouts to ``provider_timeout``
        except httpx.TransportError as error:
            # httpx messages can carry the URL and the raw exception text;
            # only the error class name is surfaced.
            raise NetworkError(
                "Google Cloud Translation request to translation.googleapis.com "
                f"failed: {type(error).__name__}"
            ) from error
        self.raise_status(response)
        try:
            body = response.json()
        except ValueError as error:
            raise TranslationRequestError(
                "invalid_response", "Google Cloud Translation returned a non-JSON response"
            ) from error
        data = body.get("data") if isinstance(body, dict) else None
        translations = data.get("translations") if isinstance(data, dict) else None
        if not translations or not isinstance(translations[0], dict):
            raise TranslationRequestError(
                "empty_response", "Google Cloud Translation returned no translation"
            )
        first = translations[0]
        detected = first.get("detectedSourceLanguage")
        if detected:
            self.last_detected_source_language = str(detected)
        text = html.unescape(str(first.get("translatedText") or ""))
        model = str(first.get("model") or "").strip().lower() or self.model
        return text, model
