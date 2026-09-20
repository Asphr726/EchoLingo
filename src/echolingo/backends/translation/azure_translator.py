"""Azure AI Translator (Text API v3.0) adapter.

Azure Translator is a request/response API: one HTTPS ``POST /translate`` per
stable source unit and one FINAL event per request (``RestTranslationBase``).
The registry marks the provider ``streaming_partials=False`` so the scheduler
never sends the provisional live tail; the F0 free tier (2 million characters
per month) is spent on committed text only.

Endpoint
    ``https://api.cognitive.microsofttranslator.com`` by default (the global
    endpoint, which routes to the nearest datacenter). A custom value must
    point at the directory that contains ``/translate``; for a custom-domain
    resource that is ``https://<name>.cognitiveservices.azure.com/translator/text/v3.0``.

Authentication
    The subscription key travels only in the ``Ocp-Apim-Subscription-Key``
    header, never in the URL, log lines or error messages. Regional resources
    (created in e.g. ``eastasia``) also need ``Ocp-Apim-Subscription-Region``;
    global resources do not, so the header is omitted when the region setting
    is empty.

What leaves the machine
    The source text of the unit being translated, the language codes and a
    random ``X-ClientTraceId``. No audio, no history, no target text, no
    glossary terms (Azure's dynamic dictionary needs HTML markup, which this
    adapter does not use: ``textType=plain``).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from ...errors import AuthenticationError, NetworkError, RateLimitError
from ...models import TranslationRequest
from ...translation.policy import TranslationRequestError
from .rest_base import RestTranslationBase

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
TRANSLATE_PATH = "/translate"
API_VERSION = "3.0"
TEXT_TYPE = "plain"
MODEL_NAME = "translator-v3"
USER_AGENT = "EchoLingo"

# Azure language codes are BCP-47; Chinese needs an explicit script tag.
_LANGS = {
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-sg": "zh-Hans",
    "zh-hans": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zh-hk": "zh-Hant",
    "zh-mo": "zh-Hant",
    "zh-hant": "zh-Hant",
    "ja": "ja",
    "ko": "ko",
}
_AUTO_SOURCE = {"", "auto", "und"}

# Azure error codes (the ``error.code`` number in the JSON body) that mean the
# language pair itself is not supported rather than the request being malformed.
_LANGUAGE_ERROR_CODES = {400003, 400006, 400019, 400023, 400035, 400036}
_TOO_LONG_ERROR_CODES = {400050, 400077}
_FREE_QUOTA_ERROR_CODE = 403001
_SPEECH_KEY_ERROR_CODE = 401015


def _normalise_lang(code: str | None) -> str:
    return (code or "").strip().replace("_", "-").lower()


def source_lang_code(code: str | None) -> str | None:
    """Azure ``from`` for an EchoLingo code, ``None`` for auto-detect."""
    value = _normalise_lang(code)
    if value in _AUTO_SOURCE:
        return None
    return _LANGS.get(value) or _LANGS.get(value.split("-", 1)[0]) or value


def target_lang_code(code: str) -> str:
    """Azure ``to`` for an EchoLingo code (``zh`` needs the ``zh-Hans`` script tag)."""
    value = _normalise_lang(code)
    return _LANGS.get(value) or _LANGS.get(value.split("-", 1)[0]) or value


def normalise_endpoint(endpoint: str | None) -> str:
    value = (endpoint or "").strip().rstrip("/")
    return value or DEFAULT_ENDPOINT


def normalise_region(region: str | None) -> str:
    """Azure region identifiers are lowercase without spaces (``eastasia``)."""
    return (region or "").strip().lower().replace(" ", "")


class AzureTranslator(RestTranslationBase):
    provider_id = "azure_translator"
    display_name = "Azure AI Translator"
    model = MODEL_NAME

    def __init__(
        self,
        *,
        api_key: str | None,
        region: str = "",
        endpoint: str = DEFAULT_ENDPOINT,
        transcript_upload_allowed: bool = False,
        timeout_s: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.region = normalise_region(region)
        self.endpoint = normalise_endpoint(endpoint)
        self.last_detected_source_language: str | None = None
        self.last_trace_id: str | None = None
        self._glossary_warned = False
        super().__init__(
            api_key=api_key,
            transcript_upload_allowed=transcript_upload_allowed,
            timeout_s=timeout_s,
            client=client,
        )

    # ------------------------------------------------------------- endpoint

    @property
    def host(self) -> str:
        """Scheme and host of the endpoint, for messages and diagnostics."""
        parsed = httpx.URL(self.endpoint)
        return f"{parsed.scheme}://{parsed.host}" if parsed.host else self.endpoint

    def translate_url(self) -> str:
        return f"{self.endpoint}{TRANSLATE_PATH}"

    def build_params(self, request: TranslationRequest) -> dict[str, str]:
        params = {"api-version": API_VERSION, "to": target_lang_code(request.target_lang)}
        source = source_lang_code(request.source_lang)
        if source is not None:
            params["from"] = source
        params["textType"] = TEXT_TYPE
        return params

    def headers(self, trace_id: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        headers["X-ClientTraceId"] = trace_id or str(uuid.uuid4())
        return headers

    @staticmethod
    def build_body(request: TranslationRequest) -> list[dict[str, str]]:
        return [{"Text": request.source_text}]

    def describe_endpoint(self) -> dict[str, object]:
        """Non-secret diagnostics for doctor output and tests."""
        return {
            "host": self.host,
            "endpoint": self.endpoint,
            "region": self.region or None,
            "custom_endpoint": self.endpoint != DEFAULT_ENDPOINT,
        }

    def _warn_glossary(self, request: TranslationRequest) -> None:
        terms = request.terms or self.glossary
        if terms and not self._glossary_warned:
            self._glossary_warned = True
            logger.warning(
                "Azure Translator ignores %d glossary term(s): the dynamic dictionary "
                "needs HTML markup and this adapter sends textType=plain",
                len(terms),
            )

    # --------------------------------------------------------------- errors

    def _redact(self, text: str) -> str:
        if self.api_key and self.api_key in text:
            text = text.replace(self.api_key, "[redacted]")
        return text

    @staticmethod
    def _provider_error(response: httpx.Response) -> tuple[int | None, str]:
        """Azure's ``{"error": {"code": 400036, "message": ...}}`` body, if any."""
        try:
            body = response.json()
        except Exception:
            return None, ""
        if not isinstance(body, dict):
            return None, ""
        error = body.get("error")
        if not isinstance(error, dict):
            return None, ""
        code: int | None = None
        raw_code = error.get("code")
        if isinstance(raw_code, int) and not isinstance(raw_code, bool):
            code = raw_code
        elif isinstance(raw_code, str) and raw_code.strip().isdigit():
            code = int(raw_code.strip())
        message = str(error.get("message") or "")[:200].strip()
        return code, message

    def raise_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        code, message = self._provider_error(response)
        message = self._redact(message)
        parts = [f"HTTP {status}"]
        if code is not None:
            parts.append(f"Azure {code}")
        where = f" ({', '.join(parts)})"
        suffix = f" Azure said: {message}" if message else ""
        if status == 401:
            if code == _SPEECH_KEY_ERROR_CODE:
                raise AuthenticationError(
                    f"Azure Translator rejected the key{where}: this is a Speech resource key; "
                    "create a Translator resource and use its key." + suffix
                )
            raise AuthenticationError(
                f"Azure Translator rejected the key{where}. Check the key and set the "
                "resource region (e.g. eastasia) for regional resources." + suffix
            )
        if status == 403:
            if code == _FREE_QUOTA_ERROR_CODE:
                raise AuthenticationError(
                    f"Azure Translator free (F0) quota exceeded for this month{where}. "
                    "Wait for the next billing period or move the resource to the S1 tier."
                    + suffix
                )
            raise AuthenticationError(
                f"Azure Translator refused the operation{where}. Check that the resource "
                "is a Translator resource and the key belongs to it." + suffix
            )
        if status == 429:
            raise RateLimitError(
                f"Azure Translator rate limit exceeded{where}; retry later." + suffix
            )
        if status == 400:
            if code in _LANGUAGE_ERROR_CODES:
                raise TranslationRequestError(
                    "bad_request",
                    f"Azure Translator does not support the requested language pair{where}."
                    + suffix,
                )
            if code in _TOO_LONG_ERROR_CODES:
                raise TranslationRequestError(
                    "request_too_large",
                    f"Azure Translator rejected the request as too long{where}." + suffix,
                )
            raise TranslationRequestError(
                "bad_request", f"Azure Translator rejected the request{where}." + suffix
            )
        if status == 404:
            raise TranslationRequestError(
                "not_found",
                f"Azure Translator endpoint not found on {self.host}{where}; "
                "a custom endpoint must end with the directory that contains /translate.",
            )
        if status == 408:
            raise TranslationRequestError(
                "provider_timeout", f"Azure Translator timed out server-side{where}." + suffix
            )
        if status == 413:
            raise TranslationRequestError(
                "request_too_large", f"Azure Translator rejected the request as too large{where}."
            )
        raise TranslationRequestError(
            f"http_{status}", f"Azure Translator returned HTTP {status} from {self.host}." + suffix
        )

    # ------------------------------------------------------------ translate

    async def translate_text(self, request: TranslationRequest) -> tuple[str, str]:
        if not request.source_text.strip():
            return "", self.model
        self._warn_glossary(request)
        trace_id = str(uuid.uuid4())
        self.last_trace_id = trace_id
        budget_s = self.request_timeout(request)
        try:
            # httpx timeouts are per phase; the scheduler's budget is wall-clock.
            response = await asyncio.wait_for(
                self.client.post(
                    self.translate_url(),
                    params=self.build_params(request),
                    headers=self.headers(trace_id),
                    json=self.build_body(request),
                    timeout=budget_s,
                ),
                timeout=budget_s,
            )
        except httpx.TimeoutException:
            raise  # the base maps timeouts to ``provider_timeout``
        except asyncio.TimeoutError as error:
            raise TranslationRequestError(
                "provider_timeout", f"Azure Translator timed out after {budget_s:.1f} s"
            ) from error
        except httpx.TransportError as error:
            # httpx messages can carry the URL; the key is never in it, but
            # keep the message to the host and the error class anyway.
            raise NetworkError(
                f"Azure Translator request to {self.host} failed: {type(error).__name__}"
            ) from error
        self.raise_status(response)
        try:
            body: Any = response.json()
        except ValueError as error:
            raise TranslationRequestError(
                "invalid_response", "Azure Translator returned a non-JSON response"
            ) from error
        first = body[0] if isinstance(body, list) and body else None
        translations = first.get("translations") if isinstance(first, dict) else None
        if not translations or not isinstance(translations[0], dict):
            raise TranslationRequestError(
                "empty_response", "Azure Translator returned no translation"
            )
        detected = first.get("detectedLanguage")
        if isinstance(detected, dict) and detected.get("language"):
            self.last_detected_source_language = str(detected["language"])
        return str(translations[0].get("text") or ""), self.model
