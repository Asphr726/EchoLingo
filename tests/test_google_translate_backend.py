"""Google Cloud Translation (Basic v2) adapter tests (httpx.MockTransport, no live network)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from echolingo.backends.translation.google_translate import (
    API_KEY_HEADER,
    TRANSLATE_URL,
    GoogleTranslateV2,
    lang_code,
)
from echolingo.errors import AuthenticationError, NetworkError, PolicyDeniedError, RateLimitError
from echolingo.models import (
    BackendLocality,
    GlossaryTerm,
    TranslationContextSegment,
    TranslationKind,
    TranslationRequest,
)
from echolingo.translation.policy import TranslationRequestError

# Shaped like a real Google API key (AIza + 35 characters) but not one.
KEY = "AIzaSyTESTKEY0123456789abcdefghijklmnop"


def request(**overrides) -> TranslationRequest:
    values = dict(
        request_id="r",
        source_revision_id=3,
        source_text="Welcome to the lecture.",
        source_lang="en",
        target_lang="zh",
        source_committed=True,
    )
    values.update(overrides)
    return TranslationRequest(**values)


def ok(text: str = "欢迎来到讲座。", **extra) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": {"translations": [{"translatedText": text, "detectedSourceLanguage": "en", **extra}]}},
    )


def google_error(status: int, message: str, reason: str = "") -> httpx.Response:
    body = {"error": {"code": status, "message": message, "status": reason or "ERROR"}}
    return httpx.Response(status, json=body)


class Recorder:
    """Captures the requests a backend sends through the mock transport."""

    def __init__(self, responses) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

    def __call__(self, http_request: httpx.Request) -> httpx.Response:
        self.requests.append(http_request)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def last_body(self) -> dict:
        return json.loads(self.last.content.decode("utf-8"))


def backend(*responses, key=KEY, allowed=True, **kwargs):
    recorder = Recorder(responses or [ok()])
    instance = GoogleTranslateV2(
        api_key=key,
        transcript_upload_allowed=allowed,
        client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        **kwargs,
    )
    return instance, recorder


# --------------------------------------------------------- headers + auth


async def test_key_travels_only_in_header_never_in_url() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request())
    sent = recorder.last
    assert sent.method == "POST"
    assert str(sent.url) == "https://translation.googleapis.com/language/translate/v2"
    assert str(sent.url) == TRANSLATE_URL
    assert sent.url.params == httpx.QueryParams()
    assert "key=" not in str(sent.url) and KEY not in str(sent.url)
    assert sent.headers[API_KEY_HEADER] == KEY
    assert sent.headers["x-goog-api-key"] == KEY
    assert sent.headers["Content-Type"] == "application/json"
    assert sent.headers["User-Agent"] == "EchoLingo"
    assert "Authorization" not in sent.headers
    assert KEY not in sent.content.decode("utf-8")
    assert KEY not in instance.translate_url()
    await instance.close()


def test_describe_endpoint_and_descriptor_have_no_secret() -> None:
    instance, _ = backend()
    assert instance.provider_id == "google_translate"
    assert instance.display_name == "Google Cloud Translation"
    assert instance.model == "nmt"
    assert instance.streaming_partials is False
    assert instance.describe_endpoint() == {
        "host": "translation.googleapis.com",
        "api_version": "v2",
        "model": "nmt",
    }
    assert KEY not in json.dumps(instance.describe_endpoint())
    assert instance.descriptor.provider == "google_translate"
    assert instance.descriptor.model == "nmt"
    assert instance.descriptor.locality is BackendLocality.CLOUD
    assert instance.descriptor.transcript_upload_required is True


# ------------------------------------------------------- body + languages


async def test_body_matches_v2_contract() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request())
    assert recorder.last_body() == {
        "q": ["Welcome to the lecture."],
        "target": "zh-CN",
        "source": "en",
        "format": "text",
    }


def test_language_code_mapping() -> None:
    assert lang_code("en") == "en"
    assert lang_code("zh") == "zh-CN"
    assert lang_code("ja") == "ja"
    assert lang_code("ko") == "ko"
    assert lang_code("zh-CN") == "zh-CN"
    assert lang_code("zh_Hans") == "zh-CN"
    assert lang_code("zh-TW") == "zh-TW"
    assert lang_code("zh-Hant") == "zh-TW"
    assert lang_code("en-US") == "en"
    assert lang_code("DE") == "de"
    assert lang_code("auto") is None
    assert lang_code("") is None
    assert lang_code(None) is None


async def test_each_product_language_as_target() -> None:
    instance, recorder = backend()
    for code, expected in (("en", "en"), ("zh", "zh-CN"), ("ja", "ja"), ("ko", "ko")):
        await instance.retranslate_window(request(source_lang="auto", target_lang=code))
        body = recorder.last_body()
        assert body["target"] == expected
        assert "source" not in body


async def test_auto_source_omits_source_and_records_detection() -> None:
    instance, recorder = backend(ok("Welcome.", detectedSourceLanguage="ja"))
    await instance.retranslate_window(request(source_lang="auto", target_lang="en", source_text="ようこそ。"))
    assert "source" not in recorder.last_body()
    assert instance.last_detected_source_language == "ja"


async def test_missing_target_language_is_rejected_before_any_request() -> None:
    instance, recorder = backend()
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(target_lang="auto"))
    assert info.value.error_code == "bad_request"
    assert recorder.requests == []


async def test_context_and_target_history_are_never_sent() -> None:
    instance, recorder = backend()
    context = (
        TranslationContextSegment("First we cover vectors.", "首先我们讲向量。"),
        TranslationContextSegment("Then matrices.", "然后是矩阵。"),
    )
    await instance.retranslate_window(request(source_text="Now determinants.", context=context))
    raw = recorder.last.content.decode("utf-8")
    assert recorder.last_body()["q"] == ["Now determinants."]
    for text in ("First we cover vectors.", "Then matrices.", "首先我们讲向量。", "然后是矩阵。"):
        assert text not in raw


# ---------------------------------------------------------------- events


async def test_translate_incremental_yields_single_final() -> None:
    instance, _ = backend()
    events = [event async for event in instance.translate_incremental(request())]
    assert len(events) == 1
    event = events[0]
    assert event.kind == TranslationKind.FINAL
    assert event.text == "欢迎来到讲座。"
    assert event.committed_text == "欢迎来到讲座。" and event.editable_text == ""
    assert event.provider == "google_translate"
    assert event.model == "nmt"
    assert event.locality is BackendLocality.CLOUD
    assert event.request_id == "r" and event.source_revision_id == 3
    assert event.source_committed is True
    assert event.truncated is False and event.finish_reason == "stop"
    assert event.total_latency_ms is not None and event.first_delta_latency_ms is not None
    await instance.close()


async def test_html_entities_are_unescaped_even_for_text_format() -> None:
    instance, _ = backend(ok("Tom &amp; Jerry&#39;s &quot;lecture&quot; &lt;3"))
    event = await instance.retranslate_window(request(target_lang="en"))
    assert event.text == "Tom & Jerry's \"lecture\" <3"


async def test_provider_reported_model_overrides_default() -> None:
    instance, _ = backend(ok(model="NMT"))
    event = await instance.retranslate_window(request())
    assert event.model == "nmt"
    instance, _ = backend(ok(model="base"))
    event = await instance.retranslate_window(request())
    assert event.model == "base"


async def test_empty_source_text_is_not_sent() -> None:
    instance, recorder = backend()
    event = await instance.retranslate_window(request(source_text="   "))
    assert event.text == "" and event.kind == TranslationKind.FINAL
    assert recorder.requests == []


async def test_glossary_terms_are_ignored_and_never_logged(caplog) -> None:
    instance, recorder = backend()
    await instance.set_glossary((GlossaryTerm("determinant", "行列式"),))
    terms = (GlossaryTerm("eigenvalue", "特征值"),)
    with caplog.at_level(logging.WARNING, logger="echolingo.backends.translation.google_translate"):
        await instance.retranslate_window(request(terms=terms))
        await instance.retranslate_window(request(terms=terms))
    body = recorder.last_body()
    assert set(body) == {"q", "target", "source", "format"}
    raw = recorder.last.content.decode("utf-8")
    assert "eigenvalue" not in raw and "行列式" not in raw
    warnings = [record for record in caplog.records if "glossary" in record.getMessage()]
    assert len(warnings) == 1
    assert "eigenvalue" not in warnings[0].getMessage()
    assert "特征值" not in warnings[0].getMessage()
    assert KEY not in warnings[0].getMessage()


# ---------------------------------------------------------------- errors


async def test_400_invalid_key_maps_to_authentication_error_without_key() -> None:
    instance, _ = backend(
        google_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    )
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    message = str(info.value)
    assert "HTTP 400" in message and "not valid" in message
    assert KEY not in message and "key=" not in message
    assert "translation.googleapis.com" not in message
    assert info.value.error_code == "authentication_failed"


async def test_400_invalid_key_detected_in_non_json_body() -> None:
    instance, _ = backend(httpx.Response(400, text=f"API key not valid: {KEY}"))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    assert KEY not in str(info.value)


async def test_403_maps_to_authentication_error_with_required_message() -> None:
    instance, _ = backend(httpx.Response(403))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    assert str(info.value) == (
        "Google rejected the API key or the Cloud Translation API is not enabled "
        "for the project (HTTP 403)."
    )
    assert info.value.error_code == "authentication_failed"

    instance, _ = backend(
        google_error(
            403,
            "Cloud Translation API has not been used in project 123 before or it is disabled.",
            "PERMISSION_DENIED",
        )
    )
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    message = str(info.value)
    assert message.startswith(
        "Google rejected the API key or the Cloud Translation API is not enabled "
        "for the project (HTTP 403)."
    )
    assert "has not been used" in message
    assert KEY not in message and "translation.googleapis.com" not in message


async def test_401_maps_to_authentication_error() -> None:
    instance, _ = backend(google_error(401, "Request had invalid authentication credentials.", "UNAUTHENTICATED"))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    assert "HTTP 401" in str(info.value) and KEY not in str(info.value)


async def test_429_maps_to_rate_limit() -> None:
    instance, _ = backend(google_error(429, "Quota exceeded for quota metric 'v2 characters'", "RESOURCE_EXHAUSTED"))
    with pytest.raises(RateLimitError) as info:
        await instance.retranslate_window(request())
    message = str(info.value)
    assert "429" in message and "Quota exceeded" in message
    assert info.value.error_code == "rate_limited" and info.value.recoverable is True
    assert KEY not in message and "translation.googleapis.com" not in message

    instance, _ = backend(httpx.Response(429, text="Too many requests"))
    with pytest.raises(RateLimitError):
        await instance.retranslate_window(request())


async def test_other_4xx_and_5xx_map_to_http_codes() -> None:
    instance, _ = backend(google_error(400, "Invalid Value", "INVALID_ARGUMENT"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "bad_request"
    assert "Invalid Value" in str(info.value) and KEY not in str(info.value)

    instance, _ = backend(google_error(503, "The service is currently unavailable.", "UNAVAILABLE"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "http_503"
    assert "503" in str(info.value) and KEY not in str(info.value)
    assert "translation.googleapis.com" not in str(info.value)

    instance, _ = backend(httpx.Response(413))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "request_too_large"

    instance, _ = backend(httpx.Response(404))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "http_404"


async def test_key_echoed_by_provider_is_redacted() -> None:
    instance, _ = backend(google_error(500, f"key {KEY} broke", "INTERNAL"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert KEY not in str(info.value) and "[redacted]" in str(info.value)

    # A key other than the configured one (rotated, or echoed by a proxy) is
    # still recognised by its shape.
    other = "AIzaSyOTHERKEY0123456789abcdefghijklmnop"
    instance, _ = backend(google_error(500, f"bad {other}", "INTERNAL"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert other not in str(info.value) and "[redacted]" in str(info.value)


async def test_malformed_and_empty_responses() -> None:
    instance, _ = backend(httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "invalid_response"

    instance, _ = backend(httpx.Response(200, json={"data": {"translations": []}}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "empty_response"

    instance, _ = backend(httpx.Response(200, json={"translations": [{"translatedText": "x"}]}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "empty_response"


async def test_timeout_and_transport_errors_are_mapped_without_key() -> None:
    def timeout(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=http_request)

    instance = GoogleTranslateV2(
        api_key=KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(timeout)),
    )
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(timeout_s=0.05))
    assert info.value.error_code == "provider_timeout"

    def refused(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {KEY} {http_request.url}", request=http_request)

    instance = GoogleTranslateV2(
        api_key=KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(refused)),
    )
    with pytest.raises(NetworkError) as info:
        await instance.retranslate_window(request())
    assert "ConnectError" in str(info.value) and KEY not in str(info.value)
    assert info.value.error_code == "network_error"


# ------------------------------------------------------- privacy + creds


async def test_privacy_gate_blocks_upload_before_any_request() -> None:
    instance, recorder = backend(allowed=False)
    with pytest.raises(PolicyDeniedError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "privacy_policy_denied"
    with pytest.raises(PolicyDeniedError):
        await anext(instance.translate_incremental(request()))
    assert recorder.requests == []
    await instance.close()


async def test_missing_key_raises_authentication_error_without_request() -> None:
    for key in ("", "   ", None):
        instance, recorder = backend(key=key)
        assert instance.credentials_present() is False
        with pytest.raises(AuthenticationError) as info:
            await instance.retranslate_window(request())
        assert "Google Cloud Translation" in str(info.value)
        assert recorder.requests == []
        assert API_KEY_HEADER not in instance.headers()


async def test_registry_factory_builds_google_adapter() -> None:
    from echolingo.backends.registry import find
    from echolingo.config.loader import load_config

    config = load_config()
    config.privacy.transcript_upload_allowed = True
    spec = find("translation", "google_translate")
    assert spec is not None
    built = spec.factory(config, {"GOOGLE_TRANSLATE_API_KEY": KEY})
    try:
        assert isinstance(built, GoogleTranslateV2)
        assert built.api_key == KEY
        assert built.model == spec.model_for(config) == "nmt"
        assert built.transcript_upload_allowed is True
        assert spec.streaming_partials is False
        assert spec.transcript_upload_required is True
    finally:
        await built.close()
