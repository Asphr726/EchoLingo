"""Azure AI Translator (Text v3.0) adapter tests (httpx.MockTransport, no live network)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import httpx
import pytest

from echolingo.backends.translation.azure_translator import (
    DEFAULT_ENDPOINT,
    AzureTranslator,
    normalise_endpoint,
    normalise_region,
    source_lang_code,
    target_lang_code,
)
from echolingo.errors import AuthenticationError, NetworkError, PolicyDeniedError, RateLimitError
from echolingo.models import BackendLocality, GlossaryTerm, TranslationKind, TranslationRequest
from echolingo.translation.policy import TranslationRequestError

KEY = "0123456789abcdef0123456789abcdef"


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


def ok(text: str = "欢迎来到讲座。", to: str = "zh-Hans", detected: str | None = "en") -> httpx.Response:
    item: dict = {"translations": [{"text": text, "to": to}]}
    if detected:
        item["detectedLanguage"] = {"language": detected, "score": 1.0}
    return httpx.Response(200, json=[item])


def azure_error(status: int, code: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


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

    def last_body(self):
        return json.loads(self.last.content.decode("utf-8"))


def backend(*responses, key=KEY, allowed=True, **kwargs):
    recorder = Recorder(responses or [ok()])
    instance = AzureTranslator(
        api_key=key,
        transcript_upload_allowed=allowed,
        client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        **kwargs,
    )
    return instance, recorder


# ----------------------------------------------------------------- codes


def test_language_codes() -> None:
    assert target_lang_code("zh") == "zh-Hans"
    assert target_lang_code("zh-CN") == "zh-Hans"
    assert target_lang_code("zh_Hans") == "zh-Hans"
    assert target_lang_code("zh-TW") == "zh-Hant"
    assert target_lang_code("zh-Hant") == "zh-Hant"
    assert target_lang_code("en") == "en"
    assert target_lang_code("en-US") == "en"
    assert target_lang_code("ja") == "ja"
    assert target_lang_code("ko") == "ko"
    assert target_lang_code("fr") == "fr"
    assert source_lang_code("en") == "en"
    assert source_lang_code("zh") == "zh-Hans"
    assert source_lang_code("ja") == "ja"
    assert source_lang_code("ko") == "ko"
    assert source_lang_code("auto") is None
    assert source_lang_code("") is None
    assert source_lang_code(None) is None
    assert source_lang_code("und") is None


def test_endpoint_and_region_normalisation() -> None:
    assert normalise_endpoint(None) == DEFAULT_ENDPOINT
    assert normalise_endpoint("  ") == DEFAULT_ENDPOINT
    assert normalise_endpoint("https://api.cognitive.microsofttranslator.com/") == DEFAULT_ENDPOINT
    assert normalise_endpoint("https://example.invalid/translator/text/v3.0//") == "https://example.invalid/translator/text/v3.0"
    assert normalise_region(None) == ""
    assert normalise_region(" East Asia ") == "eastasia"
    assert normalise_region("eastasia") == "eastasia"


# --------------------------------------------------------------- request


async def test_request_url_query_headers_and_body() -> None:
    instance, recorder = backend(region="eastasia")
    assert instance.descriptor.provider == "azure_translator"
    assert instance.descriptor.locality is BackendLocality.CLOUD
    assert instance.descriptor.transcript_upload_required is True
    assert instance.streaming_partials is False
    assert instance.model == "translator-v3"
    assert instance.display_name == "Azure AI Translator"

    event = await instance.retranslate_window(request())
    await instance.close()

    sent = recorder.last
    assert sent.method == "POST"
    assert sent.url.scheme == "https"
    assert sent.url.host == "api.cognitive.microsofttranslator.com"
    assert sent.url.path == "/translate"
    params = dict(sent.url.params)
    assert params == {"api-version": "3.0", "to": "zh-Hans", "from": "en", "textType": "plain"}
    # The key never travels in the URL.
    assert KEY not in str(sent.url)

    assert sent.headers["Ocp-Apim-Subscription-Key"] == KEY
    assert sent.headers["Ocp-Apim-Subscription-Region"] == "eastasia"
    assert sent.headers["Content-Type"] == "application/json"
    assert sent.headers["User-Agent"] == "EchoLingo"
    trace = uuid.UUID(sent.headers["X-ClientTraceId"])
    assert trace.version == 4
    assert instance.last_trace_id == str(trace)

    assert recorder.last_body() == [{"Text": "Welcome to the lecture."}]

    assert event.kind == TranslationKind.FINAL
    assert event.text == "欢迎来到讲座。"
    assert event.model == "translator-v3"
    assert event.provider == "azure_translator"
    assert instance.last_detected_source_language == "en"


async def test_auto_source_omits_from_and_region_header_omitted_when_empty() -> None:
    instance, recorder = backend(region="")
    await instance.retranslate_window(request(source_lang="auto"))
    await instance.close()
    params = dict(recorder.last.url.params)
    assert "from" not in params
    assert params["to"] == "zh-Hans" and params["api-version"] == "3.0" and params["textType"] == "plain"
    assert "Ocp-Apim-Subscription-Region" not in recorder.last.headers
    assert recorder.last.headers["Ocp-Apim-Subscription-Key"] == KEY

    # Whitespace-only region behaves like empty (global resource).
    instance, recorder = backend(region="   ")
    await instance.retranslate_window(request())
    await instance.close()
    assert "Ocp-Apim-Subscription-Region" not in recorder.last.headers


async def test_language_pairs_in_query() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request(source_lang="ja", target_lang="ko"))
    params = dict(recorder.last.url.params)
    assert params["from"] == "ja" and params["to"] == "ko"
    await instance.retranslate_window(request(source_lang="zh-TW", target_lang="en"))
    params = dict(recorder.last.url.params)
    assert params["from"] == "zh-Hant" and params["to"] == "en"
    await instance.close()


async def test_translate_incremental_yields_single_final() -> None:
    instance, recorder = backend()
    events = [event async for event in instance.translate_incremental(request())]
    await instance.close()
    assert len(events) == 1
    assert events[0].kind == TranslationKind.FINAL
    assert events[0].committed_text == "欢迎来到讲座。"
    assert events[0].editable_text == ""
    assert events[0].source_committed is True
    assert events[0].source_revision_id == 3
    assert events[0].total_latency_ms is not None
    assert len(recorder.requests) == 1


async def test_each_request_gets_a_fresh_trace_id() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request())
    first = recorder.last.headers["X-ClientTraceId"]
    await instance.retranslate_window(request())
    second = recorder.last.headers["X-ClientTraceId"]
    await instance.close()
    assert first != second


async def test_empty_source_text_sends_nothing() -> None:
    instance, recorder = backend()
    event = await instance.retranslate_window(request(source_text="   "))
    await instance.close()
    assert event.text == "" and event.kind == TranslationKind.FINAL
    assert recorder.requests == []


async def test_custom_endpoint_strips_trailing_slash_and_keeps_path() -> None:
    instance, recorder = backend(
        endpoint="https://my-translator.cognitiveservices.azure.com/translator/text/v3.0/",
        region="westeurope",
    )
    await instance.retranslate_window(request())
    await instance.close()
    assert instance.endpoint == "https://my-translator.cognitiveservices.azure.com/translator/text/v3.0"
    assert instance.translate_url() == "https://my-translator.cognitiveservices.azure.com/translator/text/v3.0/translate"
    assert recorder.last.url.host == "my-translator.cognitiveservices.azure.com"
    assert recorder.last.url.path == "/translator/text/v3.0/translate"
    assert dict(recorder.last.url.params)["api-version"] == "3.0"
    described = instance.describe_endpoint()
    assert described["custom_endpoint"] is True
    assert described["host"] == "https://my-translator.cognitiveservices.azure.com"
    assert described["region"] == "westeurope"
    assert KEY not in json.dumps(described)

    default = AzureTranslator(api_key=KEY, endpoint="")
    assert default.endpoint == DEFAULT_ENDPOINT
    assert default.describe_endpoint()["custom_endpoint"] is False
    await default.close()


# ---------------------------------------------------------------- errors


async def test_401_maps_to_authentication_error_without_leaking_key() -> None:
    instance, _ = backend(
        azure_error(401, 401000, "The request is not authorized because credentials are missing or invalid.")
    )
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    message = str(info.value)
    assert message.startswith("Azure Translator rejected the key (HTTP 401")
    assert "Azure 401000" in message
    assert "set the resource region (e.g. eastasia) for regional resources" in message
    assert KEY not in message
    assert info.value.error_code == "authentication_failed"


async def test_401_with_speech_key_code_explains_resource_type() -> None:
    instance, _ = backend(azure_error(401, 401015, "The credentials provided are for the Speech API."))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert "Speech resource key" in str(info.value)
    assert "Azure 401015" in str(info.value)


async def test_401_without_json_body_still_maps() -> None:
    instance, _ = backend(httpx.Response(401, text="Access denied"))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert "HTTP 401" in str(info.value) and KEY not in str(info.value)


async def test_403_quota_and_forbidden_map_to_authentication_error() -> None:
    instance, _ = backend(
        azure_error(403, 403001, "The operation is not allowed because the subscription has exceeded its free quota.")
    )
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert "quota" in str(info.value).lower()
    assert "HTTP 403, Azure 403001" in str(info.value)
    assert KEY not in str(info.value)

    instance, _ = backend(azure_error(403, 403000, "The operation is not allowed."))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert "refused the operation" in str(info.value)
    assert "HTTP 403, Azure 403000" in str(info.value)


async def test_429_maps_to_rate_limit() -> None:
    instance, _ = backend(azure_error(429, 429001, "The server rejected the request because the client has exceeded request limits."))
    with pytest.raises(RateLimitError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "rate_limited" and info.value.recoverable is True
    assert "HTTP 429" in str(info.value) and KEY not in str(info.value)


async def test_400_includes_azure_error_code_number() -> None:
    instance, _ = backend(azure_error(400, 400036, "The target language is not valid."))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(target_lang="xx"))
    await instance.close()
    assert info.value.error_code == "bad_request"
    assert "400036" in str(info.value)
    assert "language pair" in str(info.value)
    assert "The target language is not valid." in str(info.value)
    assert KEY not in str(info.value)

    instance, _ = backend(azure_error(400, 400074, "The body of the request is not valid JSON."))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "bad_request"
    assert "Azure 400074" in str(info.value)
    assert info.value.retry_without_context is False

    instance, _ = backend(azure_error(400, 400050, "The input text is too long."))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "request_too_large"
    assert "400050" in str(info.value)


async def test_400_with_string_code_and_without_body() -> None:
    instance, _ = backend(httpx.Response(400, json={"error": {"code": "400019", "message": "One of the specified languages is not supported."}}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "bad_request" and "Azure 400019" in str(info.value)

    instance, _ = backend(httpx.Response(400, text="<html>bad</html>"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "bad_request" and "HTTP 400" in str(info.value)


async def test_other_statuses_map_to_codes() -> None:
    instance, _ = backend(azure_error(503, 503000, "Service is temporarily unavailable."))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "http_503"
    assert "api.cognitive.microsofttranslator.com" in str(info.value)
    assert KEY not in str(info.value)

    instance, _ = backend(httpx.Response(500))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "http_500"

    instance, _ = backend(httpx.Response(404, text="Resource not found"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "not_found"

    instance, _ = backend(azure_error(408, 408001, "The translation system requested is being prepared."))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "provider_timeout"


async def test_key_echoed_by_provider_is_redacted() -> None:
    instance, _ = backend(azure_error(401, 401000, f"Key {KEY} is not valid"))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert KEY not in str(info.value)
    assert "[redacted]" in str(info.value)


async def test_invalid_and_empty_responses() -> None:
    instance, _ = backend(httpx.Response(200, text="not json"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "invalid_response"

    for body in ([], [{}], [{"translations": []}], {"translations": [{"text": "x"}]}):
        instance, _ = backend(httpx.Response(200, json=body))
        with pytest.raises(TranslationRequestError) as info:
            await instance.retranslate_window(request())
        await instance.close()
        assert info.value.error_code == "empty_response"


async def test_network_and_timeout_errors_do_not_leak_key() -> None:
    def broken(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"boom {http_request.url} {http_request.headers['Ocp-Apim-Subscription-Key']}")

    instance = AzureTranslator(
        api_key=KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(broken)),
    )
    with pytest.raises(NetworkError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert KEY not in str(info.value)
    assert "ConnectError" in str(info.value)
    assert info.value.error_code == "network_error"

    async def slow(http_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.5)
        return ok()

    instance = AzureTranslator(
        api_key=KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(slow)),
    )
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(timeout_s=0.05))
    await instance.close()
    assert info.value.error_code == "provider_timeout"
    assert KEY not in str(info.value)

    def read_timeout(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"read timed out {KEY}", request=http_request)

    instance = AzureTranslator(
        api_key=KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(read_timeout)),
    )
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    await instance.close()
    assert info.value.error_code == "provider_timeout"
    assert KEY not in str(info.value)


# --------------------------------------------------------------- gating


async def test_privacy_gate_blocks_before_any_request() -> None:
    instance, recorder = backend(allowed=False)
    with pytest.raises(PolicyDeniedError) as info:
        await anext(instance.translate_incremental(request()))
    await instance.close()
    assert info.value.error_code == "privacy_policy_denied"
    assert recorder.requests == []


async def test_missing_key_is_an_authentication_error_before_any_request() -> None:
    for key in (None, "", "   "):
        instance, recorder = backend(key=key)
        assert instance.credentials_present() is False
        with pytest.raises(AuthenticationError):
            await instance.retranslate_window(request())
        await instance.close()
        assert recorder.requests == []


async def test_glossary_is_ignored_with_one_warning_without_terms(caplog) -> None:
    instance, recorder = backend()
    await instance.set_glossary((GlossaryTerm("GPU", "图形处理器"),))
    with caplog.at_level(logging.WARNING, logger="echolingo.backends.translation.azure_translator"):
        await instance.retranslate_window(request(terms=(GlossaryTerm("VAD", "语音活动检测"),)))
        await instance.retranslate_window(request())
    await instance.close()
    warnings = [record for record in caplog.records if "glossary" in record.getMessage()]
    assert len(warnings) == 1
    assert "VAD" not in warnings[0].getMessage() and "GPU" not in warnings[0].getMessage()
    assert KEY not in caplog.text
    # Glossary terms never reach the wire: body stays the plain text list.
    assert recorder.last_body() == [{"Text": "Welcome to the lecture."}]


async def test_registry_factory_builds_adapter_from_config_and_env(monkeypatch) -> None:
    from echolingo.backends import registry
    from echolingo.config.schema import AppConfig

    config = AppConfig()
    config.privacy.transcript_upload_allowed = True
    spec = registry.get("translation", "azure_translator")
    instance = spec.factory(config, {"AZURE_TRANSLATOR_KEY": KEY, "ECHOLINGO_AZURE_TRANSLATOR_REGION": "eastasia"})
    try:
        assert isinstance(instance, AzureTranslator)
        assert instance.api_key == KEY
        assert instance.region == "eastasia"
        assert instance.endpoint == DEFAULT_ENDPOINT
        assert instance.transcript_upload_allowed is True
        assert spec.streaming_partials is False
        assert spec.model_for(config) == instance.model == "translator-v3"
    finally:
        await instance.close()


async def test_registry_probe_translates_fixed_sentence_only(monkeypatch) -> None:
    from dataclasses import replace

    from echolingo.backends import registry
    from echolingo.config.schema import AppConfig

    instance, recorder = backend(allowed=False, region="eastasia")
    spec = registry.get("translation", "azure_translator")
    monkeypatch.setitem(
        registry.PROVIDERS,
        ("translation", "azure_translator"),
        replace(spec, factory=lambda config, env: instance),
    )
    result = await registry.PROVIDERS[("translation", "azure_translator")].probe(
        registry.PROVIDERS[("translation", "azure_translator")], AppConfig(), {}
    )
    assert result["provider"] == "azure_translator"
    assert result["status"] == "connected"
    assert result["model"] == "translator-v3"
    assert isinstance(result["latency_ms"], float)
    assert len(recorder.requests) == 1
    assert recorder.last_body() == [{"Text": "Welcome to the lecture."}]
    assert dict(recorder.last.url.params) == {"api-version": "3.0", "to": "zh-Hans", "from": "en", "textType": "plain"}
    assert KEY not in json.dumps(result)
