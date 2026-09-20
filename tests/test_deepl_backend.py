"""DeepL API v2 adapter tests (httpx.MockTransport, no live network)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from echolingo.backends.translation.deepl import (
    DeepLTranslation,
    source_lang_code,
    target_lang_code,
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

FREE_KEY = "0123abcd-4567-89ef-0123-456789abcdef:fx"
PRO_KEY = "0123abcd-4567-89ef-0123-456789abcdef"


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
        200, json={"translations": [{"detected_source_language": "EN", "text": text, **extra}]}
    )


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


def backend(*responses, key=FREE_KEY, allowed=True, **kwargs):
    recorder = Recorder(responses or [ok()])
    instance = DeepLTranslation(
        api_key=key,
        transcript_upload_allowed=allowed,
        client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        **kwargs,
    )
    return instance, recorder


# ----------------------------------------------------------------- hosts


def test_free_and_pro_tiers_select_host() -> None:
    free, _ = backend(key=PRO_KEY, tier="free")
    pro, _ = backend(key=PRO_KEY, tier="pro")
    assert free.translate_url() == "https://api-free.deepl.com/v2/translate"
    assert pro.translate_url() == "https://api.deepl.com/v2/translate"
    assert free.describe_endpoint()["free_key"] is False
    upper, _ = backend(key=PRO_KEY, tier="PRO")
    assert upper.tier == "pro" and upper.host == "https://api.deepl.com"
    with pytest.raises(ValueError):
        DeepLTranslation(api_key=PRO_KEY, tier="platinum")


def test_fx_key_forces_free_host_regardless_of_tier() -> None:
    forced, _ = backend(key=FREE_KEY, tier="pro")
    assert forced.uses_free_host is True
    assert forced.translate_url() == "https://api-free.deepl.com/v2/translate"
    assert forced.describe_endpoint() == {
        "host": "https://api-free.deepl.com",
        "tier": "pro",
        "free_key": True,
        "model_type": "latency_optimized",
    }


async def test_request_uses_pro_host_for_pro_key_and_pro_tier() -> None:
    instance, recorder = backend(key=PRO_KEY, tier="pro")
    await instance.retranslate_window(request())
    assert str(recorder.last.url) == "https://api.deepl.com/v2/translate"


# --------------------------------------------------------- headers + body


async def test_headers_carry_key_only_in_authorization_header() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request())
    sent = recorder.last
    assert sent.method == "POST"
    assert sent.headers["Authorization"] == f"DeepL-Auth-Key {FREE_KEY}"
    assert sent.headers["Content-Type"] == "application/json"
    assert sent.headers["User-Agent"] == "EchoLingo"
    assert FREE_KEY not in str(sent.url) and "auth_key" not in str(sent.url)
    assert FREE_KEY not in sent.content.decode("utf-8")
    await instance.close()


async def test_body_matches_deepl_v2_contract() -> None:
    instance, recorder = backend(model_type="quality_optimized")
    await instance.retranslate_window(request())
    assert recorder.last_body() == {
        "text": ["Welcome to the lecture."],
        "target_lang": "ZH-HANS",
        "source_lang": "EN",
        "model_type": "quality_optimized",
        "preserve_formatting": True,
        "split_sentences": "nonewlines",
    }
    assert instance.model == "deepl-quality_optimized"
    assert instance.descriptor.provider == "deepl"
    assert instance.descriptor.model == "deepl-quality_optimized"
    assert instance.descriptor.locality is BackendLocality.CLOUD
    assert instance.descriptor.transcript_upload_required is True
    assert instance.streaming_partials is False


async def test_context_uses_previous_source_spans_only() -> None:
    instance, recorder = backend()
    context = (
        TranslationContextSegment("Older sentence that is dropped.", "更早的句子。"),
        TranslationContextSegment("First we cover vectors.", "首先我们讲向量。"),
        TranslationContextSegment("  ", "空的"),
        TranslationContextSegment("Then matrices.", "然后是矩阵。"),
    )
    await instance.retranslate_window(request(source_text="Now determinants.", context=context))
    body = recorder.last_body()
    assert body["context"] == "First we cover vectors.\nThen matrices."
    assert body["text"] == ["Now determinants."]
    raw = recorder.last.content.decode("utf-8")
    for target in ("更早的句子。", "首先我们讲向量。", "然后是矩阵。", "空的"):
        assert target not in raw
    assert "Older sentence" not in raw


async def test_context_omitted_when_empty_and_bounded_by_chars() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request(context=()))
    assert "context" not in recorder.last_body()
    await instance.retranslate_window(
        request(context=(TranslationContextSegment("   ", "x"),))
    )
    assert "context" not in recorder.last_body()

    small, recorder = backend(context_max_chars=20)
    await small.retranslate_window(
        request(
            context=(
                TranslationContextSegment("a" * 15, ""),
                TranslationContextSegment("b" * 10, ""),
            )
        )
    )
    assert recorder.last_body()["context"] == "b" * 10

    none, recorder = backend(context_spans=0)
    await none.retranslate_window(request(context=(TranslationContextSegment("x", ""),)))
    assert "context" not in recorder.last_body()


def test_language_code_mapping() -> None:
    assert source_lang_code("en") == "EN"
    assert source_lang_code("zh") == "ZH"
    assert source_lang_code("ja") == "JA"
    assert source_lang_code("ko") == "KO"
    assert source_lang_code("en-US") == "EN"
    assert source_lang_code("zh_CN") == "ZH"
    assert source_lang_code("de") == "DE"
    assert source_lang_code("auto") is None
    assert source_lang_code("") is None
    assert source_lang_code(None) is None
    assert target_lang_code("en") == "EN-US"
    assert target_lang_code("zh") == "ZH-HANS"
    assert target_lang_code("ja") == "JA"
    assert target_lang_code("ko") == "KO"
    assert target_lang_code("en-GB") == "EN-GB"
    assert target_lang_code("zh-TW") == "ZH-HANT"
    assert target_lang_code("zh_Hant") == "ZH-HANT"
    assert target_lang_code("de") == "DE"
    assert target_lang_code("pt") == "PT-BR"


async def test_auto_source_omits_source_lang() -> None:
    instance, recorder = backend()
    await instance.retranslate_window(request(source_lang="auto", target_lang="en"))
    body = recorder.last_body()
    assert "source_lang" not in body
    assert body["target_lang"] == "EN-US"
    assert instance.last_detected_source_language == "EN"


async def test_target_language_variants_for_each_product_language() -> None:
    instance, recorder = backend()
    for code, expected in (("en", "EN-US"), ("zh", "ZH-HANS"), ("ja", "JA"), ("ko", "KO")):
        await instance.retranslate_window(request(source_lang="auto", target_lang=code))
        assert recorder.last_body()["target_lang"] == expected


# ---------------------------------------------------------------- events


async def test_translate_incremental_yields_single_final_with_model() -> None:
    instance, _ = backend(model_type="latency_optimized")
    events = [event async for event in instance.translate_incremental(request())]
    assert len(events) == 1
    event = events[0]
    assert event.kind == TranslationKind.FINAL
    assert event.text == "欢迎来到讲座。"
    assert event.committed_text == "欢迎来到讲座。" and event.editable_text == ""
    assert event.provider == "deepl"
    assert event.model == "deepl-latency_optimized"
    assert event.locality is BackendLocality.CLOUD
    assert event.request_id == "r" and event.source_revision_id == 3
    assert event.source_committed is True
    assert event.truncated is False and event.finish_reason == "stop"
    assert event.total_latency_ms is not None and event.first_delta_latency_ms is not None
    await instance.close()


async def test_model_name_reports_model_type_used_when_provider_reports_it() -> None:
    instance, _ = backend(
        ok(model_type_used="quality_optimized"), model_type="prefer_quality_optimized"
    )
    assert instance.model == "deepl-prefer_quality_optimized"
    event = await instance.retranslate_window(request())
    assert event.model == "deepl-quality_optimized"


async def test_empty_source_text_is_not_sent() -> None:
    instance, recorder = backend()
    event = await instance.retranslate_window(request(source_text="   "))
    assert event.text == "" and event.kind == TranslationKind.FINAL
    assert recorder.requests == []


async def test_glossary_terms_are_ignored_and_never_logged(caplog) -> None:
    instance, recorder = backend()
    await instance.set_glossary((GlossaryTerm("determinant", "行列式"),))
    terms = (GlossaryTerm("eigenvalue", "特征值"),)
    with caplog.at_level(logging.WARNING, logger="echolingo.backends.translation.deepl"):
        await instance.retranslate_window(request(terms=terms))
        await instance.retranslate_window(request(terms=terms))
    body = recorder.last_body()
    assert "glossary" not in body and "glossary_id" not in body
    raw = recorder.last.content.decode("utf-8")
    assert "eigenvalue" not in raw and "行列式" not in raw
    warnings = [record for record in caplog.records if "glossary" in record.getMessage()]
    assert len(warnings) == 1
    assert "eigenvalue" not in warnings[0].getMessage()
    assert "特征值" not in warnings[0].getMessage()


# ---------------------------------------------------------------- errors


async def test_403_maps_to_authentication_error_without_key() -> None:
    instance, _ = backend(
        httpx.Response(403, json={"message": "Wrong endpoint. Use https://api.deepl.com"}),
        key=PRO_KEY,
        tier="free",
    )
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    message = str(info.value)
    assert message.startswith(
        "DeepL rejected the authentication key (HTTP 403). "
        "Free keys end with ':fx' and must use the API Free plan."
    )
    assert "Wrong endpoint" in message
    assert PRO_KEY not in message
    assert info.value.error_code == "authentication_failed"


async def test_403_with_empty_body_keeps_required_message() -> None:
    instance, _ = backend(httpx.Response(403))
    with pytest.raises(AuthenticationError) as info:
        await instance.retranslate_window(request())
    assert str(info.value) == (
        "DeepL rejected the authentication key (HTTP 403). "
        "Free keys end with ':fx' and must use the API Free plan."
    )


async def test_456_and_429_map_to_rate_limit() -> None:
    instance, _ = backend(httpx.Response(456, json={"message": "Quota Exceeded"}))
    with pytest.raises(RateLimitError) as info:
        await instance.retranslate_window(request())
    assert "quota" in str(info.value).lower() and "456" in str(info.value)
    assert info.value.error_code == "rate_limited" and info.value.recoverable is True
    assert FREE_KEY not in str(info.value)

    instance, _ = backend(httpx.Response(429, text="Too many requests"))
    with pytest.raises(RateLimitError) as info:
        await instance.retranslate_window(request())
    assert "429" in str(info.value)

    instance, _ = backend(httpx.Response(529, json={"message": "Too many requests"}))
    with pytest.raises(RateLimitError):
        await instance.retranslate_window(request())


async def test_400_unsupported_language_and_generic_bad_request() -> None:
    instance, _ = backend(
        httpx.Response(400, json={"message": "Value for 'target_lang' not supported."})
    )
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(target_lang="xx", context=(TranslationContextSegment("c", ""),)))
    assert info.value.error_code == "bad_request"
    assert "language" in str(info.value).lower()
    assert "target_lang" in str(info.value)
    # Dropping context cannot fix an unsupported language.
    assert info.value.retry_without_context is False

    instance, _ = backend(httpx.Response(400, json={"message": "Bad request. Reason: context too long."}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(context=(TranslationContextSegment("c", ""),)))
    assert info.value.error_code == "bad_request"
    assert "language" not in str(info.value).lower()
    assert info.value.retry_without_context is True

    instance, _ = backend(httpx.Response(400))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "bad_request" and info.value.retry_without_context is False


async def test_5xx_and_other_statuses_map_to_http_codes() -> None:
    instance, _ = backend(httpx.Response(503, json={"message": "Service unavailable"}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "http_503"
    assert "503" in str(info.value) and FREE_KEY not in str(info.value)

    instance, _ = backend(httpx.Response(413))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "request_too_large"

    instance, _ = backend(httpx.Response(404))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "not_found"


async def test_key_echoed_by_provider_is_redacted() -> None:
    instance, _ = backend(httpx.Response(500, json={"message": f"key {FREE_KEY} broke"}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert FREE_KEY not in str(info.value) and "[redacted]" in str(info.value)


async def test_malformed_and_empty_responses() -> None:
    instance, _ = backend(httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "invalid_response"

    instance, _ = backend(httpx.Response(200, json={"translations": []}))
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request())
    assert info.value.error_code == "empty_response"


async def test_timeout_and_transport_errors_are_mapped() -> None:
    def timeout(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=http_request)

    instance = DeepLTranslation(
        api_key=FREE_KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(timeout)),
    )
    with pytest.raises(TranslationRequestError) as info:
        await instance.retranslate_window(request(timeout_s=0.05))
    assert info.value.error_code == "provider_timeout"

    def refused(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {FREE_KEY}", request=http_request)

    instance = DeepLTranslation(
        api_key=FREE_KEY,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(refused)),
    )
    with pytest.raises(NetworkError) as info:
        await instance.retranslate_window(request())
    assert "ConnectError" in str(info.value) and FREE_KEY not in str(info.value)
    assert info.value.error_code == "network_error"


# ------------------------------------------------------- privacy + creds


async def test_privacy_gate_blocks_upload_before_any_request() -> None:
    instance, recorder = backend(allowed=False)
    with pytest.raises(PolicyDeniedError):
        await instance.retranslate_window(request())
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
        assert "DeepL" in str(info.value)
        assert recorder.requests == []
        assert "Authorization" not in instance.headers()


async def test_registry_factory_builds_deepl_adapter(monkeypatch) -> None:
    from echolingo.backends.registry import find
    from echolingo.config.loader import load_config

    config = load_config()
    config.privacy.transcript_upload_allowed = True
    spec = find("translation", "deepl")
    assert spec is not None
    env = {"DEEPL_API_KEY": FREE_KEY, "ECHOLINGO_DEEPL_TIER": "PRO"}
    built = spec.factory(config, env)
    try:
        assert isinstance(built, DeepLTranslation)
        assert built.tier == "pro"
        # A ':fx' key still goes to the Free host.
        assert built.host == "https://api-free.deepl.com"
        assert built.model == spec.model_for(config)
        assert built.transcript_upload_allowed is True
        assert spec.streaming_partials is False
    finally:
        await built.close()
