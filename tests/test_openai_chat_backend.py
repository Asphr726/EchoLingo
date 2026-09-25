"""OpenAI-compatible chat translation adapter (openai/deepseek/gemini/groq/openrouter/siliconflow/custom)."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from echolingo.backends.registry import CHAT_PRESETS
from echolingo.backends.translation.openai_chat import (
    DISPLAY_NAMES,
    OPENROUTER_HEADERS,
    STREAM_USAGE_PROVIDERS,
    OpenAiChatTranslation,
)
from echolingo.errors import (
    AuthenticationError,
    ConfigurationError,
    PolicyDeniedError,
    RateLimitError,
)
from echolingo.models import (
    BackendLocality,
    GlossaryTerm,
    TranslationContextSegment,
    TranslationKind,
    TranslationRequest,
)
from echolingo.translation.policy import TranslationRequestError

KEY = "sk-test-secret-key-0123456789abcdef"
PRESETS = tuple(CHAT_PRESETS)
KEYED_PRESETS = tuple(preset for preset in PRESETS if preset != "custom_chat")


def request(**overrides) -> TranslationRequest:
    values = dict(
        request_id="r",
        source_revision_id=1,
        source_text="hello world",
        source_lang="en",
        target_lang="zh",
        source_committed=True,
    )
    values.update(overrides)
    return TranslationRequest(**values)


def contextual_request(**overrides) -> TranslationRequest:
    return request(
        context=(
            TranslationContextSegment("first sentence", "第一句"),
            TranslationContextSegment("second sentence", "第二句"),
            TranslationContextSegment("third sentence", "第三句"),
        ),
        terms=(GlossaryTerm("EchoLingo", "回声语"), GlossaryTerm("gradient descent", "梯度下降")),
        **overrides,
    )


def sse(*contents: str, finish: str | None = None, usage: dict | None = None) -> str:
    lines = [f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}' for c in contents]
    if finish:
        lines.append(f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]})}')
    if usage:
        lines.append(f'data: {json.dumps({"choices": [], "usage": usage})}')
    lines.append("data: [DONE]")
    return "\n\n".join(lines)


def backend(preset: str, handler, *, api_key: str | None = KEY, base_url: str | None = None, **overrides):
    _group, preset_url, model, _env = CHAT_PRESETS[preset]
    values = dict(
        provider=preset,
        base_url=preset_url if base_url is None else base_url,
        model=model or "qwen2.5:7b",
        api_key=api_key,
        transcript_upload_allowed=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    values.update(overrides)
    return OpenAiChatTranslation(**values)


def system_and_user(payload: dict) -> tuple[str, str]:
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    return messages[0]["content"], messages[1]["content"]


# ----------------------------------------------------------------- presets


@pytest.mark.parametrize("preset", KEYED_PRESETS)
async def test_each_preset_posts_to_its_chat_completions_url(preset: str) -> None:
    seen: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen["url"] = str(http_request.url)
        seen["headers"] = dict(http_request.headers)
        seen["payload"] = json.loads(http_request.content)
        return httpx.Response(200, text=sse("你好", finish="stop"))

    _group, preset_url, model, _env = CHAT_PRESETS[preset]
    adapter = backend(preset, handler)
    events = [event async for event in adapter.translate_incremental(request())]
    await adapter.close()

    assert seen["url"] == f"{preset_url}/chat/completions"
    assert seen["headers"]["authorization"] == f"Bearer {KEY}"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["payload"]["model"] == model
    assert seen["payload"]["stream"] is True
    assert events[-1].kind == TranslationKind.FINAL and events[-1].text == "你好"
    assert events[-1].provider == preset
    assert events[-1].locality is BackendLocality.CLOUD
    assert adapter.descriptor.provider == preset and adapter.descriptor.model == model
    assert adapter.descriptor.transcript_upload_required is True
    assert adapter.display_name == DISPLAY_NAMES[preset]
    # Attribution headers belong to OpenRouter only.
    if preset == "openrouter_chat":
        for name, value in OPENROUTER_HEADERS.items():
            assert seen["headers"][name.lower()] == value
    else:
        assert "http-referer" not in seen["headers"] and "x-title" not in seen["headers"]
    # Gemini's OpenAI endpoint may reject stream_options; the rest report usage.
    if preset in STREAM_USAGE_PROVIDERS:
        assert seen["payload"]["stream_options"] == {"include_usage": True}
    else:
        assert "stream_options" not in seen["payload"]


def test_gemini_and_custom_omit_stream_options_and_window_requests_never_carry_it() -> None:
    assert "gemini_chat" not in STREAM_USAGE_PROVIDERS
    assert "custom_chat" not in STREAM_USAGE_PROVIDERS
    gemini = OpenAiChatTranslation(provider="gemini_chat", base_url=CHAT_PRESETS["gemini_chat"][1], model="gemini-2.0-flash", api_key=KEY)
    assert "stream_options" not in gemini.build_payload(request(), model="gemini-2.0-flash", stream=True)
    openai = OpenAiChatTranslation(provider="openai_chat", base_url=CHAT_PRESETS["openai_chat"][1], model="gpt-4o-mini", api_key=KEY)
    assert "stream_options" not in openai.build_payload(request(), model="gpt-4o-mini", stream=False)
    assert openai.build_payload(request(), model="gpt-4o-mini", stream=True)["stream_options"] == {"include_usage": True}


def test_deepseek_requests_turn_thinking_off() -> None:
    deepseek = OpenAiChatTranslation(provider="deepseek_chat", base_url=CHAT_PRESETS["deepseek_chat"][1], model="deepseek-flash", api_key=KEY)
    for stream in (True, False):
        assert deepseek.build_payload(request(), model="deepseek-flash", stream=stream)["thinking"] == {"type": "disabled"}
    for preset in set(CHAT_PRESETS) - {"deepseek_chat"}:
        other = OpenAiChatTranslation(provider=preset, base_url="http://127.0.0.1:11434/v1", model="m", api_key=KEY)
        assert "thinking" not in other.build_payload(request(), model="m", stream=True)


def test_display_names_cover_every_registered_preset() -> None:
    assert set(DISPLAY_NAMES) == set(CHAT_PRESETS)
    assert DISPLAY_NAMES["custom_chat"] == "Custom endpoint"
    assert DISPLAY_NAMES["gemini_chat"] == "Google Gemini"


# ------------------------------------------------------------------ custom


async def test_custom_endpoint_runs_without_a_key_and_sends_no_authorization() -> None:
    seen: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen["url"] = str(http_request.url)
        seen["headers"] = dict(http_request.headers)
        seen["payload"] = json.loads(http_request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "你好，世界"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 4}})

    adapter = backend("custom_chat", handler, api_key="", base_url="http://127.0.0.1:11434/v1/", model="qwen2.5:7b")
    assert adapter.credentials_present() is True
    event = await adapter.retranslate_window(request())
    await adapter.close()

    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert "authorization" not in seen["headers"]
    assert "http-referer" not in seen["headers"]
    assert "stream_options" not in seen["payload"]
    assert seen["payload"]["model"] == "qwen2.5:7b"
    assert event.kind == TranslationKind.FINAL and event.text == "你好，世界"
    assert event.prompt_tokens == 5 and event.completion_tokens == 4
    assert event.provider == "custom_chat"


async def test_custom_endpoint_uses_key_when_given_and_bypasses_proxy_for_loopback(monkeypatch) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.headers["Authorization"] == "Bearer local-token"
        return httpx.Response(200, text=sse("ok", finish="stop"))

    adapter = backend("custom_chat", handler, api_key="local-token", base_url="http://localhost:1234/v1", model="local")
    events = [event async for event in adapter.translate_incremental(request())]
    await adapter.close()
    assert events[-1].text == "ok"

    # Without an injected client a loopback server gets a client that ignores HTTPS_PROXY.
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    owned = OpenAiChatTranslation(provider="custom_chat", base_url="http://127.0.0.1:11434/v1", model="m", transcript_upload_allowed=True)
    assert owned.client.trust_env is False
    assert owned._owns_client is True
    await owned.close()
    remote = OpenAiChatTranslation(provider="openai_chat", base_url="https://api.openai.com/v1", model="m", api_key=KEY)
    assert remote.client.trust_env is True
    await remote.close()


async def test_custom_endpoint_without_base_url_raises_configuration_error() -> None:
    adapter = OpenAiChatTranslation(provider="custom_chat", base_url="", model="m", api_key="", transcript_upload_allowed=True)
    assert adapter.credentials_present() is False
    with pytest.raises(ConfigurationError) as info:
        await adapter.retranslate_window(request())
    assert str(info.value) == "Set the base URL for the custom endpoint"
    with pytest.raises(ConfigurationError):
        await anext(adapter.translate_incremental(request()))
    await adapter.close()


@pytest.mark.parametrize("preset", KEYED_PRESETS)
async def test_keyed_presets_require_an_api_key(preset: str) -> None:
    adapter = backend(preset, lambda http_request: httpx.Response(200), api_key="")
    assert adapter.credentials_present() is False
    with pytest.raises(AuthenticationError) as info:
        await adapter.retranslate_window(request())
    assert DISPLAY_NAMES[preset] in str(info.value)
    await adapter.close()


# ------------------------------------------------------------------ prompt


def test_prompt_carries_glossary_and_source_only_background() -> None:
    adapter = OpenAiChatTranslation(provider="deepseek_chat", base_url="https://api.deepseek.com/v1", model="deepseek-flash", api_key=KEY, background_spans=2)
    payload = adapter.build_payload(contextual_request(), model="deepseek-flash", stream=True)
    system, user = system_and_user(payload)

    assert user == "hello world"
    assert system.startswith("You are a professional simultaneous interpreter for lecture transcripts.")
    assert "Translate from English into Chinese." in system
    assert "Output ONLY the translation" in system
    assert "Keep numbers, names, code identifiers and technical terms accurate." in system
    assert "Glossary:\nEchoLingo => 回声语\ngradient descent => 梯度下降" in system
    # Only the last background_spans SOURCE spans, in order.
    assert "Earlier context (do not translate):\nsecond sentence\nthird sentence" in system
    assert "first sentence" not in system
    # Target-language text from earlier segments is never placed in the prompt.
    for target in ("第一句", "第二句", "第三句"):
        assert target not in json.dumps(payload, ensure_ascii=False)
    # A committed span gets no provisional instruction.
    assert "unfinished sentence" not in system
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == max(24, 3 * len("hello world") + 16)


def test_prompt_uses_backend_glossary_when_request_has_none() -> None:
    adapter = OpenAiChatTranslation(provider="groq_chat", base_url="https://api.groq.com/openai/v1", model="m", api_key=KEY)
    asyncio.run(adapter.set_glossary((GlossaryTerm("transformer", "变换器"),)))
    system, _user = system_and_user(adapter.build_payload(request(), model="m", stream=False))
    assert "Glossary:\ntransformer => 变换器" in system
    # Request terms win over the backend glossary.
    system, _user = system_and_user(adapter.build_payload(request(terms=(GlossaryTerm("a", "b"),)), model="m", stream=False))
    assert "a => b" in system and "transformer" not in system
    # No glossary, no context: no section headers.
    plain = OpenAiChatTranslation(provider="groq_chat", base_url="https://api.groq.com/openai/v1", model="m", api_key=KEY)
    system, _user = system_and_user(plain.build_payload(request(), model="m", stream=False))
    assert "Glossary" not in system and "Earlier context" not in system


def test_prompt_adds_provisional_instruction_for_uncommitted_source() -> None:
    adapter = OpenAiChatTranslation(provider="openai_chat", base_url="https://api.openai.com/v1", model="m", api_key=KEY)
    system, _user = system_and_user(adapter.build_payload(request(source_committed=False), model="m", stream=True))
    assert system.endswith("The text may be an unfinished sentence; translate what is present without completing it.")
    system, _user = system_and_user(adapter.build_payload(request(source_committed=True), model="m", stream=True))
    assert "unfinished sentence" not in system


def test_prompt_language_names_fall_back_to_codes_and_handle_auto() -> None:
    adapter = OpenAiChatTranslation(provider="openai_chat", base_url="https://api.openai.com/v1", model="m", api_key=KEY)
    system, _ = system_and_user(adapter.build_payload(request(source_lang="ja", target_lang="ko"), model="m", stream=False))
    assert "Translate from Japanese into Korean." in system
    system, _ = system_and_user(adapter.build_payload(request(source_lang="xx", target_lang="yy"), model="m", stream=False))
    assert "Translate from xx into yy." in system
    system, _ = system_and_user(adapter.build_payload(request(source_lang="auto", target_lang="zh"), model="m", stream=False))
    assert "into Chinese." in system and "auto" not in system


def test_background_respects_span_and_char_budgets() -> None:
    adapter = OpenAiChatTranslation(provider="openai_chat", base_url="https://api.openai.com/v1", model="m", api_key=KEY, background_spans=0)
    system, _ = system_and_user(adapter.build_payload(contextual_request(), model="m", stream=False))
    assert "Earlier context" not in system
    adapter = OpenAiChatTranslation(provider="openai_chat", base_url="https://api.openai.com/v1", model="m", api_key=KEY, background_spans=3, background_max_chars=20)
    system, _ = system_and_user(adapter.build_payload(contextual_request(), model="m", stream=False))
    assert "Earlier context (do not translate):\nthird sentence" in system
    assert "second sentence" not in system


def test_max_tokens_is_capped_by_max_output_tokens() -> None:
    adapter = OpenAiChatTranslation(provider="openai_chat", base_url="https://api.openai.com/v1", model="m", api_key=KEY, max_output_tokens=100)
    assert adapter.build_payload(request(source_text="x"), model="m", stream=False)["max_tokens"] == 24
    assert adapter.build_payload(request(source_text="x" * 500), model="m", stream=False)["max_tokens"] == 100


# --------------------------------------------------------------- streaming


async def test_streams_partials_and_reports_usage_from_final_chunk() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse("你", "好", "，世界", finish="stop", usage={"prompt_tokens": 40, "completion_tokens": 6}))

    adapter = backend("openai_chat", handler)
    events = [event async for event in adapter.translate_incremental(request())]
    await adapter.close()

    assert [event.kind for event in events] == [TranslationKind.PARTIAL] * 3 + [TranslationKind.FINAL]
    assert [event.text for event in events] == ["你", "你好", "你好，世界", "你好，世界"]
    assert events[0].first_delta_latency_ms is not None
    assert events[-1].prompt_tokens == 40 and events[-1].completion_tokens == 6
    assert events[-1].finish_reason == "stop" and events[-1].truncated is False
    assert events[-1].committed_text == "你好，世界" and events[-1].editable_text == ""
    assert events[-1].model == "gpt-4o-mini" and events[-1].provider == "openai_chat"


async def test_strips_echoed_preface_labels_and_wrapping_quotes() -> None:
    bodies = iter(
        [
            sse("Translation: ", "「你好", "，世界」", finish="stop"),
            sse("Sure, here is the translation: ", '"你好，世界"', finish="stop"),
            sse("**译文：**", "你好，世界", finish="stop"),
        ]
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=next(bodies))

    adapter = backend("siliconflow_chat", handler)
    for _ in range(3):
        events = [event async for event in adapter.translate_incremental(request())]
        assert events[-1].text == "你好，世界"
        assert events[-1].truncated is False
    await adapter.close()


async def test_streaming_keeps_trailing_whitespace_between_deltas() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse("Hello ", "world,", " this is ", "a test.\n", finish="stop"))

    adapter = backend("gemini_chat", handler)
    events = [event async for event in adapter.translate_incremental(request(source_lang="zh", target_lang="en", source_text="你好世界，这是一个测试。"))]
    await adapter.close()
    assert [event.text for event in events[:-1]] == ["Hello ", "Hello world,", "Hello world, this is ", "Hello world, this is a test.\n"]
    assert events[-1].text == "Hello world, this is a test."


async def test_window_retranslation_strips_quotes_and_labels() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert payload["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": 'Translated text: "你好，世界"'}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 5}})

    adapter = backend("deepseek_chat", handler)
    event = await adapter.retranslate_window(request())
    await adapter.close()
    assert event.text == "你好，世界" and event.kind == TranslationKind.FINAL
    assert event.prompt_tokens == 9 and event.completion_tokens == 5


async def test_repetition_guard_and_timeout_yield_truncated_finals(caplog) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse("好的", *(["好的"] * 8), finish="length"))

    adapter = backend("groq_chat", handler)
    events = [event async for event in adapter.translate_incremental(request())]
    await adapter.close()
    assert events[-1].truncated is True and events[-1].finish_reason == "repetition"

    async def slow(http_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.5)
        return httpx.Response(200, text=sse("x"))

    adapter = backend("groq_chat", slow)
    with caplog.at_level(logging.WARNING):
        events = [event async for event in adapter.translate_incremental(request(timeout_s=0.05))]
    await adapter.close()
    assert events[-1].kind == TranslationKind.FINAL
    assert events[-1].finish_reason == "timeout" and events[-1].truncated is True
    assert any("Groq timed out" in record.getMessage() for record in caplog.records)
    assert KEY not in caplog.text


# ------------------------------------------------------------------ errors


async def test_maps_401_429_404_and_400_without_leaking_the_key() -> None:
    responses = iter(
        [
            httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {KEY}"}}),
            httpx.Response(429, json={"error": {"message": "Rate limit reached"}}),
            httpx.Response(404, json={"error": {"message": "The model `nope` does not exist"}}),
            httpx.Response(400, json={"error": {"message": "Unrecognized request argument supplied: stream_options"}}),
            httpx.Response(400, json={"error": {"message": "This model's maximum context length is 8192 tokens"}}),
            httpx.Response(503, text="upstream unavailable"),
        ]
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        return next(responses)

    adapter = backend("openrouter_chat", handler)
    with pytest.raises(AuthenticationError) as auth:
        await adapter.retranslate_window(request())
    assert "HTTP 401" in str(auth.value) and "OpenRouter" in str(auth.value)
    with pytest.raises(RateLimitError) as limit:
        await adapter.retranslate_window(request())
    assert "HTTP 429" in str(limit.value)
    with pytest.raises(TranslationRequestError) as missing:
        [event async for event in adapter.translate_incremental(request())]
    assert missing.value.error_code == "model_not_found"
    with pytest.raises(TranslationRequestError) as bad:
        await adapter.retranslate_window(contextual_request())
    assert bad.value.error_code == "bad_request" and bad.value.retry_without_context is False
    assert "stream_options" in str(bad.value)
    with pytest.raises(TranslationRequestError) as overflow:
        await adapter.retranslate_window(contextual_request())
    assert overflow.value.error_code == "context_overflow" and overflow.value.retry_without_context is True
    with pytest.raises(TranslationRequestError) as down:
        await adapter.retranslate_window(request())
    assert down.value.error_code == "http_503"
    await adapter.close()

    for info in (auth, limit, missing, bad, overflow, down):
        text = str(info.value)
        assert KEY not in text
        assert "openrouter.ai" not in text and "http://" not in text and "https://" not in text


async def test_privacy_gate_blocks_every_preset_before_any_request(caplog) -> None:
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=sse("x", finish="stop"))

    for preset in PRESETS:
        adapter = backend(preset, handler, transcript_upload_allowed=False, base_url="http://127.0.0.1:11434/v1" if preset == "custom_chat" else None)
        with pytest.raises(PolicyDeniedError):
            await anext(adapter.translate_incremental(request()))
        with pytest.raises(PolicyDeniedError):
            await adapter.retranslate_window(request())
        await adapter.close()
    assert calls == 0
    assert KEY not in caplog.text


async def test_payload_and_url_never_contain_the_key() -> None:
    seen: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen["url"] = str(http_request.url)
        seen["body"] = http_request.content.decode()
        return httpx.Response(200, text=sse("你好", finish="stop"))

    adapter = backend("openai_chat", handler)
    [event async for event in adapter.translate_incremental(contextual_request())]
    await adapter.close()
    assert KEY not in seen["url"] and KEY not in seen["body"]
    assert KEY not in repr(adapter.descriptor)
    assert KEY not in json.dumps(adapter.build_payload(request(), model="m", stream=True))


# ---------------------------------------------------------------- registry


@pytest.mark.parametrize("preset", PRESETS)
async def test_registry_factory_builds_every_preset_from_the_injected_environment(preset: str, monkeypatch) -> None:
    from echolingo.backends import registry
    from echolingo.config import AppConfig

    group, preset_url, model, key_env = CHAT_PRESETS[preset]
    monkeypatch.delenv(key_env, raising=False)
    config = AppConfig()
    config.privacy.transcript_upload_allowed = True
    env = {key_env: KEY, f"ECHOLINGO_{group.upper()}_CHAT_MODEL": "override-model"}
    if preset == "custom_chat":
        env["ECHOLINGO_CUSTOM_OPENAI_BASE_URL"] = "http://127.0.0.1:11434/v1"
    spec = registry.get("translation", preset)
    adapter = spec.factory(config, env)
    try:
        assert isinstance(adapter, OpenAiChatTranslation)
        assert adapter.provider_id == preset and adapter.descriptor.provider == preset
        assert adapter.model == "override-model"
        assert adapter.api_key == KEY
        assert adapter.transcript_upload_allowed is True
        assert adapter.base_url == (preset_url if preset_url else "http://127.0.0.1:11434/v1")
        assert adapter.chat_url().endswith("/chat/completions")
        assert adapter.temperature == config.translation.openai_chat.temperature
        assert adapter.background_spans == config.translation.openai_chat.background_spans
        assert adapter.max_output_tokens == config.translation.openai_chat.max_output_tokens
        assert adapter.authorize() is None
    finally:
        await adapter.close()
