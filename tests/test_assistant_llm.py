"""OpenAI-compatible assistant client: SSE parsing, continuation, error
mapping and cancellation. Mock transports only; no network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from echolingo.assistant import llm
from echolingo.assistant.llm import (
    CONTINUE_PROMPT,
    AssistantError,
    ChatClient,
    LlmEndpoint,
    resolve_endpoint,
)
from echolingo.backends import dashscope
from echolingo.errors import AuthenticationError, RateLimitError

SECRET = "sk-test-secret-value-1234567890"


def sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def text_chunk(text: str, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


def usage_chunk(prompt: int, completion: int) -> dict:
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def dashscope_endpoint(region: str = "beijing") -> LlmEndpoint:
    resolved = dashscope.resolve_endpoint(region, None)
    return LlmEndpoint(
        provider="dashscope",
        display_name="Qwen (Alibaba Model Studio)",
        base_url=resolved.compatible_base_url,
        model="qwen-plus",
        api_key=SECRET,
        stream_usage=True,
        dashscope_endpoint=resolved,
    )


def openai_endpoint(**overrides) -> LlmEndpoint:
    values = dict(
        provider="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key=SECRET,
        stream_usage=True,
    )
    values.update(overrides)
    return LlmEndpoint(**values)


def client_for(endpoint: LlmEndpoint, handler) -> ChatClient:
    return ChatClient(
        endpoint, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def collect(client: ChatClient, **kwargs):
    texts: list[str] = []
    final = None
    async for delta in client.stream_text(
        [{"role": "user", "content": "hi"}], max_tokens=kwargs.pop("max_tokens", 100), **kwargs
    ):
        if delta.text:
            texts.append(delta.text)
        else:
            final = delta
    return "".join(texts), final


async def test_stream_parses_sse_deltas_usage_and_dashscope_options() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = sse(
            text_chunk("# Title\n"),
            text_chunk("- point"),
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            usage_chunk(120, 7),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = client_for(dashscope_endpoint(), handler)
    text, final = await collect(client, temperature=0.3)
    assert text == "# Title\n- point"
    assert final.finish_reason == "stop"
    assert final.usage == {"prompt_tokens": 120, "completion_tokens": 7}
    assert final.continuations == 0

    request = requests[0]
    assert str(request.url) == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    payload = json.loads(request.content)
    assert payload["stream"] is True
    assert payload["enable_thinking"] is False
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["model"] == "qwen-plus"
    assert payload["max_tokens"] == 100


async def test_non_dashscope_payload_omits_enable_thinking() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=sse(text_chunk("ok", "stop")))

    client = client_for(openai_endpoint(stream_usage=False), handler)
    text, _final = await collect(client)
    assert text == "ok"
    assert "enable_thinking" not in seen[0]
    assert "stream_options" not in seen[0]


async def test_length_finish_continues_with_partial_answer_up_to_twice() -> None:
    requests: list[dict] = []
    answers = [
        sse(text_chunk("## Part one\n- alpha "), text_chunk("beta", "length"), usage_chunk(100, 50)),
        sse(text_chunk("- gamma", "length"), usage_chunk(160, 40)),
        sse(text_chunk("\n- delta", "length"), usage_chunk(200, 30)),
        sse(text_chunk("never requested", "stop")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=answers[len(requests) - 1])

    client = client_for(openai_endpoint(), handler)
    text, final = await collect(client)
    assert len(requests) == 3  # the first call plus MAX_CONTINUATIONS
    assert text == "## Part one\n- alpha beta- gamma\n- delta"
    assert final.finish_reason == "length"
    assert final.continuations == 2
    assert final.usage == {"prompt_tokens": 460, "completion_tokens": 120}
    second = requests[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "## Part one\n- alpha beta"}
    assert second[-1] == {"role": "user", "content": CONTINUE_PROMPT}
    third = requests[2]["messages"]
    assert third[-2] == {"role": "assistant", "content": "- gamma"}


async def test_continuation_drops_a_restated_tail() -> None:
    first = "The receptive field of a V1 simple cell is elongated and oriented. "
    answers = [
        sse(text_chunk(first, "length")),
        sse(text_chunk("of a V1 simple cell is elongated and oriented. Complex cells pool.", "stop")),
    ]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=answers[calls - 1])

    client = client_for(openai_endpoint(), handler)
    text, final = await collect(client)
    assert text == first + "Complex cells pool."
    assert final.continuations == 1


async def test_dashscope_401_names_region_and_host_without_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key {SECRET}"}})

    client = client_for(dashscope_endpoint("singapore"), handler)
    with pytest.raises(AuthenticationError) as caught:
        await collect(client)
    message = str(caught.value)
    assert "HTTP 401" in message
    assert "Singapore" in message and "dashscope-intl.aliyuncs.com" in message
    assert "Beijing" in message
    assert SECRET not in message
    assert caught.value.error_code == "authentication_failed"


async def test_dashscope_403_uses_forbidden_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "denied"}})

    client = client_for(dashscope_endpoint("beijing"), handler)
    with pytest.raises(AuthenticationError, match="HTTP 403"):
        await client.complete([{"role": "user", "content": "hi"}])


async def test_rate_limit_maps_to_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = client_for(openai_endpoint(), handler)
    with pytest.raises(RateLimitError) as caught:
        await collect(client)
    assert caught.value.error_code == "rate_limited"
    assert "429" in str(caught.value)


async def test_context_length_400_maps_to_context_too_long() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "This model's maximum context length is 128000 tokens.",
                    "code": "context_length_exceeded",
                }
            },
        )

    client = client_for(openai_endpoint(), handler)
    with pytest.raises(AssistantError) as caught:
        await collect(client)
    assert caught.value.code == "context_too_long"
    assert "128000" not in str(caught.value)


async def test_other_errors_report_only_the_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"internal trace with {SECRET} and prompt text")

    client = client_for(openai_endpoint(), handler)
    with pytest.raises(AssistantError) as caught:
        await client.complete([{"role": "user", "content": "hi"}])
    assert caught.value.code == "provider_error"
    assert str(caught.value) == "OpenAI returned HTTP 500."


async def test_rejected_optional_fields_are_dropped_and_retried() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "stream_options" in payload:
            return httpx.Response(400, json={"error": {"message": "Unknown field stream_options"}})
        if payload["max_tokens"] > 4096:
            return httpx.Response(400, json={"error": {"message": "max_tokens must be <= 4096"}})
        return httpx.Response(200, content=sse(text_chunk("ok", "stop")))

    client = client_for(openai_endpoint(), handler)
    text, _final = await collect(client, max_tokens=8192)
    assert text == "ok"
    assert len(payloads) == 3
    assert payloads[-1]["max_tokens"] == 4096 and "stream_options" not in payloads[-1]


async def test_complete_returns_message_content_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    client = client_for(openai_endpoint(), handler)
    result = await client.complete([{"role": "user", "content": "Reply with OK"}])
    assert result.text == "OK"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 1}


async def test_cancelling_the_consumer_closes_the_stream() -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    async def body():
        try:
            yield sse(text_chunk("partial"), done=False)
            started.set()
            await asyncio.Event().wait()  # the provider stalls
        finally:
            closed.set()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    client = client_for(openai_endpoint(), handler)
    received: list[str] = []

    async def consume() -> None:
        async for delta in client.stream_text([{"role": "user", "content": "hi"}]):
            if delta.text:
                received.append(delta.text)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 2)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), 2)
    assert received == ["partial"]


async def test_read_timeout_maps_to_provider_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    client = client_for(openai_endpoint(), handler)
    with pytest.raises(AssistantError) as caught:
        await collect(client)
    assert caught.value.code == "provider_timeout"


async def test_connect_error_maps_to_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = client_for(openai_endpoint(), handler)
    with pytest.raises(AssistantError) as caught:
        await collect(client)
    assert caught.value.code == "network_error"
    assert "api.openai.com" in str(caught.value)


# ------------------------------------------------------------ endpoint resolution


def test_dashscope_endpoint_follows_region_and_workspace_settings() -> None:
    env = {"DASHSCOPE_API_KEY": SECRET, "ECHOLINGO_QWEN_REGION": "beijing"}
    endpoint = resolve_endpoint({"group": "dashscope", "model": ""}, env)
    assert endpoint.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert endpoint.model == "qwen-plus"
    assert endpoint.dashscope_endpoint is not None
    assert SECRET not in repr(endpoint)

    default_region = resolve_endpoint({"group": "dashscope", "model": "qwen-max"}, {"DASHSCOPE_API_KEY": SECRET})
    assert default_region.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert default_region.model == "qwen-max"

    workspace = resolve_endpoint(
        {"group": "dashscope", "model": ""},
        {**env, "DASHSCOPE_WORKSPACE_ID": "ws123"},
    )
    assert workspace.base_url == "https://ws123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def test_custom_endpoint_reads_settings_and_skips_proxy_for_loopback() -> None:
    env = {
        "ECHOLINGO_CUSTOM_OPENAI_BASE_URL": "http://127.0.0.1:11434/v1/",
        "ECHOLINGO_CUSTOM_OPENAI_CHAT_MODEL": "qwen2.5:7b",
    }
    endpoint = resolve_endpoint({"group": "custom_openai", "model": ""}, env)
    assert endpoint.base_url == "http://127.0.0.1:11434/v1"
    assert endpoint.model == "qwen2.5:7b"
    assert endpoint.api_key is None
    local = ChatClient(endpoint)
    assert local.client.trust_env is False
    remote = ChatClient(openai_endpoint())
    assert remote.client.trust_env is True
    assert llm.is_loopback_url("http://localhost:8080/v1")
    assert not llm.is_loopback_url("https://api.openai.com/v1")


@pytest.mark.parametrize(
    ("llm_field", "env", "code"),
    [
        (None, {}, "not_configured"),
        ({"group": "", "model": ""}, {}, "not_configured"),
        ({"group": "nope", "model": ""}, {}, "invalid_request"),
        ({"group": "openai", "model": 7}, {"OPENAI_API_KEY": SECRET}, "invalid_request"),
        ({"group": "openai", "model": ""}, {}, "not_configured"),
        ({"group": "custom_openai", "model": ""}, {}, "not_configured"),
        (
            {"group": "custom_openai", "model": ""},
            {"ECHOLINGO_CUSTOM_OPENAI_BASE_URL": "ftp://x"},
            "not_configured",
        ),
    ],
)
def test_resolve_endpoint_rejects_incomplete_configuration(llm_field, env, code) -> None:
    with pytest.raises(AssistantError) as caught:
        resolve_endpoint(llm_field, env)
    assert caught.value.code == code


async def test_reasoning_model_parameter_rejections_are_adapted() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "max_tokens" in payload:
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}},
            )
        if "temperature" in payload:
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2 with this model. Only the default (1) value is supported."}},
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        )

    client = client_for(openai_endpoint(model="o4-mini"), handler)
    result = await client.complete([{"role": "user", "content": "hi"}], max_tokens=64)
    assert result.text == "OK"
    assert len(payloads) == 3
    assert payloads[-1]["max_completion_tokens"] == 64 and "temperature" not in payloads[-1]
