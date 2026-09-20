from dataclasses import replace
from types import SimpleNamespace

import pytest

from echolingo.backends import registry
from echolingo.errors import AuthenticationError
from echolingo.service.cloud_probe import probe_cloud, probe_qwen_cloud


class ProbeAsr:
    model = "qwen3-asr-flash-realtime"

    def __init__(self, **kwargs):
        self.audio_upload_allowed = True

    async def probe_connection(self):
        assert self.audio_upload_allowed is False
        return 12.34

    def describe_endpoint(self):
        return {"region": "beijing", "host": "dashscope.aliyuncs.com", "workspace_scoped": False}


class ProbeTranslation:
    def __init__(self, **kwargs):
        self.transcript_upload_allowed = False
        self.closed = False

    async def retranslate_window(self, request):
        assert self.transcript_upload_allowed is True
        assert request.source_text == "Welcome to the lecture."
        return SimpleNamespace(model="qwen-mt-plus", total_latency_ms=45.67)

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_qwen(monkeypatch):
    asr = registry.get("asr", "qwen_cloud")
    mt = registry.get("translation", "qwen_cloud")
    monkeypatch.setitem(
        registry.PROVIDERS, ("asr", "qwen_cloud"), replace(asr, factory=lambda c, e: ProbeAsr())
    )
    monkeypatch.setitem(
        registry.PROVIDERS,
        ("translation", "qwen_cloud"),
        replace(mt, factory=lambda c, e: ProbeTranslation()),
    )


async def test_cloud_probe_does_not_upload_audio_and_can_test_translation(fake_qwen) -> None:
    result = await probe_cloud("qwen_cloud", "qwen_cloud")

    assert result["ok"] is True
    assert result["audio_uploaded"] is False
    assert result["asr"]["handshake_latency_ms"] == 12.3
    assert result["asr"]["provider"] == "qwen_cloud"
    assert result["translation"]["latency_ms"] == 45.7
    assert result["region"] == "beijing"
    assert result["host"] == "dashscope.aliyuncs.com"
    assert result["workspace_scoped"] is False


async def test_legacy_probe_wrapper_keeps_shape(fake_qwen) -> None:
    result = await probe_qwen_cloud(True)
    assert result["ok"] is True
    assert result["translation"]["status"] == "connected"
    result = await probe_qwen_cloud(False)
    assert result["translation"]["status"] == "skipped"


async def test_cloud_probe_returns_sanitized_authentication_failure(monkeypatch) -> None:
    class DeniedAsr(ProbeAsr):
        async def probe_connection(self):
            raise AuthenticationError(
                "Qwen Realtime ASR access was denied (HTTP 403). Verify the workspace."
            )

    asr = registry.get("asr", "qwen_cloud")
    monkeypatch.setitem(
        registry.PROVIDERS, ("asr", "qwen_cloud"), replace(asr, factory=lambda c, e: DeniedAsr())
    )

    result = await probe_cloud("qwen_cloud", "qwen_cloud")

    assert result["ok"] is False
    assert result["code"] == "authentication_failed"
    assert "HTTP 403" in result["message"]
    assert result["asr"]["status"] == "failed"
    assert result["translation"]["status"] == "not_tested"


async def test_cloud_probe_rejects_unknown_provider() -> None:
    result = await probe_cloud("nope")
    assert result["ok"] is False
    assert result["code"] == "unknown_provider"


async def test_cloud_probe_never_exposes_generic_exception_text(monkeypatch) -> None:
    class Exploding(ProbeAsr):
        async def probe_connection(self):
            raise RuntimeError("secret sk-123 leaked in response body")

    asr = registry.get("asr", "qwen_cloud")
    monkeypatch.setitem(
        registry.PROVIDERS, ("asr", "qwen_cloud"), replace(asr, factory=lambda c, e: Exploding())
    )
    result = await probe_cloud("qwen_cloud")
    assert result["code"] == "cloud_probe_failed"
    assert "sk-123" not in result["message"]
