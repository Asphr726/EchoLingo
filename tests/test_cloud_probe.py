from types import SimpleNamespace

from echolingo.errors import AuthenticationError
from echolingo.service.cloud_probe import probe_qwen_cloud


class ProbeAsr:
    model = "qwen3-asr-flash-realtime"

    def __init__(self, **kwargs):
        assert kwargs["audio_upload_allowed"] is False

    async def probe_connection(self):
        return 12.34


class ProbeTranslation:
    def __init__(self, **kwargs):
        assert kwargs["transcript_upload_allowed"] is True
        self.closed = False

    async def retranslate_window(self, request):
        assert request.source_text == "Welcome to the lecture."
        return SimpleNamespace(model="qwen-mt-plus", total_latency_ms=45.67)

    async def close(self):
        self.closed = True


async def test_cloud_probe_does_not_upload_audio_and_can_test_translation() -> None:
    result = await probe_qwen_cloud(
        True, asr_factory=ProbeAsr, translation_factory=ProbeTranslation
    )

    assert result["ok"] is True
    assert result["audio_uploaded"] is False
    assert result["asr"]["handshake_latency_ms"] == 12.3
    assert result["translation"]["latency_ms"] == 45.7


async def test_cloud_probe_returns_sanitized_authentication_failure() -> None:
    class DeniedAsr(ProbeAsr):
        async def probe_connection(self):
            raise AuthenticationError(
                "Qwen Realtime ASR access was denied (HTTP 403). Verify the workspace."
            )

    result = await probe_qwen_cloud(False, asr_factory=DeniedAsr)

    assert result["ok"] is False
    assert result["code"] == "authentication_failed"
    assert "HTTP 403" in result["message"]
    assert result["translation"]["status"] == "not_tested"
