from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from ..backends.asr.cloud_qwen import CloudQwenAsrBackend
from ..backends.translation.cloud_qwen_mt import CloudQwenMtBackend
from ..config.loader import qwen_region_from_environment
from ..errors import AuthenticationError, BackendError
from ..models import TranslationRequest


async def probe_qwen_cloud(
    include_translation: bool,
    *,
    asr_factory: Callable[..., Any] = CloudQwenAsrBackend,
    translation_factory: Callable[..., Any] = CloudQwenMtBackend,
) -> dict[str, Any]:
    """Probe cloud routes without uploading audio or returning provider content."""
    region = qwen_region_from_environment()
    result: dict[str, Any] = {
        "ok": False,
        "region": region,
        "audio_uploaded": False,
        "asr": {"status": "pending"},
        "translation": {"status": "pending" if include_translation else "skipped"},
    }
    asr = asr_factory(audio_upload_allowed=False, region=region)
    try:
        handshake_ms = await asr.probe_connection()
        result["asr"] = {
            "status": "connected",
            "model": asr.model,
            "handshake_latency_ms": round(handshake_ms, 1),
        }
    except Exception as error:
        result["code"] = _error_code(error)
        result["message"] = _safe_message(error)
        result["translation"] = {"status": "not_tested"}
        return result

    if include_translation:
        translation = translation_factory(transcript_upload_allowed=True, region=region)
        started_ns = time.monotonic_ns()
        try:
            event = await translation.retranslate_window(
                TranslationRequest(
                    request_id=str(uuid.uuid4()),
                    source_revision_id=0,
                    source_text="Welcome to the lecture.",
                    source_lang="en",
                    target_lang="zh",
                    final=True,
                )
            )
            result["translation"] = {
                "status": "connected",
                "model": event.model,
                "latency_ms": round(
                    event.total_latency_ms
                    or (time.monotonic_ns() - started_ns) / 1_000_000.0,
                    1,
                ),
            }
        except Exception as error:
            result["code"] = _error_code(error)
            result["message"] = _safe_message(error)
            result["translation"] = {"status": "failed"}
            return result
        finally:
            await translation.close()

    result["ok"] = True
    return result


def _error_code(error: Exception) -> str:
    if isinstance(error, BackendError):
        return error.code
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return "network_error"
    return "cloud_probe_failed"


def _safe_message(error: Exception) -> str:
    if isinstance(error, (AuthenticationError, BackendError, ConnectionError)):
        return str(error)
    if isinstance(error, (TimeoutError, OSError)):
        return "Qwen Cloud could not be reached. Check the network and try again."
    return "Qwen Cloud validation failed without exposing provider diagnostics."
