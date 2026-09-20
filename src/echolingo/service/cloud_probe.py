"""Connection probes for cloud providers.

A probe validates credentials, region and network reachability. ASR probes
perform the authenticated handshake only and never upload audio; translation
probes send one fixed English sentence and are only run when the caller has
confirmed transcript upload consent (the desktop "Test" button does so
explicitly). Results never include provider response bodies or secrets.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from ..backends import registry
from ..config import load_config
from ..errors import AuthenticationError, BackendError
from ..translation.policy import TranslationRequestError

log = logging.getLogger("echolingo.cloud_probe")


async def probe_cloud(
    asr_provider: str | None = None,
    translation_provider: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    config=None,
) -> dict[str, Any]:
    """Probe the selected providers; ``ok`` is true only when every probe passed."""
    environ = os.environ if environ is None else environ
    if config is None:
        config = load_config()
    result: dict[str, Any] = {
        "ok": False,
        "audio_uploaded": False,
        "asr": {"status": "skipped"},
        "translation": {"status": "skipped"},
    }
    if asr_provider:
        result["asr"] = {"status": "pending", "provider": asr_provider}
    if translation_provider:
        result["translation"] = {"status": "pending", "provider": translation_provider}

    for kind, provider_id in (("asr", asr_provider), ("translation", translation_provider)):
        if not provider_id:
            continue
        spec = registry.find(kind, provider_id)
        if spec is None or spec.probe is None:
            result[kind] = {"status": "failed", "provider": provider_id}
            result["code"] = "unknown_provider"
            result["message"] = f"{provider_id!r} is not a probeable {kind} provider"
            return result
        try:
            result[kind] = await spec.probe(spec, config, environ)
        except Exception as error:
            result[kind] = {"status": "failed", "provider": provider_id}
            result["code"] = _error_code(error)
            result["message"] = _safe_message(spec.display_name, error)
            log.warning("%s probe failed: %s (%s)", spec.display_name, result["code"], result["message"])
            other = "translation" if kind == "asr" else "asr"
            if result[other].get("status") == "pending":
                result[other] = {"status": "not_tested", "provider": result[other]["provider"]}
            return result
        # DashScope endpoints report region/host so the UI can show what was tried.
        for key in ("region", "host", "workspace_scoped"):
            if key in result[kind] and key not in result:
                result[key] = result[kind][key]
        log.info("%s probe connected", spec.display_name)

    result["ok"] = True
    return result


async def probe_qwen_cloud(include_translation: bool, **_: Any) -> dict[str, Any]:
    """Backwards-compatible wrapper for the original single-provider probe."""
    result = await probe_cloud("qwen_cloud", "qwen_cloud" if include_translation else None)
    result.setdefault("region", load_config().asr.qwen_cloud.region)
    return result


def _error_code(error: Exception) -> str:
    if isinstance(error, BackendError):
        return error.code
    if isinstance(error, TranslationRequestError):
        return error.error_code
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return "network_error"
    return "cloud_probe_failed"


def _safe_message(display_name: str, error: Exception) -> str:
    if isinstance(error, (AuthenticationError, BackendError, ConnectionError, TranslationRequestError)):
        # Adapter-raised errors carry sanitized, provider-neutral text.
        return str(error)
    if isinstance(error, (TimeoutError, OSError)):
        return f"{display_name} could not be reached. Check the network and try again."
    return f"{display_name} validation failed without exposing provider diagnostics."
