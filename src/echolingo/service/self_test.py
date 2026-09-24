"""Self-test of a sidecar build: ``echolingo-sidecar self-test [--qwen-runtime] [--cuda]``.

Every check loads what the packaged runtime needs at run time (native
extensions, bundled models and data files, the TLS trust store) so a broken
bundle is caught before it ships or before a downloaded runtime is activated.
stdout carries exactly one JSON object on a single line; anything the checked
libraries print goes to stderr. The exit code is 0 iff every check passed.
Heavy modules are imported inside the checks only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import platform
import ssl
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from .. import __version__

Check = Callable[[], str]
_DETAIL_LIMIT = 400


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def check_numpy() -> str:
    import numpy as np

    total = float(np.arange(4, dtype=np.float32).sum())
    if total != 6.0:
        raise RuntimeError(f"unexpected sum {total}")
    return _version(np)


def check_onnxruntime() -> str:
    import numpy as np
    import onnxruntime

    from ..vad import SileroOnnxVad
    from .session import runtime_resource_path

    model_path = runtime_resource_path("models/silero_vad.onnx")
    if not model_path.is_file():
        raise FileNotFoundError(f"bundled Silero VAD model is missing ({model_path})")
    probability = SileroOnnxVad(model_path).probability(np.zeros(1024, dtype=np.float32))
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Silero VAD returned {probability}")
    return f"{_version(onnxruntime)}; silero_vad p(silence)={probability:.3f}"


def check_samplerate() -> str:
    import numpy as np
    import samplerate

    from ..resample import StreamingResampler

    output = StreamingResampler(48_000, 16_000).process(
        np.zeros(4_800, dtype=np.float32), end_of_input=True
    )
    if output.size != 1_600:
        raise RuntimeError(f"resampled 4800 samples to {output.size}, expected 1600")
    return _version(samplerate)


def check_pywebrtc_audio() -> str:
    import numpy as np

    from ..enhancement import WebRtcProcessor
    from ..models import AudioFrame

    processor = WebRtcProcessor(48_000, 1, "webrtc_ns_agc")
    frame = AudioFrame(0, 0, None, 48_000, 1, np.zeros(960, dtype=np.float32), "self-test")
    output, _, _ = processor.process(frame)
    if output.shape != (960, 1):
        raise RuntimeError(f"noise suppression returned shape {output.shape}")
    return "noise suppression + AGC on 20 ms at 48 kHz"


def check_pypdf() -> str:
    import io

    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = len(pypdf.PdfReader(io.BytesIO(buffer.getvalue())).pages)
    if pages != 1:
        raise RuntimeError(f"round-tripped PDF has {pages} pages")
    return _version(pypdf)


def check_httpx() -> str:
    import httpx

    # Building a client loads its certifi-backed TLS context.
    with httpx.Client():
        pass
    return _version(httpx)


def check_websockets() -> str:
    import websockets
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    async def round_trip() -> str:
        async def echo(connection: Any) -> None:
            async for message in connection:
                await connection.send(message)

        async with serve(echo, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None, open_timeout=5) as client:
                await client.send("ping")
                return str(await asyncio.wait_for(client.recv(), 5))

    reply = asyncio.run(round_trip())
    if reply != "ping":
        raise RuntimeError(f"loopback echo returned {reply!r}")
    return f"{_version(websockets)}; loopback round trip"


def check_torch() -> str:
    import torch

    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    total = float((matrix @ matrix.T).sum())
    if total != 83.0:
        raise RuntimeError(f"unexpected matmul sum {total}")
    return f"{torch.__version__}; threads={torch.get_num_threads()}"


def check_transformers() -> str:
    import transformers
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer  # noqa: F401

    return _version(transformers)


def check_qwen_asr() -> str:
    from qwen_asr import Qwen3ForcedAligner

    return f"{Qwen3ForcedAligner.__name__} importable"


def check_nagisa() -> str:
    import nagisa

    # Loads the bundled tokenizer model the Japanese forced alignment uses.
    words = nagisa.tagging("東京で講義を聞く").words
    if not words:
        raise RuntimeError("tokenizer returned no words")
    return f"{_version(nagisa)}; {len(words)} tokens"


def check_librosa() -> str:
    import librosa

    return _version(librosa)


def ssl_ca_certificate_count() -> int:
    return int(ssl.create_default_context().cert_store_stats().get("x509_ca", 0))


def check_ssl() -> str:
    count = ssl_ca_certificate_count()
    source = "SSL_CERT_FILE" if os.environ.get("SSL_CERT_FILE") else "default paths"
    if count <= 0:
        raise RuntimeError(f"no CA certificates loaded from {source}")
    return f"{ssl.OPENSSL_VERSION}; {count} CA certificates from {source}"


def check_sidecar_service() -> str:
    # The default (WebSocket service) mode's whole import graph.
    importlib.import_module("echolingo.service.server")
    return __version__


def import_without_cli_arguments(module_name: str) -> Any:
    """Import a module that parses ``sys.argv`` at import time with an empty CLI."""
    saved = sys.argv
    sys.argv = [saved[0] if saved else "echolingo-sidecar"]
    try:
        return importlib.import_module(module_name)
    finally:
        sys.argv = saved


def check_qwen3_streaming() -> str:
    import_without_cli_arguments("whisperlivekit.qwen3_streaming")
    return "whisperlivekit.qwen3_streaming"


def check_basic_server() -> str:
    # WhisperLiveKit parses its command line when basic_server is imported.
    import_without_cli_arguments("whisperlivekit.basic_server")
    return "whisperlivekit.basic_server"


def check_qwen_server() -> str:
    import_without_cli_arguments("echolingo.service.qwen_server")
    return "echolingo.service.qwen_server"


def default_checks() -> dict[str, Check]:
    return {
        "numpy": check_numpy,
        "onnxruntime": check_onnxruntime,
        "samplerate": check_samplerate,
        "pywebrtc_audio": check_pywebrtc_audio,
        "pypdf": check_pypdf,
        "httpx": check_httpx,
        "websockets": check_websockets,
        "ssl": check_ssl,
        "torch": check_torch,
        "transformers": check_transformers,
        "qwen_asr": check_qwen_asr,
        "nagisa": check_nagisa,
        "librosa": check_librosa,
        "sidecar_service": check_sidecar_service,
    }


def qwen_runtime_checks() -> dict[str, Check]:
    return {
        "qwen3_streaming": check_qwen3_streaming,
        "wlk_basic_server": check_basic_server,
        "qwen_server": check_qwen_server,
    }


def _error_detail(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:_DETAIL_LIMIT]


def run_check(check: Check) -> dict[str, Any]:
    try:
        return {"ok": True, "detail": str(check())[:_DETAIL_LIMIT]}
    except Exception as error:
        return {"ok": False, "detail": _error_detail(error)}


def _matmul_matches(torch: Any, device: str, dtype: Any) -> bool:
    # Small integers keep every product and sum exact in fp16/bf16 (and TF32).
    left = (torch.arange(256).reshape(16, 16) % 7 - 3).to(torch.float32)
    right = (torch.arange(256).reshape(16, 16) % 5 - 2).to(torch.float32)
    expected = left @ right
    actual = left.to(device=device, dtype=dtype) @ right.to(device=device, dtype=dtype)
    if device != "cpu":
        torch.cuda.synchronize()
    return bool(torch.equal(actual.float().cpu(), expected))


def cuda_report(torch_module: Any | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_version": None,
        "cuda_version": None,
        "available": False,
        "device_name": None,
        "capability": None,
        "bf16": None,
        "arch_list": [],
        "matmul_ok": None,
        "error": None,
    }
    try:
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
        report["torch_version"] = str(torch.__version__)
        cuda_version = getattr(torch.version, "cuda", None)
        report["cuda_version"] = str(cuda_version) if cuda_version else None
        report["arch_list"] = [str(arch) for arch in torch.cuda.get_arch_list()]
        report["available"] = bool(torch.cuda.is_available())
    except Exception as error:
        report["error"] = _error_detail(error)
        return report
    if not report["available"]:
        return report
    try:
        from .qwen_server import cuda_bf16_supported

        report["device_name"] = str(torch.cuda.get_device_name(0))
        major, minor = torch.cuda.get_device_capability(0)
        report["capability"] = f"{major}.{minor}"
        report["bf16"] = cuda_bf16_supported(torch)
        # float32 plus the dtype the Qwen server will load the model in.
        model_dtype = torch.bfloat16 if report["bf16"] else torch.float16
        report["matmul_ok"] = all(
            _matmul_matches(torch, "cuda", dtype) for dtype in (torch.float32, model_dtype)
        )
    except Exception as error:
        report["error"] = _error_detail(error)
        report["matmul_ok"] = False
    return report


def _cuda_check(report: Mapping[str, Any]) -> dict[str, Any]:
    if report["error"] is not None:
        return {"ok": False, "detail": str(report["error"])}
    if not report["available"]:
        # A machine without a usable NVIDIA GPU is not a broken build.
        return {"ok": True, "detail": f"CUDA unavailable (torch CUDA {report['cuda_version']})"}
    if not report["matmul_ok"]:
        return {"ok": False, "detail": f"GPU matmul mismatch on {report['device_name']}"}
    return {"ok": True, "detail": f"{report['device_name']} (sm {report['capability']})"}


def run_self_test(
    *,
    qwen_runtime: bool = False,
    cuda: bool = False,
    install_qwen_shims: Callable[[], None] | None = None,
    checks: Mapping[str, Check] | None = None,
    runtime_checks: Mapping[str, Check] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    if checks is None:
        checks = default_checks()
    results = {name: run_check(check) for name, check in checks.items()}
    if qwen_runtime:
        # Same order as the qwen-asr-server mode: shims before WhisperLiveKit.
        if install_qwen_shims is not None:
            install_qwen_shims()
        if runtime_checks is None:
            runtime_checks = qwen_runtime_checks()
        for name, check in runtime_checks.items():
            results[name] = run_check(check)
    cuda_result = None
    if cuda:
        cuda_result = cuda_report(torch_module)
        results["cuda"] = _cuda_check(cuda_result)
    return {
        "ok": all(result["ok"] for result in results.values()),
        "version": __version__,
        "platform": f"{sys.platform}-{platform.machine().lower()}",
        "frozen": bool(getattr(sys, "frozen", False)),
        "utf8_mode": bool(sys.flags.utf8_mode),
        "ssl_ca_certs": _safe_ca_count(),
        "checks": results,
        "cuda": cuda_result,
    }


def _safe_ca_count() -> int:
    try:
        return ssl_ca_certificate_count()
    except Exception:
        return 0


def _flush_c_stdio() -> None:
    if sys.platform == "win32":
        return
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Route Python and native writes to stdout into stderr for the block."""
    sys.stdout.flush()
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        _flush_c_stdio()
        if saved is not None:
            os.dup2(saved, 1)
            os.close(saved)


def main(
    argv: Sequence[str] | None = None,
    *,
    install_qwen_shims: Callable[[], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="echolingo-sidecar self-test")
    parser.add_argument(
        "--qwen-runtime",
        action="store_true",
        help="also import the Qwen ASR server stack",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="report CUDA and run a small matmul on the GPU when one is usable",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    stdout = sys.stdout
    with _stdout_to_stderr():
        report = run_self_test(
            qwen_runtime=args.qwen_runtime,
            cuda=args.cuda,
            install_qwen_shims=install_qwen_shims,
        )
    stdout.write(json.dumps(report, separators=(",", ":")) + "\n")
    stdout.flush()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
