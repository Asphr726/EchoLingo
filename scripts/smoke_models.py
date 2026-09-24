"""End-to-end model smoke test of a packaged local runtime.

Downloads the pinned Qwen3-ASR and Hy-MT2 models (the desktop catalog's
revisions and digests), starts ``qwen-asr-server`` and ``llama-server`` with
the arguments the desktop app uses, streams whisper.cpp's ``jfk.wav`` through
``LocalQwenAsrBackend`` and translates one sentence through
``LocalHyMtBackend``.  Only correctness fails the run; real-time factor,
latency and tokens per second are reported to stdout and the GitHub step
summary.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from echolingo.backends.asr.local_qwen import LocalQwenAsrBackend  # noqa: E402
from echolingo.backends.translation.local_hymt import LocalHyMtBackend  # noqa: E402
from echolingo.models import (  # noqa: E402
    AsrAudioChunk,
    AsrSessionConfig,
    TranscriptKind,
    TranslationKind,
    TranslationRequest,
)

# Mirrors catalog() in crates/runtime-manager/src/lib.rs (tests keep them equal).
ASR_MODEL = {
    "id": "qwen3-asr-0.6b",
    "repository": "Qwen/Qwen3-ASR-0.6B",
    "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "expected_bytes": 1_876_091_704,
    "required_file": "model.safetensors",
    "required_file_sha256": "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
    "allow_patterns": ["*.json", "*.txt", "*.md", "model.safetensors"],
}
MT_MODEL = {
    "id": "hymt2-1.8b",
    "repository": "tencent/Hy-MT2-1.8B-GGUF",
    "revision": "1cd5208700acedef4ef93019b6cfc148b8522d45",
    "expected_bytes": 1_133_080_448,
    "required_file": "Hy-MT2-1.8B-Q4_K_M.gguf",
    "required_file_sha256": "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
}
JFK_URL = "https://github.com/ggml-org/whisper.cpp/raw/v1.8.0/samples/jfk.wav"
JFK_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
JFK_REFERENCE = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)
JFK_REQUIRED_PHRASE = "ask not what your country can do for you"
TRANSLATION_SOURCE = (
    "And so, my fellow Americans: ask not what your country can do for you, "
    "ask what you can do for your country."
)
# The desktop app's Qwen streaming defaults (QwenStreamingProfile::default()).
QWEN_STREAMING_ARGS = [
    "--qwen3-streaming-chunk-sec", "1",
    "--qwen3-streaming-stable-iterations", "1",
    "--qwen3-streaming-hold-back-words", "4",
    "--qwen3-streaming-left-context-sec", "12",
    "--qwen3-streaming-right-context-ms", "640",
    "--qwen3-streaming-segment-max-steps", "200",
]  # fmt: skip
EXE = ".exe" if sys.platform == "win32" else ""


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_ready(directory: Path, spec: dict[str, Any]) -> bool:
    required = directory / spec["required_file"]
    return (
        required.is_file()
        and required.stat().st_size == spec["expected_bytes"]
        and sha256_file(required) == spec["required_file_sha256"]
    )


def ensure_model(models_dir: Path, spec: dict[str, Any]) -> Path:
    directory = models_dir / spec["id"]
    if model_ready(directory, spec):
        return directory
    from huggingface_hub import hf_hub_download, snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    if "allow_patterns" in spec:
        snapshot_download(
            repo_id=spec["repository"],
            revision=spec["revision"],
            allow_patterns=spec["allow_patterns"],
            local_dir=directory,
        )
    else:
        hf_hub_download(
            repo_id=spec["repository"],
            filename=spec["required_file"],
            revision=spec["revision"],
            local_dir=directory,
        )
    if not model_ready(directory, spec):
        raise RuntimeError(f"{spec['id']} failed its size/SHA256 check after download")
    return directory


def ensure_jfk(cache: Path) -> Path:
    path = cache / "jfk.wav"
    if path.is_file() and sha256_file(path) == JFK_SHA256:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("jfk.wav.download")
    urllib.request.urlretrieve(JFK_URL, temporary)
    if sha256_file(temporary) != JFK_SHA256:
        temporary.unlink()
        raise RuntimeError("jfk.wav checksum mismatch")
    temporary.replace(path)
    return path


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16_000):
            raise RuntimeError("jfk.wav is expected to be 16 kHz mono 16-bit PCM")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, other in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other))
        previous = current
    return previous[-1] / max(1, len(ref))


def transcript_ok(text: str) -> bool:
    return " ".join(JFK_REQUIRED_PHRASE.split()) in " ".join(normalize_words(text))


def translation_ok(text: str) -> bool:
    han = re.findall(r"[一-鿿]", text)
    return len(han) >= 6 and "国家" in text


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def healthy(port: int) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def child_environment(extra: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        ECHOLINGO_PARENT_PID=str(os.getpid()),
        **extra,
    )
    return environment


class Service:
    def __init__(self, name: str, command: list[str], environment: dict[str, str], port: int, log: Path):
        self.name = name
        self.port = port
        self.log = log
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log.open("wb")
        self.started = time.perf_counter()
        options: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            **options,
        )

    def check(self) -> float | None:
        """Seconds from launch to healthy, or None while still starting."""
        if healthy(self.port):
            return time.perf_counter() - self.started
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.name} exited with {self.process.returncode}\n{self.tail()}")
        return None

    def tail(self, limit: int = 6000) -> str:
        self._log_handle.flush()
        return self.log.read_bytes()[-limit:].decode("utf-8", "replace")

    def stop(self) -> None:
        """Stop the service and its children (watch-process wraps llama-server)."""
        if self.process.poll() is None:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                    capture_output=True,
                )
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if sys.platform != "win32":
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.kill()
                self.process.wait()
        self._log_handle.close()


def wait_healthy(services: list[Service], timeout_s: float) -> dict[str, float]:
    """Poll every service until healthy; return each one's load time."""
    loaded: dict[str, float] = {}
    deadline = time.perf_counter() + timeout_s
    while True:
        for service in services:
            if service.name not in loaded:
                elapsed = service.check()
                if elapsed is not None:
                    loaded[service.name] = round(elapsed, 1)
        if len(loaded) == len(services):
            return loaded
        if time.perf_counter() > deadline:
            waiting = [service for service in services if service.name not in loaded]
            raise RuntimeError(
                f"{waiting[0].name} not healthy after {timeout_s:.0f} s\n{waiting[0].tail()}"
            )
        time.sleep(0.5)


def streaming_args(overrides: list[str]) -> list[str]:
    """QWEN_STREAMING_ARGS with ``name=value`` overrides (e.g. ``chunk-sec=2``)."""
    values = dict(zip(QWEN_STREAMING_ARGS[0::2], QWEN_STREAMING_ARGS[1::2]))
    for override in overrides:
        name, _, value = override.partition("=")
        flag = f"--qwen3-streaming-{name}"
        if flag not in values or not value:
            raise SystemExit(f"unknown streaming override {override!r}")
        values[flag] = value
    return [item for pair in values.items() for item in pair]


def qwen_command(
    sidecar: Path, model: Path, port: int, device: str, streaming: list[str] | None = None
) -> list[str]:
    # Same arguments as LocalRuntimeManager::command_for("qwen_asr").
    return [
        str(sidecar), "qwen-asr-server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--backend", "qwen3-streaming",
        "--model_dir", str(model),
        "--lan", "en",
        "--pcm-input", "--no-vac", "--no-vad",
        "--warmup-file", "",
        "--qwen3-streaming-device", device,
        *(streaming or QWEN_STREAMING_ARGS),
        "--log-level", "INFO",
    ]  # fmt: skip


def llama_command(sidecar: Path, server: Path, model: Path, port: int) -> list[str]:
    # Same arguments as LocalRuntimeManager::command_for("hymt").
    return [
        str(sidecar), "watch-process", str(server),
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", "tencent/Hy-MT2-1.8B",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--n-gpu-layers", "99",
        "--jinja",
        "--n-predict", "400",
    ]  # fmt: skip


async def transcribe(url: str, samples: np.ndarray, pace: float) -> dict[str, Any]:
    backend = LocalQwenAsrBackend(url=url, language="en", finish_timeout_s=120.0)
    events: list[tuple[float, Any]] = []
    await backend.start_session(AsrSessionConfig(f"smoke-{uuid.uuid4().hex}", "en"))
    started = time.perf_counter()

    async def consume() -> None:
        async for event in backend.events():
            events.append((time.perf_counter() - started, event))

    consumer = asyncio.create_task(consume())
    chunk_samples = 1_600
    max_lag_ms = 0.0
    try:
        for sequence, offset in enumerate(range(0, samples.size, chunk_samples)):
            chunk = samples[offset : offset + chunk_samples]
            await backend.push_audio(
                AsrAudioChunk(
                    sequence=sequence,
                    start_ms=offset / 16.0,
                    end_ms=(offset + chunk.size) / 16.0,
                    sample_rate_hz=16_000,
                    samples=chunk,
                    speech_detected=True,
                )
            )
            max_lag_ms = max(max_lag_ms, backend.lag_ms or 0.0)
            if pace > 0:
                await asyncio.sleep(chunk.size / 16_000 / pace)
        audio_sent = time.perf_counter() - started
        await backend.finish_session()
        await consumer
    finally:
        await backend.close()
    done = time.perf_counter() - started
    audio_seconds = samples.size / 16_000
    finals = [event for _, event in events if event.kind == TranscriptKind.FINAL]
    text = finals[-1].text if finals else ""
    first_partial = next(
        (elapsed for elapsed, event in events if event.kind == TranscriptKind.PARTIAL), None
    )
    last_output = max((elapsed for elapsed, event in events if event.text), default=done)
    return {
        "text": text,
        "wer": round(word_error_rate(JFK_REFERENCE, text), 3),
        "audio_seconds": round(audio_seconds, 2),
        "first_partial_s": round(first_partial, 2) if first_partial is not None else None,
        "finish_latency_s": round(last_output - audio_sent, 2),
        "wall_over_audio": round(last_output / audio_seconds, 2),
        "max_lag_ms": round(max(max_lag_ms, backend.lag_ms or 0.0), 1),
    }


async def translate(base_url: str, api_key: str) -> dict[str, Any]:
    backend = LocalHyMtBackend(base_url=base_url, api_key=api_key, commit_timeout_s=120.0)
    request = TranslationRequest(
        request_id=uuid.uuid4().hex,
        source_revision_id=1,
        source_text=TRANSLATION_SOURCE,
        source_lang="en",
        target_lang="zh",
        final=True,
        source_committed=True,
        timeout_s=120.0,
    )
    final = None
    try:
        async for event in backend.translate_incremental(request):
            if event.kind == TranslationKind.FINAL:
                final = event
    finally:
        await backend.close()
    if final is None:
        return {"text": "", "finish_reason": "missing"}
    generation_s = None
    if final.total_latency_ms is not None and final.first_delta_latency_ms is not None:
        generation_s = (final.total_latency_ms - final.first_delta_latency_ms) / 1000.0
    tokens_per_second = None
    if final.completion_tokens and generation_s and generation_s > 0:
        tokens_per_second = round(max(0, final.completion_tokens - 1) / generation_s, 1)
    return {
        "text": final.text,
        "finish_reason": final.finish_reason,
        "truncated": final.truncated,
        "first_token_ms": round(final.first_delta_latency_ms or 0.0, 1),
        "total_ms": round(final.total_latency_ms or 0.0, 1),
        "completion_tokens": final.completion_tokens,
        "tokens_per_second": tokens_per_second,
    }


def locate(root: Path, relative: str) -> Path:
    matches = sorted(root.rglob(relative))
    if not matches:
        raise FileNotFoundError(f"{relative} not found under {root}")
    return matches[0]


def resolve_runtime(args: argparse.Namespace) -> tuple[Path, Path]:
    """(sidecar, llama-server): explicit paths, an installed app or a GPU pack.

    With a GPU pack the pack's sidecar runs both services; llama-server comes
    from ``--llama-server``/``--install-root`` when given, else from the pack.
    """
    sidecar = args.sidecar.resolve() if args.sidecar else None
    llama = args.llama_server.resolve() if args.llama_server else None
    if args.install_root:
        root = args.install_root.resolve()
        sidecar = sidecar or locate(root, f"sidecar/echolingo-sidecar{EXE}")
        llama = llama or locate(root, f"runtimes/llama.cpp/llama-server{EXE}")
    if args.gpu_pack_manifest:
        from build_gpu_pack import extract_pack

        pack = extract_pack(args.gpu_pack_manifest.resolve(), args.gpu_pack_dest.resolve())
        sidecar = pack / "sidecar" / f"echolingo-sidecar{EXE}"
        llama = llama or pack / "llama.cpp" / f"llama-server{EXE}"
    if sidecar is None or llama is None:
        raise SystemExit("pass --install-root, --gpu-pack-manifest, or --sidecar and --llama-server")
    return sidecar, llama


def summary_markdown(label: str, result: dict[str, Any]) -> str:
    asr, mt = result.get("asr") or {}, result.get("translation") or {}
    lines = [
        f"### Model smoke: {label} ({'PASS' if result['ok'] else 'FAIL'})",
        "",
        f"- Paths: sidecar `{result['sidecar']}`, models `{result['models_dir']}`",
        f"- ASR load {result.get('asr_load_s')} s; transcript: {asr.get('text')!r} (WER {asr.get('wer')})",
        f"- ASR {asr.get('audio_seconds')} s audio: wall/audio {asr.get('wall_over_audio')}, "
        f"finish latency {asr.get('finish_latency_s')} s, first partial {asr.get('first_partial_s')} s, "
        f"max lag {asr.get('max_lag_ms')} ms",
        f"- MT load {result.get('mt_load_s')} s; translation: {mt.get('text')!r}",
        f"- MT first token {mt.get('first_token_ms')} ms, total {mt.get('total_ms')} ms, "
        f"{mt.get('completion_tokens')} tokens, {mt.get('tokens_per_second')} tokens/s",
    ]
    lines += [f"- FAIL {failure}" for failure in result["failures"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, help="installed app to search for the runtime")
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--gpu-pack-manifest", type=Path, help="outer GPU pack manifest next to its parts")
    parser.add_argument("--gpu-pack-dest", type=Path, help="where to unpack the GPU pack")
    parser.add_argument("--device", default="auto", help="--qwen3-streaming-device")
    parser.add_argument(
        "--qwen-streaming",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a desktop streaming default, e.g. chunk-sec=2 or left-context-sec=8",
    )
    parser.add_argument("--pace", type=float, default=1.0, help="audio speed; 0 sends as fast as possible")
    parser.add_argument("--cache", type=Path, default=ROOT / "target" / "smoke-cache")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "target" / "smoke-logs")
    parser.add_argument("--label", default="local runtime")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.gpu_pack_manifest and not args.gpu_pack_dest:
        parser.error("--gpu-pack-dest is required with --gpu-pack-manifest")

    sidecar, llama_server = resolve_runtime(args)
    models_dir = args.models_dir.resolve()
    asr_model = ensure_model(models_dir, ASR_MODEL)
    mt_model = ensure_model(models_dir, MT_MODEL) / MT_MODEL["required_file"]
    samples = read_wav(ensure_jfk(args.cache))

    result: dict[str, Any] = {
        "label": args.label,
        "sidecar": str(sidecar),
        "llama_server": str(llama_server),
        "models_dir": str(models_dir),
        "failures": [],
    }
    token = secrets.token_hex(32)
    services: list[Service] = []
    try:
        qwen_port, mt_port = free_port(), free_port()
        qwen = Service(
            "qwen-asr-server",
            qwen_command(sidecar, asr_model, qwen_port, args.device, streaming_args(args.qwen_streaming)),
            child_environment({"WLK_API_TOKEN": token}),
            qwen_port,
            args.log_dir / "qwen_asr.log",
        )
        services.append(qwen)
        llama = Service(
            "llama-server",
            llama_command(sidecar, llama_server, mt_model, mt_port),
            child_environment({"LLAMA_API_KEY": token}),
            mt_port,
            args.log_dir / "hymt.log",
        )
        services.append(llama)
        loaded = wait_healthy(services, args.startup_timeout)
        result["asr_load_s"] = loaded[qwen.name]
        result["mt_load_s"] = loaded[llama.name]

        os.environ["WLK_API_TOKEN"] = token
        asr = asyncio.run(transcribe(f"ws://127.0.0.1:{qwen_port}/asr", samples, args.pace))
        result["asr"] = asr
        if not transcript_ok(asr["text"]):
            result["failures"].append(f"transcript lacks {JFK_REQUIRED_PHRASE!r}")
        translation = asyncio.run(translate(f"http://127.0.0.1:{mt_port}/v1", token))
        result["translation"] = translation
        if not translation_ok(translation["text"]) or translation.get("truncated"):
            result["failures"].append("translation is not a complete Chinese sentence about 国家")
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        for service in services:
            if result["failures"]:
                print(f"--- {service.name} log tail ---\n{service.tail()}", file=sys.stderr)
            service.stop()
    result["ok"] = not result["failures"]

    print(json.dumps(result, ensure_ascii=False, indent=2))
    markdown = summary_markdown(args.label, result)
    print(markdown)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
