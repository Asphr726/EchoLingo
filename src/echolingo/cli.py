from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .asr import NoopAsrBackend, WhisperLiveKitBackend
from .audio import MicrophoneSource, WavReplaySource, list_input_devices
from .enhancement import make_processor
from .farfield import generate_proxy_files
from .pipeline import FarFieldPipeline
from .sinks import RunRecorder
from .streaming import LectureSpeechPolicy
from .vad import make_vad


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frontend", choices=["raw", "webrtc_agc", "webrtc_ns_agc"], default="webrtc_ns_agc")
    parser.add_argument("--vad", choices=["auto", "silero", "webrtc", "energy"], default="auto")
    parser.add_argument("--silero-model", type=Path, default=Path("models/silero_vad.onnx"))
    parser.add_argument("--language", choices=["en", "zh", "ja", "ko"], default="en")
    parser.add_argument("--asr", choices=["none", "wlk"], default="none")
    parser.add_argument("--wlk-url", default="ws://127.0.0.1:8000/asr")
    parser.add_argument("--model-backend", default="qwen3-streaming-0.6b")
    parser.add_argument("--run-root", type=Path, default=Path("runs/spike1"))
    parser.add_argument("--duration-limit", type=float)
    parser.add_argument("--no-record-audio", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echolingo", description="EchoLingo far-field ASR spike")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="list input devices")

    listen = sub.add_parser("listen", help="capture and process a microphone stream")
    _add_pipeline_args(listen)
    listen.add_argument("--device")
    listen.add_argument("--channels", type=int, default=0, help="0 selects all input channels")
    listen.add_argument("--capture-rate", type=int, default=48_000)
    listen.add_argument("--queue-frames", type=int, default=500)

    replay = sub.add_parser("replay", help="replay a WAV through the live pipeline")
    replay.add_argument("path", type=Path)
    replay.add_argument("--realtime", action="store_true")
    _add_pipeline_args(replay)

    benchmark = sub.add_parser("benchmark", help="run raw/AGC/NS+AGC frontend comparisons")
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument("--profiles", nargs="+", default=["raw", "webrtc_agc", "webrtc_ns_agc"])
    benchmark.add_argument("--vad", choices=["auto", "silero", "webrtc", "energy"], default="auto")
    benchmark.add_argument("--silero-model", type=Path, default=Path("models/silero_vad.onnx"))
    benchmark.add_argument("--duration-limit", type=float)
    benchmark.add_argument("--run-root", type=Path, default=Path("runs/spike1/benchmark"))

    proxy = sub.add_parser("farfield-proxy", help="generate labeled SLR26 acoustic proxies")
    proxy.add_argument("source", type=Path)
    proxy.add_argument("--rir-archive", type=Path, default=Path("data/cache/sim_rir_16k.zip"))
    proxy.add_argument(
        "--rir-member",
        default="simulated_rirs_16k/smallroom/Room001/Room001-00001.wav",
    )
    proxy.add_argument("--distances", type=float, nargs="+", default=[0.3, 3, 5, 8])
    proxy.add_argument("--snr-db", type=float, default=10.0)
    proxy.add_argument("--output-dir", type=Path, default=Path("data/generated/farfield"))

    doctor = sub.add_parser("doctor", help="report optional runtime availability")
    doctor.add_argument("--json", action="store_true")
    return parser


def _make_asr(args):
    if args.asr == "none":
        return NoopAsrBackend()
    return WhisperLiveKitBackend(
        url=args.wlk_url,
        language=args.language,
        model_backend=args.model_backend,
        streaming_mode="bounded_recompute",
    )


async def _run_source(source, args) -> Path:
    processor = make_processor(args.frontend, source.sample_rate_hz, source.channels)
    model_path = args.silero_model if args.silero_model.exists() else None
    vad = make_vad(args.vad, model_path)
    policy = LectureSpeechPolicy()
    asr = _make_asr(args)
    config = {
        "audio": {
            "capture_rate_hz": source.sample_rate_hz,
            "asr_rate_hz": 16_000,
            "channels": source.channels,
            "frame_ms": 10,
        },
        "frontend": {"profile": args.frontend},
        "vad": {"backend": vad.name, "hard_gating": False},
        "asr": {
            "backend": asr.name,
            "language": args.language,
            "model_backend": args.model_backend,
        },
    }
    sink = RunRecorder(
        args.run_root,
        source.sample_rate_hz,
        source.channels,
        config,
        record_audio=not args.no_record_audio,
        console=not args.quiet,
    )
    pipeline = FarFieldPipeline(processor, vad, policy, asr, sink, source.sample_rate_hz)
    await pipeline.run(source, args.duration_limit)
    return sink.run_dir


def _doctor() -> dict[str, object]:
    import importlib.metadata
    import importlib.util
    import platform

    packages = {}
    for distribution, module in [
        ("numpy", "numpy"), ("sounddevice", "sounddevice"),
        ("pywebrtc-audio", "pywebrtc_audio"), ("samplerate", "samplerate"),
        ("onnxruntime", "onnxruntime"), ("whisperlivekit", "whisperlivekit"),
    ]:
        if importlib.util.find_spec(module):
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                packages[distribution] = "present"
        else:
            packages[distribution] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "devices":
        print(json.dumps(list_input_devices(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "doctor":
        report = _doctor()
        print(json.dumps(report, indent=2) if args.json else report)
        return 0
    if args.command == "listen":
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        source = MicrophoneSource(
            sample_rate_hz=args.capture_rate, channels=args.channels,
            device=device, queue_frames=args.queue_frames
        )
        try:
            run_dir = asyncio.run(_run_source(source, args))
        except KeyboardInterrupt:
            source.close()
            return 130
        print(f"Run recorded at {run_dir}")
        return 0
    if args.command == "replay":
        source = WavReplaySource(args.path, realtime=args.realtime)
        run_dir = asyncio.run(_run_source(source, args))
        print(f"Run recorded at {run_dir}")
        return 0
    if args.command == "benchmark":
        completed = []
        for profile in args.profiles:
            replay_args = argparse.Namespace(
                frontend=profile, vad=args.vad, silero_model=args.silero_model,
                language="en", asr="none", wlk_url="", model_backend="none",
                run_root=args.run_root / profile, duration_limit=args.duration_limit,
                no_record_audio=True, quiet=True,
            )
            run_dir = asyncio.run(_run_source(WavReplaySource(args.path), replay_args))
            completed.append(str(run_dir))
        print(json.dumps({"runs": completed}, indent=2))
        return 0
    if args.command == "farfield-proxy":
        paths = generate_proxy_files(
            args.source, args.rir_archive, args.rir_member, args.output_dir,
            args.distances, args.snr_db,
        )
        print(json.dumps({"proxies": [str(path) for path in paths]}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
