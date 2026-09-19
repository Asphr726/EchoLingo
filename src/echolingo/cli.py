from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .audio import MicrophoneSource, WavReplaySource, list_input_devices
from .config import load_config
from .enhancement import make_processor
from .farfield import generate_proxy_files
from .pipeline import FarFieldPipeline
from .runtime import BackendFactory, CapabilityDetector, RuntimeRouter
from .sinks import RunRecorder
from .streaming import LectureSpeechPolicy
from .translation import StreamingTranslationCoordinator
from .vad import make_vad


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frontend", choices=["raw", "webrtc_agc", "webrtc_ns_agc"]
    )
    parser.add_argument("--vad", choices=["auto", "silero", "webrtc", "energy"])
    parser.add_argument("--silero-model", type=Path, default=Path("models/silero_vad.onnx"))
    parser.add_argument("--language", choices=["en", "zh", "ja", "ko", "auto"])
    parser.add_argument(
        "--asr",
        choices=["none", "wlk", "qwen_local", "qwen_cloud", "simulstreaming", "mock"],
    )
    parser.add_argument("--wlk-url")
    parser.add_argument("--model-backend")
    parser.add_argument("--mode", choices=["auto", "local", "cloud"])
    parser.add_argument("--allow-audio-upload", action="store_true")
    parser.add_argument("--allow-transcript-upload", action="store_true")
    parser.add_argument(
        "--translation", choices=["none", "hymt_local", "qwen_cloud", "mock"]
    )
    parser.add_argument("--target-language", choices=["en", "zh", "ja", "ko"])
    parser.add_argument("--run-root", type=Path, default=Path("runs/spike1"))
    parser.add_argument("--duration-limit", type=float)
    parser.add_argument("--no-record-audio", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echolingo", description="EchoLingo far-field multilingual interpreter"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/lecture.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="list input devices")
    sub.add_parser("doctor", help="report runtime capabilities and selected route").add_argument(
        "--json", action="store_true"
    )

    listen = sub.add_parser("listen", help="capture and process a microphone stream")
    _add_pipeline_args(listen)
    listen.add_argument("--device")
    listen.add_argument("--channels", type=int)
    listen.add_argument("--capture-rate", type=int)
    listen.add_argument("--queue-frames", type=int)

    replay = sub.add_parser("replay", help="replay a WAV through the live pipeline")
    replay.add_argument("path", type=Path)
    replay.add_argument("--realtime", action="store_true")
    _add_pipeline_args(replay)

    benchmark = sub.add_parser("benchmark", help="run frontend comparisons")
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument(
        "--profiles", nargs="+", default=["raw", "webrtc_agc", "webrtc_ns_agc"]
    )
    benchmark.add_argument("--vad", choices=["auto", "silero", "webrtc", "energy"])
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
    return parser


def _resolve_config(args):
    config = load_config(args.config)
    if getattr(args, "frontend", None):
        config.frontend.profile = args.frontend
    if getattr(args, "vad", None):
        config.vad.backend = args.vad
    if getattr(args, "language", None):
        config.asr.language = args.language
    if getattr(args, "mode", None):
        config.inference.mode = args.mode
    if getattr(args, "asr", None):
        config.asr.provider = "qwen_local" if args.asr == "wlk" else args.asr
    if getattr(args, "wlk_url", None):
        config.asr.qwen_local.url = args.wlk_url
    if getattr(args, "model_backend", None):
        config.asr.qwen_local.lightweight_model = args.model_backend
        config.asr.local_profile = "lightweight"
    if getattr(args, "allow_audio_upload", False):
        config.privacy.audio_upload_allowed = True
    if getattr(args, "allow_transcript_upload", False):
        config.privacy.transcript_upload_allowed = True
    if getattr(args, "translation", None):
        config.translation.provider = args.translation
    if getattr(args, "target_language", None):
        config.translation.target_language = args.target_language
    config.validate()
    return config


def _route(config):
    capabilities = CapabilityDetector(Path.cwd()).detect()
    route = RuntimeRouter(config, capabilities).select()
    return capabilities, route


async def _run_source(source, args) -> Path:
    config = _resolve_config(args)
    _, route = _route(config)
    processor = make_processor(
        config.frontend.profile, source.sample_rate_hz, source.channels, config.frontend
    )
    model_path = args.silero_model if args.silero_model.exists() else None
    vad = make_vad(config.vad.backend, model_path)
    policy = LectureSpeechPolicy(
        config.vad.start_probability,
        config.vad.continue_probability,
        config.vad.min_speech_ms,
        config.vad.min_silence_ms,
    )
    asr = BackendFactory(config).asr(route.asr_provider)
    translation_backend = BackendFactory(config).translation(route.translation_provider)
    translation = None
    if translation_backend is not None:
        translation = StreamingTranslationCoordinator(
            translation_backend,
            source_lang=config.asr.language,
            target_lang=config.translation.target_language,
            context_segments=config.translation.context_segments,
        )
    resolved = config.redacted_dict()
    resolved["route"] = asdict(route)
    resolved["route"]["status"] = route.status.value
    sink = RunRecorder(
        args.run_root,
        source.sample_rate_hz,
        source.channels,
        resolved,
        record_audio=not args.no_record_audio,
        console=not args.quiet,
    )
    pipeline = FarFieldPipeline(
        processor, vad, policy, asr, sink, source.sample_rate_hz, translation
    )
    await pipeline.run(source, args.duration_limit)
    return sink.run_dir


def _doctor(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    detector = CapabilityDetector(Path.cwd())
    capabilities = detector.detect()
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "capabilities": capabilities.to_dict(),
        "config": config.redacted_dict(),
    }
    try:
        route = RuntimeRouter(config, capabilities).select()
        result["route"] = asdict(route)
        result["route"]["status"] = route.status.value
    except Exception as error:
        result["route_error"] = f"{type(error).__name__}: {error}"
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "devices":
        print(json.dumps(list_input_devices(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "doctor":
        report = _doctor(args.config)
        print(json.dumps(report, indent=2) if args.json else report)
        return 0
    if args.command == "listen":
        config = _resolve_config(args)
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        source = MicrophoneSource(
            sample_rate_hz=args.capture_rate or config.audio.capture_rate_hz,
            channels=config.audio.channels if args.channels is None else args.channels,
            device=device,
            queue_frames=args.queue_frames or config.audio.queue_frames,
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
            config = load_config(args.config)
            config.frontend.profile = profile
            config.asr.provider = "none"
            processor = make_processor(profile, 16_000, 1)
            source = WavReplaySource(args.path)
            processor = make_processor(profile, source.sample_rate_hz, source.channels)
            model_path = args.silero_model if args.silero_model.exists() else None
            vad = make_vad(args.vad or config.vad.backend, model_path)
            sink = RunRecorder(
                args.run_root / profile,
                source.sample_rate_hz,
                source.channels,
                config.redacted_dict(),
                record_audio=False,
                console=False,
            )
            pipeline = FarFieldPipeline(
                processor,
                vad,
                LectureSpeechPolicy(),
                BackendFactory(config).asr("none"),
                sink,
                source.sample_rate_hz,
            )
            asyncio.run(pipeline.run(source, args.duration_limit))
            completed.append(str(sink.run_dir))
        print(json.dumps({"runs": completed}, indent=2))
        return 0
    if args.command == "farfield-proxy":
        paths = generate_proxy_files(
            args.source,
            args.rir_archive,
            args.rir_member,
            args.output_dir,
            args.distances,
            args.snr_db,
        )
        print(json.dumps({"proxies": [str(path) for path in paths]}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
