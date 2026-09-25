from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .audio import MicrophoneSource, WavReplaySource, list_input_devices
from .backends import registry
from .config import load_config
from .enhancement import make_processor
from .pipeline import FarFieldPipeline
from .errors import ConfigurationError
from .runtime import BackendFactory, CapabilityDetector, RuntimeRouter
from .session_context import parse_session_context
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
        choices=["wlk", *registry.provider_ids("asr")],
    )
    parser.add_argument("--wlk-url")
    parser.add_argument("--model-backend")
    parser.add_argument("--mode", choices=["auto", "local", "cloud"])
    parser.add_argument("--allow-audio-upload", action="store_true")
    parser.add_argument("--allow-transcript-upload", action="store_true")
    parser.add_argument(
        "--translation", choices=registry.provider_ids("translation")
    )
    parser.add_argument("--target-language", choices=["en", "zh", "ja", "ko"])
    parser.add_argument(
        "--context-file",
        type=Path,
        help="UTF-8 text file with the lecture topic and terms",
    )
    parser.add_argument(
        "--glossary-file",
        type=Path,
        help="UTF-8 text file with standing 'term = translation' lines",
    )
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
    if getattr(args, "context_file", None):
        config.context.session_context = _read_text_file(args.context_file)
    if getattr(args, "glossary_file", None):
        config.context.glossary = _read_text_file(args.glossary_file)
    config.validate()
    return config


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigurationError(f"cannot read {path}: {type(error).__name__}") from error


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
    session_context = parse_session_context(
        config.context.session_context,
        config.context.glossary,
        source_language=config.asr.language,
    )
    translation = None
    if translation_backend is not None:
        translation = StreamingTranslationCoordinator(
            translation_backend,
            source_lang=config.asr.language,
            target_lang=config.translation.target_language,
            context_segments=config.translation.context_segments,
            glossary=session_context.glossary,
            domain=session_context.domain or None,
            provisional_enabled=registry.get(
                "translation", route.translation_provider
            ).streaming_partials,
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
        processor,
        vad,
        policy,
        asr,
        sink,
        source.sample_rate_hz,
        translation,
        session_context=session_context,
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
