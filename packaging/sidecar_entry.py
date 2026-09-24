"""PyInstaller entry point for EchoLingo's isolated Python processes."""

import os
import sys
from collections.abc import MutableMapping, Sequence
from types import ModuleType


def _configure_standard_streams() -> None:
    """Write UTF-8 whatever the console code page is (ANSI on Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass


# Distribution CA bundles, which include locally added (e.g. corporate) roots.
LINUX_CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu, Arch, Gentoo
    "/etc/pki/tls/certs/ca-bundle.crt",  # Fedora, RHEL
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  # CentOS, RHEL 7+
    "/etc/ssl/ca-bundle.pem",  # openSUSE
    "/etc/ssl/cert.pem",  # Alpine
)


def _configure_certificate_bundle(
    environ: MutableMapping[str, str] | None = None,
    *,
    linux_bundles: Sequence[str] = LINUX_CA_BUNDLES,
) -> None:
    """Give the frozen interpreter's OpenSSL a CA bundle.

    A frozen build still looks for the build machine's OpenSSL certificate
    directory, so ``ssl.create_default_context()`` (used by websockets for
    cloud providers) trusts nothing. httpx already uses certifi. An explicit
    SSL_CERT_FILE keeps precedence. Linux prefers the distribution bundle and
    keeps an explicit SSL_CERT_DIR; macOS and Windows use certifi.
    """
    environ = os.environ if environ is None else environ
    if not getattr(sys, "frozen", False) or environ.get("SSL_CERT_FILE"):
        return
    if sys.platform.startswith("linux"):
        for bundle in linux_bundles:
            if os.path.isfile(bundle):
                environ["SSL_CERT_FILE"] = bundle
                return
        if environ.get("SSL_CERT_DIR"):
            return
    try:
        import certifi
    except ImportError:
        return
    environ["SSL_CERT_FILE"] = certifi.where()


def _install_qwen_import_shims() -> None:
    """Keep unused Whisper JIT modules out of the hardened Qwen process.

    WhisperLiveKit 0.2.24 imports every local backend from ``core`` even when
    qwen3-streaming is selected.  The native Whisper timing module initializes
    Numba/LLVM executable memory during import, which macOS correctly rejects
    in a hardened, distributable application.  Qwen's code path never uses
    these symbols, so expose fail-closed placeholders until WhisperLiveKit
    makes those optional imports lazy upstream.
    """

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unused Whisper fallback loaded in Qwen-only runtime")

    online_asr = ModuleType("whisperlivekit.local_agreement.online_asr")
    online_asr.OnlineASRProcessor = unavailable  # type: ignore[attr-defined]
    sys.modules[online_asr.__name__] = online_asr

    whisper_online = ModuleType("whisperlivekit.local_agreement.whisper_online")
    whisper_online.backend_factory = unavailable  # type: ignore[attr-defined]
    sys.modules[whisper_online.__name__] = whisper_online

    simul_whisper = ModuleType("whisperlivekit.simul_whisper")
    simul_whisper.SimulStreamingASR = unavailable  # type: ignore[attr-defined]
    sys.modules[simul_whisper.__name__] = simul_whisper

    # qwen_asr eagerly imports its ForcedAligner even though the streaming ASR
    # runtime only needs qwen_asr.core.  nagisa 0.2.11 uses Python-2-style
    # absolute imports that cannot be represented reliably in a frozen module
    # graph.  Keep the tokenizer unavailable in this ASR-only process; the
    # separate alignment worker imports the real package without this shim.
    nagisa = ModuleType("nagisa")
    nagisa.tagging = unavailable  # type: ignore[attr-defined]
    sys.modules[nagisa.__name__] = nagisa


def main() -> int:
    _configure_standard_streams()
    if len(sys.argv) > 1 and sys.argv[1] == "watch-process":
        from echolingo.service.process_watchdog import run_child_until_parent_exit

        return run_child_until_parent_exit(sys.argv[2:])

    # After watch-process: the native child's environment stays untouched.
    _configure_certificate_bundle()
    from echolingo.runtime.ascii_paths import install_nagisa_ascii_paths

    install_nagisa_ascii_paths()
    from echolingo.service.parent_watchdog import start_parent_watchdog_from_environment

    start_parent_watchdog_from_environment()
    if len(sys.argv) > 1 and sys.argv[1] == "self-test":
        from echolingo.service.self_test import main as self_test_main

        return self_test_main(sys.argv[2:], install_qwen_shims=_install_qwen_import_shims)
    if len(sys.argv) > 1 and sys.argv[1] == "qwen-asr-server":
        _install_qwen_import_shims()
        from echolingo.service.qwen_server import main as qwen_server_main

        return qwen_server_main(sys.argv[2:])

    from echolingo.service.server import main as sidecar_main

    return sidecar_main()


if __name__ == "__main__":
    raise SystemExit(main())
