"""PyInstaller entry point for EchoLingo's isolated Python processes."""

import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "qwen-asr-server":
        del sys.argv[1]
        from whisperlivekit.basic_server import main as qwen_server_main

        result = qwen_server_main()
        return int(result or 0)

    from echolingo.service.server import main as sidecar_main

    return sidecar_main()


if __name__ == "__main__":
    raise SystemExit(main())
