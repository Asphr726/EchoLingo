"""PyInstaller entry point for the EchoLingo inference sidecar."""

from echolingo.service.server import main


if __name__ == "__main__":
    raise SystemExit(main())
