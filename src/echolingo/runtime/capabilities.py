from __future__ import annotations

import json
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..backends import registry


@dataclass(slots=True)
class RuntimeCapabilities:
    os: str
    architecture: str
    cpu: str
    cpu_count: int
    ram_bytes: int | None
    cuda_available: bool
    cuda_vram_bytes: int | None
    apple_silicon: bool
    metal_available: bool
    local_models: dict[str, bool] = field(default_factory=dict)
    local_runtimes: dict[str, bool] = field(default_factory=dict)
    local_services: dict[str, bool] = field(default_factory=dict)
    network_available: bool = False
    credentials: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _system_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


class CapabilityDetector:
    def __init__(
        self,
        project_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        network_probe: Callable[[], bool] | None = None,
    ) -> None:
        self.project_root = project_root or Path.cwd()
        self.environ = environ if environ is not None else os.environ
        self.network_probe = network_probe or self._default_network_probe

    @staticmethod
    def _default_network_probe() -> bool:
        try:
            socket.getaddrinfo("ap-southeast-1.maas.aliyuncs.com", 443)
            return True
        except OSError:
            return False

    @staticmethod
    def _cuda() -> tuple[bool, int | None]:
        binary = shutil.which("nvidia-smi")
        if binary is None:
            return False, None
        result = subprocess.run(
            [binary, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if result.returncode != 0:
            return False, None
        try:
            first_mb = int(result.stdout.splitlines()[0].strip())
        except (IndexError, ValueError):
            return True, None
        return True, first_mb * 1024 * 1024

    @staticmethod
    def _metal(apple_silicon: bool) -> bool:
        if not apple_silicon:
            return False
        try:
            import torch

            return bool(torch.backends.mps.is_available())
        except (ImportError, AttributeError):
            return False

    @staticmethod
    def _loopback_service(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.08) as connection:
                connection.sendall(
                    b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
                )
                response = connection.recv(64)
                return response.startswith((b"HTTP/1.1 200", b"HTTP/1.0 200"))
        except OSError:
            return False

    def _loopback_port(self, variable: str, default: int) -> int:
        try:
            port = int(self.environ.get(variable, str(default)))
        except ValueError:
            return default
        return port if 0 < port <= 65_535 else default

    def _model_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        configured = self.environ.get("ECHOLINGO_MODEL_ROOT")
        if configured:
            roots.append(Path(configured).expanduser())
        try:
            from platformdirs import user_data_path

            roots.append(user_data_path("EchoLingo") / "models")
        except ImportError:
            pass
        roots.append(self.project_root / "models")
        return tuple(dict.fromkeys(roots))

    def _has_model(self, *relative_paths: str) -> bool:
        return any(
            (root / relative_path).is_dir()
            for root in self._model_roots()
            for relative_path in relative_paths
        )

    def _local_runtimes(self) -> dict[str, bool]:
        qwen_command = self.environ.get("ECHOLINGO_QWEN_ASR_COMMAND")
        llama_command = self.environ.get("ECHOLINGO_LLAMA_SERVER")
        return {
            "qwen_asr": bool(qwen_command)
            or importlib.util.find_spec("whisperlivekit") is not None,
            "hymt": bool(llama_command)
            or shutil.which("llama-server") is not None,
        }

    def detect(self) -> RuntimeCapabilities:
        machine = platform.machine().lower()
        apple_silicon = platform.system() == "Darwin" and machine in {"arm64", "aarch64"}
        cuda, vram = self._cuda()
        models = {
            "qwen3-asr-0.6b": self._has_model("qwen3-asr-0.6b"),
            "qwen3-asr-1.7b": self._has_model("qwen3-asr-1.7b"),
            "hymt2-1.8b": self._has_model("hymt2-1.8b", "hymt2-1.8b-gguf"),
            "hymt2-7b": self._has_model("hymt2-7b", "hymt2-7b-gguf"),
        }
        credentials = {
            registry.credential_key(field_spec): bool(self.environ.get(field_spec.env_var))
            for _, field_spec in registry.credential_fields()
        }
        services = {
            "qwen_asr": self._loopback_service(
                self._loopback_port("ECHOLINGO_LOCAL_QWEN_PORT", 8000)
            ),
            "hymt": self._loopback_service(
                self._loopback_port("ECHOLINGO_LOCAL_HYMT_PORT", 8010)
            ),
        }
        return RuntimeCapabilities(
            os=platform.system(),
            architecture=machine,
            cpu=platform.processor() or platform.machine(),
            cpu_count=os.cpu_count() or 1,
            ram_bytes=_system_ram_bytes(),
            cuda_available=cuda,
            cuda_vram_bytes=vram,
            apple_silicon=apple_silicon,
            metal_available=self._metal(apple_silicon),
            local_models=models,
            local_runtimes=self._local_runtimes(),
            local_services=services,
            network_available=self.network_probe(),
            credentials=credentials,
        )

    def report_json(self) -> str:
        return json.dumps(self.detect().to_dict(), indent=2, sort_keys=True)
