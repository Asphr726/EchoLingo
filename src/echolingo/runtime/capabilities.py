from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping


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

    def detect(self) -> RuntimeCapabilities:
        machine = platform.machine().lower()
        apple_silicon = platform.system() == "Darwin" and machine in {"arm64", "aarch64"}
        cuda, vram = self._cuda()
        models = {
            "qwen3-asr-0.6b": (self.project_root / "models/qwen3-asr-0.6b").is_dir(),
            "qwen3-asr-1.7b": (self.project_root / "models/qwen3-asr-1.7b").is_dir(),
            "hymt2-1.8b": (self.project_root / "models/hymt2-1.8b").is_dir(),
            "hymt2-7b": (self.project_root / "models/hymt2-7b").is_dir(),
        }
        credentials = {
            "dashscope_api_key": bool(self.environ.get("DASHSCOPE_API_KEY")),
            "dashscope_workspace_id": bool(self.environ.get("DASHSCOPE_WORKSPACE_ID")),
            "openai_api_key": bool(self.environ.get("OPENAI_API_KEY")),
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
            network_available=self.network_probe(),
            credentials=credentials,
        )

    def report_json(self) -> str:
        return json.dumps(self.detect().to_dict(), indent=2, sort_keys=True)

