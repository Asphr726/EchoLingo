from __future__ import annotations

import os
from collections.abc import Mapping

from ..backends import registry
from ..config.schema import AppConfig


class BackendFactory:
    """Instantiate backends through the provider registry.

    ``environ`` carries credentials and provider settings (the desktop shell
    injects them from the OS secure store); the factory never logs it.
    """

    def __init__(self, config: AppConfig, environ: Mapping[str, str] | None = None) -> None:
        self.config = config
        self.environ = os.environ if environ is None else environ

    def asr(self, provider: str):
        spec = registry.find("asr", provider)
        if spec is None or spec.factory is None:
            raise ValueError(f"unknown ASR provider: {provider}")
        return spec.factory(self.config, self.environ)

    def translation(self, provider: str):
        spec = registry.find("translation", provider)
        if spec is None or spec.factory is None:
            raise ValueError(f"unknown translation provider: {provider}")
        return spec.factory(self.config, self.environ)
