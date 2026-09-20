from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path
from collections.abc import Mapping
from typing import Any, TypeVar, get_type_hints

from ..errors import ConfigurationError
from .schema import AppConfig


T = TypeVar("T")


def _merge_dataclass(instance: T, values: dict[str, Any], path: str) -> T:
    fields = {item.name: item for item in dataclasses.fields(instance)}
    type_hints = get_type_hints(type(instance))
    for key, value in values.items():
        if key not in fields:
            raise ConfigurationError(f"unknown config key: {path}{key}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigurationError(f"{path}{key} must be a table")
            _merge_dataclass(current, value, f"{path}{key}.")
            continue
        expected = type_hints.get(key)
        if expected in {int, float, str, bool} and not isinstance(value, expected):
            if expected is float and isinstance(value, int):
                value = float(value)
            else:
                raise ConfigurationError(f"{path}{key} must be {expected.__name__}")
        setattr(instance, key, value)
    return instance


QWEN_REGION_ENV = "ECHOLINGO_QWEN_REGION"


def qwen_region_from_environment(default: str = "singapore") -> str:
    """DashScope region. Beijing is the only region with new-user free quota."""
    value = os.environ.get(QWEN_REGION_ENV, "").strip().lower()
    return value if value in {"singapore", "beijing"} else default


# Provider settings the desktop shell forwards as environment variables
# (see ``backends.registry`` ``ProviderSetting.env_var``). Each entry maps the
# variable to the config attributes it overrides; adapters resolve the
# remaining ``ECHOLINGO_*_CHAT_MODEL`` / ``*_BASE_URL`` variables themselves
# through ``registry.chat_preset_values`` so presets stay in one table.
SETTING_ENV_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    QWEN_REGION_ENV: (("asr.qwen_cloud", "region"), ("translation.qwen_cloud", "region")),
    "ECHOLINGO_OPENAI_REALTIME_MODEL": (("asr.openai_realtime", "model"),),
    "ECHOLINGO_DEEPGRAM_MODEL": (("asr.deepgram", "model"),),
    "ECHOLINGO_DEEPL_TIER": (("translation.deepl", "tier"),),
    "ECHOLINGO_AZURE_TRANSLATOR_REGION": (("translation.azure_translator", "region"),),
}


def _resolve(config: AppConfig, dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


_SETTING_CHOICES: dict[str, frozenset[str]] = {
    QWEN_REGION_ENV: frozenset({"singapore", "beijing"}),
    "ECHOLINGO_DEEPL_TIER": frozenset({"free", "pro"}),
}


def apply_environment_overrides(config: AppConfig, environ: Mapping[str, str] | None = None) -> None:
    """Apply provider settings forwarded by the desktop shell as env vars.

    Unknown values for enumerated settings are ignored so a stale preference
    can never make the configuration unloadable.
    """
    environ = os.environ if environ is None else environ
    for env_var, targets in SETTING_ENV_TARGETS.items():
        value = environ.get(env_var, "").strip()
        if not value:
            continue
        choices = _SETTING_CHOICES.get(env_var)
        if choices is not None:
            value = value.lower()
            if value not in choices:
                continue
        for dotted, attribute in targets:
            setattr(_resolve(config, dotted), attribute, value)


def load_config(path: Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is not None:
        try:
            with path.open("rb") as handle:
                values = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigurationError(f"cannot load config {path}: {error}") from error
        _merge_dataclass(config, values, "")
    apply_environment_overrides(config)
    config.validate()
    return config

