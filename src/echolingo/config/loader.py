from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
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


def load_config(path: Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is not None:
        try:
            with path.open("rb") as handle:
                values = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigurationError(f"cannot load config {path}: {error}") from error
        _merge_dataclass(config, values, "")
    config.validate()
    return config

