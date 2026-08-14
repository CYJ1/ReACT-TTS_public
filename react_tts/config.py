"""Minimal YAML -> attribute-access config loader (no external dataclass framework needed)."""
from __future__ import annotations

import copy
from typing import Any, Dict

import yaml


class Config(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        merged = copy.deepcopy(self)
        merged.update(overrides)
        return Config(merged)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw or {})
