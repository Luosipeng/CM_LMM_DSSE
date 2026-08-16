"""Configuration helpers with deterministic path handling."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = (config_path.parent / parent).resolve()
    return _deep_merge(_read_config(parent_path), config)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config, optional inheritance, and project-relative paths."""
    config_path = Path(path).resolve()
    config = _read_config(config_path)
    for section, key in (("data", "output_path"), ("training", "output_dir")):
        value = Path(config[section][key])
        if not value.is_absolute():
            config[section][key] = str(PROJECT_ROOT / value)
    return config
