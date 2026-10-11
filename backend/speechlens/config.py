from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_cache: dict[str, dict] = {}


def load_config(name: str) -> dict:
    if name not in _cache:
        path = _CONFIG_DIR / f"{name}.yaml"
        with open(path) as f:
            _cache[name] = yaml.safe_load(f)
    return _cache[name]


def clear_cache() -> None:
    _cache.clear()
