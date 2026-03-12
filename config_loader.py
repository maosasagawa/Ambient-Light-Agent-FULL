from __future__ import annotations

import os
from typing import Any

_ENV_CACHE: dict[str, str] | None = None


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == value[-1]) and value.startswith(("'", '"'))):
        return value[1:-1]
    return value


def _load_env_file() -> dict[str, str]:
    env_path = os.environ.get("ENV_FILE", ".env")
    if not os.path.isabs(env_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, env_path)

    if not os.path.exists(env_path):
        return {}

    data: dict[str, str] = {}
    try:
        with open(env_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = _strip_quotes(value.strip())
                if key:
                    data[key] = value
    except Exception:
        return {}
    return data


def _get_cache() -> dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env_file()
    return _ENV_CACHE


def get_config(key: str, default: Any | None = None) -> Any:
    cache = _get_cache()
    if key in cache:
        return cache[key]
    return os.environ.get(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    value = get_config(key, None)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_int(key: str, default: int) -> int:
    value = get_config(key, None)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def get_float(key: str, default: float) -> float:
    value = get_config(key, None)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default
