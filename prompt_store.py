from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from config_loader import get_bool, get_config

_PROMPT_CACHE: dict[str, Any] | None = None
_PROMPT_STATE_CACHE: dict[str, Any] | None = None


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _resolve_prompt_path() -> str:
    path = get_config("PROMPT_STORE_FILE", "prompts.json")
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, path)


def _resolve_state_path() -> str:
    path = get_config("PROMPT_STATE_FILE", "prompt_state.json")
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, path)


def _ensure_state_shape(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state.get("variants"), dict):
        state["variants"] = {}
    if not isinstance(state.get("ab_test"), bool):
        state["ab_test"] = False
    return state


def load_prompt_store() -> dict[str, Any]:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    path = _resolve_prompt_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            _PROMPT_CACHE = json.load(f)
    except Exception:
        _PROMPT_CACHE = {}
    if _PROMPT_CACHE is None:
        _PROMPT_CACHE = {}
    return _PROMPT_CACHE


def save_prompt_store(store: dict[str, Any]) -> dict[str, Any]:
    global _PROMPT_CACHE
    path = _resolve_prompt_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    _PROMPT_CACHE = store
    return store


def load_prompt_state() -> dict[str, Any]:
    global _PROMPT_STATE_CACHE
    if _PROMPT_STATE_CACHE is not None:
        return _PROMPT_STATE_CACHE

    path = _resolve_state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            _PROMPT_STATE_CACHE = json.load(f)
    except Exception:
        _PROMPT_STATE_CACHE = {"variants": {}, "ab_test": False}
    if _PROMPT_STATE_CACHE is None:
        _PROMPT_STATE_CACHE = {"variants": {}, "ab_test": False}
    _PROMPT_STATE_CACHE = _ensure_state_shape(_PROMPT_STATE_CACHE)
    return _PROMPT_STATE_CACHE


def save_prompt_state(state: dict[str, Any]) -> dict[str, Any]:
    global _PROMPT_STATE_CACHE
    state = _ensure_state_shape(state)
    path = _resolve_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    _PROMPT_STATE_CACHE = state
    return state


def _select_variant(
    entry: dict[str, Any],
    key: str,
    seed: str | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    variants = entry.get("variants", []) if isinstance(entry, dict) else []
    if not variants:
        return {"id": "default", "template": ""}

    explicit_env = get_config(f"PROMPT_VARIANT_{key.upper()}", None) or get_config(
        "PROMPT_VARIANT", None
    )
    if explicit_env:
        for variant in variants:
            if str(variant.get("id")) == str(explicit_env):
                return variant

    state_variants = state.get("variants", {}) if isinstance(state, dict) else {}
    explicit_state = state_variants.get(key)
    if explicit_state:
        for variant in variants:
            if str(variant.get("id")) == str(explicit_state):
                return variant

    ab_test = get_bool("PROMPT_AB_TEST", False) or bool(state.get("ab_test"))
    if ab_test and seed:
        return _weighted_pick(variants, seed)

    return variants[0]


def _weighted_pick(variants: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    weights = [max(0.0, float(v.get("weight", 1))) for v in variants]
    total = sum(weights)
    if total <= 0:
        return variants[0]

    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % int(total * 1000)
    cumulative = 0.0
    for variant, weight in zip(variants, weights):
        cumulative += weight * 1000
        if bucket < cumulative:
            return variant
    return variants[-1]


def render_prompt_with_meta(
    key: str,
    variables: dict[str, Any],
    *,
    seed: str | None = None,
) -> dict[str, str]:
    store = load_prompt_store()
    prompts = store.get("prompts", {}) if isinstance(store, dict) else {}
    entry = prompts.get(key, {}) if isinstance(prompts, dict) else {}
    state = load_prompt_state()
    variant = _select_variant(entry, key, seed, state)
    template = str(variant.get("template", ""))
    prompt = template.format_map(_SafeDict(variables or {}))
    return {
        "prompt": prompt,
        "variant_id": str(variant.get("id", "default")),
        "template": template,
    }


def render_prompt(key: str, variables: dict[str, Any], *, seed: str | None = None) -> str:
    return render_prompt_with_meta(key, variables, seed=seed)["prompt"]
