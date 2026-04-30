from __future__ import annotations

import json
import time
from typing import Any

import requests

from config_loader import get_config, get_float

AIHUBMIX_BASE_URL = "https://aihubmix.com"
DEFAULT_TIMEOUT_S = get_float("GEMINI_TIMEOUT_S", 180.0)


def build_auth_headers(api_key: str | None = None) -> dict[str, str]:
    token = api_key if api_key is not None else get_config("AIHUBMIX_API_KEY", "")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def post_chat_json(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    timeout_s: float | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    started_at = time.perf_counter()
    response = requests.post(
        f"{AIHUBMIX_BASE_URL}/v1/chat/completions",
        headers=build_auth_headers(api_key),
        json=payload,
        timeout=timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content), time.perf_counter() - started_at
