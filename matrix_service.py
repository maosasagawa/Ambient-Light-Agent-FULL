import io
import base64
import threading
import requests
import time
import json
import os
from PIL import Image
from typing import Any

# `google-genai` is optional: service should still boot without it.
# We use importlib to avoid hard import failures and namespace-package quirks.
import importlib

genai: Any = None
types: Any = None
try:
    genai = importlib.import_module("google.genai")
    types = importlib.import_module("google.genai.types")
except Exception:  # pragma: no cover
    genai = None
    types = None

from image_processor import process_image_to_led_data

# --- Config ---
API_KEY = os.environ.get("AIHUBMIX_API_KEY", "")
AIHUBMIX_BASE_URL = "https://aihubmix.com"
DATA_FILE = "latest_led_data.json"

# --- AI Client Setup ---
genai_client = None
if genai is not None and API_KEY and "sk-" in API_KEY:
    try:
        genai_client = genai.Client(
            api_key=API_KEY,
            http_options={"base_url": f"{AIHUBMIX_BASE_URL}/gemini"},
        )
    except Exception as e:
        print(f"google-genai client init failed: {e}")

UNIFIED_API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --- Data Persistence ---
data_lock = threading.Lock()

def load_data_from_file():
    with data_lock:
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                    if 'raw' in data and 'json' in data:
                        data['raw'] = bytearray(data['raw'])
                        return data
            except Exception:
                pass
        return {"raw": bytearray(16*16*3), "json": []}

def save_data_to_file(data):
    with data_lock:
        try:
            savable_data = {
                "raw": list(data.get("raw", [])),
                "json": data.get("json", [])
            }
            with open(DATA_FILE, 'w') as f:
                json.dump(savable_data, f)
        except Exception as e:
            print(f"Failed to save data: {e}")

# Global state
latest_led_data = load_data_from_file()


def analyze_mood_and_generate_prompt(instruction: str) -> dict[str, Any]:
    """Generate a matrix-friendly scene prompt for voice/preview flows.

    This function is intentionally lightweight and does NOT generate any image.
    It is primarily used by `api_core.accept_instruction`.
    """

    text = (instruction or "").strip().replace("\n", " ")
    if not text:
        return {
            "scene_prompt": "",
            "suggested_colors": [],
            "reason": "empty instruction",
        }

    # Keep prompt bounded for safety and downstream model stability.
    max_len = int(os.environ.get("VOICE_PROMPT_MAX_CHARS", "500"))
    if len(text) > max_len:
        text = text[:max_len]

    scene_prompt = f"正常场景风格，发光主体占画面主要部分，{text}"
    reason = "为适配矩阵展示，强调正常场景并突出发光主体占比"

    return {
        "scene_prompt": scene_prompt,
        "suggested_colors": [],
        "reason": reason,
        "speakable_reason": "我会保持正常场景风格，并让发光主体成为画面焦点。",
    }


# --- Core Logic ---
def get_image_from_response(response_data: dict) -> Image.Image:
    if "data" in response_data and response_data["data"]:
        b64 = response_data["data"][0].get("b64_json")
        if b64: return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if "output" in response_data and response_data["output"]:
        if isinstance(response_data["output"], list):
            first = response_data["output"][0]
            if isinstance(first, dict) and "b64_json" in first:
                 return Image.open(io.BytesIO(base64.b64decode(first["b64_json"]))).convert("RGB")
    if "b64_json" in response_data:
        return Image.open(io.BytesIO(base64.b64decode(response_data["b64_json"]))).convert("RGB")
    raise ValueError("Cannot parse image from API response")

def _normalize_model_name(model: str) -> str:
    return (model or "").strip()


def _aspect_ratio_to_size(aspect_ratio: str, base: int = 768) -> tuple[int, int]:
    """Convert an aspect ratio (e.g. '1:1') to a reasonable width/height.

    Notes:
    - FLUX async API supports aspect_ratio 21:9 .. 9:21 (doc).
    - For our 16x16 downsample target, we keep a moderate base resolution.
    """

    try:
        w_str, h_str = aspect_ratio.split(":", 1)
        w = int(w_str)
        h = int(h_str)
    except Exception:
        w, h = 1, 1

    if w <= 0 or h <= 0:
        w, h = 1, 1

    # Clamp supported range: 21:9 .. 9:21
    if w / h > 21 / 9:
        w, h = 21, 9
    if w / h < 9 / 21:
        w, h = 9, 21

    if w >= h:
        width = base
        height = max(64, int(base * h / w))
    else:
        height = base
        width = max(64, int(base * w / h))

    return width, height


def call_flux_async(
    model: str,
    prompt: str,
    *,
    aspect_ratio: str = "1:1",
    safety_tolerance: int = 6,
) -> Image.Image:
    """FLUX async generation via BFL Official API (flux-2-flex).

    Endpoint: https://api.bfl.ai/v1/flux-2-flex
    Auth: x-key
    Flow: Submit -> Poll (polling_url)
    """

    # BFL Official Config
    BFL_API_KEY = os.environ.get("BFL_API_KEY", "")
    # Note: Using 'flux-2-flex' as requested.
    API_URL = "https://api.bfl.ai/v1/flux-2-flex"
    
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "x-key": BFL_API_KEY,
    }

    # BFL payload (no 'input' wrapper, top-level keys)
    # flux-2-flex typically requires explicit width/height
    width, height = _aspect_ratio_to_size(aspect_ratio)
    
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }

    # 1. Submit Request
    t0 = time.time()
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"[BFL Error Body]: {resp.text}")
        resp.raise_for_status()
        data = resp.json()
        print(f"[BFL] Submit took {time.time() - t0:.2f}s")
    except Exception as e:
        raise ValueError(f"BFL Submit failed: {e}")

    task_id = data.get("id")
    polling_url = data.get("polling_url")

    if not polling_url:
        raise ValueError(f"BFL async: missing polling_url in response: {resp.text}")

    # 2. Poll for Result
    poll_interval_s = float(os.environ.get("FLUX_POLL_INTERVAL_S", "0.5"))
    max_seconds = float(os.environ.get("FLUX_POLL_MAX_SECONDS", "60"))
    deadline = time.time() + max_seconds
    
    t_poll_start = time.time()

    while time.time() < deadline:
        time.sleep(poll_interval_s)
        try:
            r = requests.get(polling_url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            
            res = r.json()
            status = res.get("status")
            
            if status == "Ready":
                # Result structure: { "result": { "sample": "URL" } }
                img_url = res.get("result", {}).get("sample")
                if not img_url:
                    raise ValueError(f"BFL Ready but missing sample url: {res}")
                
                # Download Image
                t_dl = time.time()
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                
                total_poll = time.time() - t_poll_start
                print(f"[BFL] Poll+DL took {total_poll:.2f}s (DL: {time.time() - t_dl:.2f}s)")
                
                return Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            
            elif status in ("Error", "Failed", "Request Moderated"):
                raise ValueError(f"BFL Generation failed: {status} - {res}")
            
            # Pending/Processing... continue polling
            
        except Exception as e:
            # Don't crash loop on transient network error
            print(f"BFL Poll transient error: {e}")
            continue

    raise TimeoutError("BFL task timed out")


def call_image_model(model: str, prompt: str) -> Image.Image:
    normalized = _normalize_model_name(model)
    normalized_lower = normalized.lower()

    # Priority override: Always use BFL async for flux-kontext-pro/flux-2-flex requests
    # Mapping legacy model names to the new BFL function if needed
    if "flux" in normalized_lower:
         return call_flux_async("flux-2-flex", prompt, aspect_ratio="1:1")

    if normalized_lower.startswith("imagen-"):

        if not genai_client or types is None:
            raise ValueError("`google-genai` is required for imagen-* models")

        config = types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
        resp = genai_client.models.generate_images(model=normalized, prompt=prompt, config=config)
        return Image.open(io.BytesIO(resp.generated_images[0].image.image_bytes)).convert("RGB")

    # FLUX async models (two-step): flux-kontext-pro/max
    if normalized_lower in {"flux-kontext-pro", "flux-kontext-max"}:
        return call_flux_async(normalized, prompt, aspect_ratio="1:1", safety_tolerance=6)

    # FLUX one-step models (generic images/generations): FLUX.1-Kontext-pro / FLUX-1.1-pro
    if normalized_lower in {"flux.1-kontext-pro", "flux-1.1-pro"}:
        model_name = "FLUX.1-Kontext-pro" if normalized_lower == "flux.1-kontext-pro" else "FLUX-1.1-pro"
        api_url = f"{AIHUBMIX_BASE_URL}/v1/images/generations"
        payload = {"prompt": prompt, "model": model_name, "safety_tolerance": 6}
        resp = requests.post(api_url, headers=UNIFIED_API_HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        return get_image_from_response(resp.json())

    # Other/legacy models: keep existing fallback.
    api_url = f"{AIHUBMIX_BASE_URL}/v1/models/{normalized.replace('/', '%2F')}/predictions"
    payload = {"input": {"prompt": prompt}}
    resp = requests.post(api_url, headers=UNIFIED_API_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return get_image_from_response(resp.json())

def generate_matrix_data(
    prompt: str,
    model: str = "flux-kontext-pro",
    resolution: tuple = (16, 16),
) -> dict:
    """
    Main entry point for matrix generation.
    """
    global latest_led_data

    image = call_image_model(model, prompt)
    
    processed = process_image_to_led_data(image, resolution)
    latest_led_data = processed
    save_data_to_file(latest_led_data)
    
    return {
        "image": image,
        "data": processed,
        "prompt_used": prompt,
        "model_used": model,
    }
