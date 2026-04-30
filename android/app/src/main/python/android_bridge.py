from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from typing import Any


_BOOT_LOCK = threading.Lock()
_BOOTED = False


def boot(data_dir: str | None = None, api_key: str | None = None) -> str:
    """Initialize Android-local Python runtime paths.

    The server modules are copied mostly unchanged. This bridge maps mutable JSON
    state to Android app-private storage before importing modules that capture
    file constants at import time.
    """
    global _BOOTED
    with _BOOT_LOCK:
        if data_dir:
            os.environ["HOME"] = data_dir
            os.environ["LIGHT_DATA_DIR"] = data_dir
            os.makedirs(data_dir, exist_ok=True)
            os.chdir(data_dir)
        if api_key:
            os.environ["AIHUBMIX_API_KEY"] = api_key
        _BOOTED = True
    return json.dumps({"ok": True, "home": os.environ.get("HOME", "")}, ensure_ascii=False)


def _ensure_booted() -> None:
    if not _BOOTED:
        boot(os.environ.get("HOME"))


def accept_instruction(instruction: str) -> str:
    _ensure_booted()
    import api_core

    return json.dumps(api_core.accept_instruction(instruction), ensure_ascii=False)


def generate_lighting_effect(instruction: str) -> str:
    _ensure_booted()
    import api_core

    return json.dumps(api_core.generate_lighting_effect(instruction), ensure_ascii=False)


def downsample_image(file_name: str, content_type: str, image_bytes: bytes) -> str:
    _ensure_booted()
    from PIL import Image
    from image_processor import process_image_to_led_data
    import matrix_service

    image = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
    result = process_image_to_led_data(image, (16, 16))
    matrix_service.save_data_to_file(result)
    raw = bytes(result.get("raw", b""))
    payload = {
        "json": result.get("json"),
        "raw_base64": base64.b64encode(raw).decode("utf-8"),
        "filename": file_name,
        "content_type": content_type,
    }
    return json.dumps(payload, ensure_ascii=False)


def generate_matrix_animation_code(instruction: str, width: int, height: int, fps: float, duration_s: float) -> str:
    _ensure_booted()
    import matrix_service

    return json.dumps(
        matrix_service.generate_matrix_animation_code(instruction, width, height, fps, duration_s),
        ensure_ascii=False,
    )


def render_animation_frames_threaded(
    code: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    max_frames: int = 240,
) -> str:
    """Android-compatible generated-code executor.

    The Linux backend uses multiprocessing and resource limits as a hard sandbox.
    Chaquopy does not support that model reliably, so this keeps the same
    generated-code contract (`render_frame(t, width, height)`) but executes in
    process with strict frame/code validation and bounded frame counts. Kotlin
    should call it from Dispatchers.IO and apply its own timeout watchdog.
    """
    _ensure_booted()
    import matrix_service

    errors = matrix_service._validate_animation_code(code)
    if errors:
        raise ValueError("; ".join(errors))

    fps = max(1.0, min(60.0, float(fps)))
    if duration_s <= 0:
        total_frames = max(1, min(int(max_frames), matrix_service.ANIMATION_MAX_FRAMES))
    else:
        total_frames = max(1, min(int(round(float(duration_s) * fps)), int(max_frames), matrix_service.ANIMATION_MAX_FRAMES))

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytes": bytes,
        "callable": callable,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "getattr": getattr,
        "hasattr": hasattr,
        "hash": hash,
        "hex": hex,
        "id": id,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "setattr": setattr,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "__build_class__": matrix_service.builtins.__build_class__,
        "classmethod": classmethod,
        "object": object,
        "property": property,
        "staticmethod": staticmethod,
        "super": super,
        "__import__": matrix_service._safe_import,
    }
    env = {"__builtins__": safe_builtins, "__name__": "__matrix_animation_android__"}
    exec(compile(code, "<matrix_animation_android>", "exec"), env, env)
    render_frame = env.get("render_frame")
    if not callable(render_frame):
        raise ValueError("render_frame not defined")

    frames = []
    for i in range(total_frames):
        t = i / fps
        pixels = matrix_service._normalize_pixels(render_frame(t, width, height), width, height)
        raw = bytes(matrix_service._pixels_to_raw(pixels, width, height))
        frame_json = {"width": width, "height": height, "pixels": pixels}
        if i == total_frames - 1:
            matrix_service.save_data_to_file({"raw": bytearray(raw), "json": frame_json})
        frames.append(
            {
                "index": i,
                "ts_ms": int(time.time() * 1000),
                "raw_base64": base64.b64encode(raw).decode("utf-8"),
                "json": frame_json,
            }
        )
    return json.dumps({"status": "completed", "frame_count": len(frames), "frames": frames}, ensure_ascii=False)
