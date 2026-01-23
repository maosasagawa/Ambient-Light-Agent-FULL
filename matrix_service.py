import asyncio
import ast
import base64
import importlib
import io
import json
import multiprocessing
import os
import queue
import re
import threading
import time
from typing import Any, Awaitable, Callable

import requests
from PIL import Image

from config_loader import get_config, get_float, get_int
from prompt_store import render_prompt

# `google-genai` is optional: service should still boot without it.
# We use importlib to avoid hard import failures and namespace-package quirks.

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
API_KEY = get_config("AIHUBMIX_API_KEY", "")
AIHUBMIX_BASE_URL = "https://aihubmix.com"
DATA_FILE = "latest_led_data.json"
ANIMATION_DATA_FILE = "latest_matrix_animation.json"
ANIMATION_MODEL_ID = get_config("MATRIX_ANIMATION_MODEL", "gemini-3-flash")
ANIMATION_MAX_CODE_CHARS = get_int("MATRIX_ANIMATION_MAX_CODE_CHARS", 8000)
ANIMATION_MAX_FRAMES = get_int("MATRIX_ANIMATION_MAX_FRAMES", 3600)
ANIMATION_TIMEOUT_S = get_float("MATRIX_ANIMATION_TIMEOUT_S", 10.0)
ANIMATION_CPU_SECONDS = get_int("MATRIX_ANIMATION_CPU_SECONDS", 60)
ANIMATION_MAX_MEMORY_MB = get_int("MATRIX_ANIMATION_MAX_MEMORY_MB", 1024)
GEMINI_TIMEOUT_S = get_float("GEMINI_TIMEOUT_S", 180.0)
DEFAULT_ANIMATION_CODE = (
    "import math\n"
    "\n"
    "def render_frame(t, width, height):\n"
    "    pixels = []\n"
    "    w = max(1, int(width))\n"
    "    h = max(1, int(height))\n"
    "    half_w = max(1.0, w / 2)\n"
    "    for y in range(h):\n"
    "        row = []\n"
    "        v = 1.0 - (y / max(1, h - 1))\n"
    "        for x in range(w):\n"
    "            nx = (x - half_w) / half_w\n"
    "            core = max(0.0, 1.0 - abs(nx))\n"
    "            flicker = 0.55 + 0.45 * math.sin(t * 6.0 + x * 0.45 + y * 0.35)\n"
    "            heat = v * core * (0.6 + 0.4 * flicker)\n"
    "            heat = max(0.0, min(1.0, heat))\n"
    "            r = int(255 * min(1.0, heat * 1.4))\n"
    "            g = int(200 * min(1.0, heat * 1.1))\n"
    "            b = int(80 * heat)\n"
    "            row.append([r, g, b])\n"
    "        pixels.append(row)\n"
    "    return pixels\n"
)


def _extract_missing_dependencies(error_text: str | None) -> list[str]:
    if not error_text:
        return []
    missing = set(re.findall(r"No module named ['\"]([^'\"]+)['\"]", error_text))
    blocked = set(re.findall(r"import ['\"]([^'\"]+)['\"] is not allowed", error_text))
    blocked |= set(re.findall(r"module ['\"]([^'\"]+)['\"] is not allowed", error_text))
    return sorted(missing | blocked)


ALLOWED_ANIMATION_IMPORTS = {

    "math",
    "random",
    "colorsys",
    "itertools",
    "numpy",
    "perlin_noise",
    "scipy",
}
BLOCKED_ANIMATION_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "help",
    "__import__",
}
BLOCKED_ANIMATION_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
}

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
            with open(DATA_FILE, "w") as f:
                json.dump(savable_data, f)
        except Exception as e:
            print(f"Failed to save data: {e}")


def save_animation_to_file(payload: dict) -> None:
    with data_lock:
        try:
            with open(ANIMATION_DATA_FILE, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            print(f"Failed to save animation: {e}")


def load_animation_from_file() -> dict:
    with data_lock:
        if os.path.exists(ANIMATION_DATA_FILE):
            try:
                with open(ANIMATION_DATA_FILE, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

# Global state
latest_led_data = load_data_from_file()


def _clamp_rgb(value: Any) -> int:
    try:
        num = int(round(float(value)))
    except Exception:
        return 0
    return max(0, min(255, num))


def _normalize_pixels(pixels: Any, width: int, height: int) -> list[list[list[int]]]:
    rows: list[list[list[int]]] = []
    src_rows = pixels if isinstance(pixels, (list, tuple)) else []

    for y in range(height):
        src_row = src_rows[y] if y < len(src_rows) and isinstance(src_rows[y], (list, tuple)) else []
        row: list[list[int]] = []
        for x in range(width):
            cell = src_row[x] if x < len(src_row) and isinstance(src_row[x], (list, tuple)) else []
            r = _clamp_rgb(cell[0]) if len(cell) > 0 else 0
            g = _clamp_rgb(cell[1]) if len(cell) > 1 else 0
            b = _clamp_rgb(cell[2]) if len(cell) > 2 else 0
            row.append([r, g, b])
        rows.append(row)
    return rows


def _pixels_to_raw(pixels: list[list[list[int]]], width: int, height: int) -> bytearray:
    raw_data = bytearray(width * height * 3)
    for y in range(height):
        row = pixels[y] if y < len(pixels) else []
        for x in range(width):
            rgb = row[x] if x < len(row) else [0, 0, 0]
            idx = (y * width + x) * 3
            raw_data[idx] = _clamp_rgb(rgb[0])
            raw_data[idx + 1] = _clamp_rgb(rgb[1])
            raw_data[idx + 2] = _clamp_rgb(rgb[2])
    return raw_data


def _validate_animation_code(code: str) -> list[str]:
    errors: list[str] = []
    if not code:
        errors.append("animation code is empty")
        return errors
    if len(code) > ANIMATION_MAX_CODE_CHARS:
        errors.append("animation code too long")
        return errors

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"syntax error: {e}")
        return errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_ANIMATION_IMPORTS:
                    errors.append(f"import '{root}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").split(".")[0]
            if module_name and module_name not in ALLOWED_ANIMATION_IMPORTS:
                errors.append(f"import '{module_name}' is not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_ANIMATION_NAMES:
                errors.append(f"call '{node.func.id}' is not allowed")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in BLOCKED_ANIMATION_MODULES:
                    errors.append(f"module '{node.func.value.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in BLOCKED_ANIMATION_MODULES:
                errors.append(f"module '{node.value.id}' is not allowed")
    return errors


def _build_animation_prompt(
    instruction: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
) -> str:
    return render_prompt(
        "matrix_animation",
        {
            "instruction": instruction,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration_s,
            "allowed_imports": ", ".join(sorted(ALLOWED_ANIMATION_IMPORTS)),
        },
        seed=instruction,
    )


def generate_matrix_animation_code(
    instruction: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
) -> dict[str, Any]:
    prompt = _build_animation_prompt(instruction, width, height, fps, duration_s)
    payload = {
        "model": ANIMATION_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.6,
    }

    t0 = time.perf_counter()

    try:
        resp = requests.post(
            f"{AIHUBMIX_BASE_URL}/v1/chat/completions",
            headers=UNIFIED_API_HEADERS,
            json=payload,
            timeout=GEMINI_TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        code = str(data.get("code", "")).strip()
        summary = str(data.get("summary", "")).strip() or "已生成矩阵动画脚本"
        errors = _validate_animation_code(code)
        if errors:
            raise ValueError("; ".join(errors))
        return {
            "code": code,
            "summary": summary,
            "model_used": ANIMATION_MODEL_ID,
            "prompt_used": prompt,
            "elapsed": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        return {
            "code": DEFAULT_ANIMATION_CODE,
            "summary": "使用默认动画方案",
            "model_used": ANIMATION_MODEL_ID,
            "prompt_used": prompt,
            "elapsed": round(time.perf_counter() - t0, 3),
            "error": str(e),
        }


def _safe_import(name: str, globals_dict: dict | None = None, locals_dict: dict | None = None,
                 fromlist: tuple | None = None, level: int = 0):
    root = name.split(".")[0]
    if root not in ALLOWED_ANIMATION_IMPORTS:
        raise ImportError(f"import '{root}' is not allowed")
    return importlib.import_module(name)


def _apply_resource_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (ANIMATION_CPU_SECONDS, ANIMATION_CPU_SECONDS))
        memory_bytes = ANIMATION_MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except Exception:
        return


def _sandbox_worker(
    code: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    max_frames: int,
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        _apply_resource_limits()
        if len(code) > ANIMATION_MAX_CODE_CHARS:
            raise ValueError("animation code too long")

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
            "__import__": _safe_import,
        }
        env: dict[str, Any] = {"__builtins__": safe_builtins}
        exec(compile(code, "<matrix_animation>", "exec"), env, env)
        render_frame = env.get("render_frame")
        if not callable(render_frame):
            raise ValueError("render_frame not defined")

        fps = max(1.0, min(60.0, float(fps)))
        if duration_s <= 0:
            total_frames = max_frames
        else:
            total_frames = int(round(duration_s * fps))
            total_frames = max(1, min(max_frames, total_frames))
        start = time.monotonic()

        for i in range(total_frames):
            target_time = start + i / fps
            now = time.monotonic()
            if target_time > now:
                time.sleep(target_time - now)
            t = i / fps
            pixels = render_frame(t, width, height)
            normalized = _normalize_pixels(pixels, width, height)
            result_queue.put(
                {
                    "type": "frame",
                    "index": i,
                    "ts_ms": int(time.time() * 1000),
                    "pixels": normalized,
                }
            )

        result_queue.put({"type": "done", "frame_count": total_frames})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result_queue.put({"type": "error", "error": str(e), "traceback": tb})


def _run_sandbox_stream(
    code: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    max_frames: int,
    emit_event: Callable[[dict], None],
    stop_event: threading.Event,
    set_process: Callable[[multiprocessing.Process | None], None],
) -> None:
    loop_forever = duration_s <= 0
    frame_offset = 0
    deadline = None
    if not loop_forever:
        deadline = time.monotonic() + max(ANIMATION_TIMEOUT_S, duration_s + 5.0)

    while True:
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue = ctx.Queue()
        process = ctx.Process(
            target=_sandbox_worker,
            args=(code, width, height, fps, duration_s, max_frames, result_queue),
        )
        set_process(process)
        process.start()

        last_item: dict | None = None
        try:
            while True:
                if stop_event.is_set():
                    emit_event({"type": "stopped"})
                    return
                if deadline is not None and time.monotonic() > deadline:
                    emit_event({"type": "error", "error": "sandbox timeout"})
                    return
                try:
                    item = result_queue.get(timeout=0.2)
                except queue.Empty:
                    if not process.is_alive():
                        # Try to get any remaining error from queue
                        exit_code = process.exitcode
                        remaining_error = None
                        try:
                            remaining_item = result_queue.get_nowait()
                            if remaining_item.get("type") == "error":
                                remaining_error = remaining_item.get("error")
                                traceback_info = remaining_item.get("traceback", "")
                                if traceback_info:
                                    remaining_error = f"{remaining_error}\n{traceback_info}"
                        except queue.Empty:
                            pass
                        error_msg = remaining_error or f"sandbox exited unexpectedly (exit_code={exit_code})"
                        emit_event({"type": "error", "error": error_msg})
                        return
                    continue

                if item.get("type") == "frame" and frame_offset:
                    item = dict(item)
                    item["index"] = frame_offset + int(item.get("index", 0))

                last_item = item
                if loop_forever and item.get("type") == "done":
                    break

                emit_event(item)
                if item.get("type") in {"done", "error"}:
                    break
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            set_process(None)

        if not loop_forever:
            break
        if last_item is None:
            break
        if last_item.get("type") == "error":
            break
        if stop_event.is_set():
            emit_event({"type": "stopped"})
            break
        if last_item.get("type") == "done":
            frame_offset += int(last_item.get("frame_count", 0)) or max_frames
            continue

        break


class MatrixFrameBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue_obj: asyncio.Queue = asyncio.Queue(maxsize=2)
        async with self._lock:
            self._subscribers.add(queue_obj)
        return queue_obj

    async def unsubscribe(self, queue_obj: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue_obj)

    async def publish(self, raw: bytes | bytearray) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)

        for queue_obj in subscribers:
            if queue_obj.full():
                try:
                    queue_obj.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue_obj.put_nowait(raw)
            except asyncio.QueueFull:
                continue


class MatrixAnimationRunner:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event = threading.Event()
        self._process: multiprocessing.Process | None = None
        self._lock = asyncio.Lock()

    async def start(
        self,
        *,
        code: str,
        instruction: str,
        summary: str,
        width: int,
        height: int,
        fps: float,
        duration_s: float,
        store_frames: bool,
        model_used: str,
        on_frame: Callable[[dict], Awaitable[None]] | None = None,
        on_complete: Callable[[dict], Awaitable[None]] | None = None,
        on_fallback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        async with self._lock:
            await self._stop_locked()
            self._stop_event = threading.Event()
            self._task = asyncio.create_task(
                self._run(
                    code=code,
                    instruction=instruction,
                    summary=summary,
                    width=width,
                    height=height,
                    fps=fps,
                    duration_s=duration_s,
                    store_frames=store_frames,
                    model_used=model_used,
                    on_frame=on_frame,
                    on_complete=on_complete,
                    on_fallback=on_fallback,
                )
            )

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)
        self._process = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(
        self,
        *,
        code: str,
        instruction: str,
        summary: str,
        width: int,
        height: int,
        fps: float,
        duration_s: float,
        store_frames: bool,
        model_used: str,
        on_frame: Callable[[dict], Awaitable[None]] | None,
        on_complete: Callable[[dict], Awaitable[None]] | None,
        on_fallback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        attempt_codes = [code]
        if code.strip() != DEFAULT_ANIMATION_CODE.strip():
            attempt_codes.append(DEFAULT_ANIMATION_CODE)

        stored_frames: list[dict] = []
        status = "running"
        error: str | None = None
        attempt_errors: list[str] = []
        success_index: int | None = None

        for attempt_index, attempt_code in enumerate(attempt_codes):
            events: asyncio.Queue = asyncio.Queue()
            stored_frames = []
            status = "running"
            error = None

            def emit_event(item: dict) -> None:
                loop.call_soon_threadsafe(events.put_nowait, item)

            def set_process(process: multiprocessing.Process | None) -> None:
                self._process = process

            sandbox_task = asyncio.create_task(
                asyncio.to_thread(
                    _run_sandbox_stream,
                    attempt_code,
                    width,
                    height,
                    fps,
                    duration_s,
                    ANIMATION_MAX_FRAMES,
                    emit_event,
                    self._stop_event,
                    set_process,
                )
            )

            try:
                while True:
                    item = await events.get()
                    item_type = item.get("type")

                    if item_type == "frame":
                        pixels = _normalize_pixels(item.get("pixels"), width, height)
                        raw = _pixels_to_raw(pixels, width, height)
                        raw_bytes = bytes(raw)
                        frame_json = {
                            "width": width,
                            "height": height,
                            "pixels": pixels,
                        }
                        save_data_to_file({"raw": raw, "json": frame_json})
                        await MATRIX_FRAME_BUS.publish(raw_bytes)

                        frame_payload = {
                            "frame_index": int(item.get("index", 0)),
                            "ts_ms": int(item.get("ts_ms", time.time() * 1000)),
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "raw": raw_bytes,
                            "json": frame_json,
                        }

                        if store_frames and len(stored_frames) < ANIMATION_MAX_FRAMES:
                            stored_frames.append(
                                {
                                    "index": frame_payload["frame_index"],
                                    "ts_ms": frame_payload["ts_ms"],
                                    "raw_base64": base64.b64encode(raw_bytes).decode("utf-8"),
                                    "json": frame_json,
                                }
                            )

                        if on_frame is not None:
                            await on_frame(frame_payload)
                        continue

                    if item_type == "done":
                        status = "completed"
                        break

                    if item_type == "stopped":
                        status = "stopped"
                        break

                    if item_type == "error":
                        status = "error"
                        error = str(item.get("error"))
                        print(f"Matrix animation sandbox error: {error}")
                        break
            finally:
                await sandbox_task

            if status != "error":
                success_index = attempt_index
                break

            attempt_errors.append(error or "unknown error")

            # Notify fallback immediately before retrying with default code
            if attempt_index < len(attempt_codes) - 1 and on_fallback is not None:
                fallback_error = error or "unknown error"
                fallback_missing = _extract_missing_dependencies(fallback_error)
                print(f"[matrix_service] Fallback triggered. Error: {fallback_error}")
                print(f"[matrix_service] Failed code:\n{attempt_code}")
                await on_fallback({
                    "reason": fallback_error,
                    "missing_dependencies": fallback_missing,
                    "failed_code": attempt_code,
                })

            if attempt_index == len(attempt_codes) - 1:
                break

        fallback_used = success_index is not None and success_index > 0
        error_detail_text = attempt_errors[0] if attempt_errors else None
        if fallback_used and not error_detail_text:
            error_detail_text = "fallback used: unknown error"
        missing_deps = _extract_missing_dependencies(error_detail_text)
        error_detail = None
        if error_detail_text:
            error_detail = {
                "message": error_detail_text,
                "missing_dependencies": missing_deps,
            }

        animation_payload = {
            "status": status,
            "instruction": instruction,
            "summary": summary,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration_s,
            "model_used": model_used,
            "generated_at_ms": int(time.time() * 1000),
            "frame_count": len(stored_frames),
            "frames": stored_frames if store_frames else [],
            "error": error,
            "fallback_used": fallback_used,
            "error_detail": error_detail,
        }
        if store_frames:
            save_animation_to_file(animation_payload)
        if on_complete is not None:
            await on_complete(animation_payload)


MATRIX_FRAME_BUS = MatrixFrameBus()
MATRIX_ANIMATION_RUNNER = MatrixAnimationRunner()


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
    max_len = get_int("VOICE_PROMPT_MAX_CHARS", 500)
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
    BFL_API_KEY = get_config("BFL_API_KEY", "")
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
    poll_interval_s = get_float("FLUX_POLL_INTERVAL_S", 0.5)
    max_seconds = get_float("FLUX_POLL_MAX_SECONDS", 60.0)
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
