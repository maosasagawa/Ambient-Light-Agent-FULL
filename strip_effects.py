import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _clamp_int(n: int, *, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(n)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c1: Sequence[int], c2: Sequence[int], t: float) -> List[int]:
    return [
        _clamp_int(round(_lerp(c1[0], c2[0], t))),
        _clamp_int(round(_lerp(c1[1], c2[1], t))),
        _clamp_int(round(_lerp(c1[2], c2[2], t))),
    ]


def _normalize_colors(colors: Any) -> List[List[int]]:
    if not isinstance(colors, list) or not colors:
        return [[0, 170, 255]]

    normalized: List[List[int]] = []
    for c in colors:
        if isinstance(c, dict) and "rgb" in c:
            c = c.get("rgb")
        if isinstance(c, list) and len(c) == 3:
            normalized.append([
                _clamp_int(int(c[0])),
                _clamp_int(int(c[1])),
                _clamp_int(int(c[2])),
            ])

    return normalized or [[0, 170, 255]]


def _gradient_at(colors: List[List[int]], pos01: float) -> List[int]:
    if len(colors) == 1:
        return colors[0]

    pos01 = max(0.0, min(1.0, pos01))
    segments = len(colors) - 1
    x = pos01 * segments
    i = int(math.floor(x))
    if i >= segments:
        return colors[-1]
    t = x - i
    return _lerp_rgb(colors[i], colors[i + 1], t)


def _apply_brightness(rgb: Sequence[int], factor: float) -> List[int]:
    factor = max(0.0, min(1.0, float(factor)))
    return [_clamp_int(round(rgb[0] * factor)), _clamp_int(round(rgb[1] * factor)), _clamp_int(round(rgb[2] * factor))]


def render_strip_frame(
    command: Dict[str, Any],
    *,
    now_s: Optional[float] = None,
    led_count: int = 60,
) -> List[List[int]]:
    """Render a single strip frame (RGB per LED).

    This is used when the server needs to provide "precomputed raw effect" frames.

    Command contract (best-effort):
    - mode: static | breath | chase | gradient
    - colors: list of RGB or list of {rgb}
    - brightness: 0..1
    - speed: mode-dependent
      - breath: seconds per cycle (default 2.0)
      - chase: LEDs per second (default 8.0)
      - gradient: seconds per cycle (default 8.0)
    """

    if now_s is None:
        now_s = time.time()

    mode = str(command.get("mode") or "static").strip().lower()
    colors = _normalize_colors(command.get("colors"))

    brightness = float(command.get("brightness", 1.0))
    brightness = max(0.0, min(1.0, brightness))

    # Defaults tuned for visual smoothness under polling.
    speed = float(command.get("speed", 1.0))
    if speed <= 0:
        speed = 1.0

    led_count = max(1, int(led_count))

    # Base gradient pattern across the strip.
    def base_color(i: int, offset01: float = 0.0) -> List[int]:
        pos01 = (i / max(1, led_count - 1))
        pos01 = (pos01 + offset01) % 1.0
        return _gradient_at(colors, pos01)

    if mode == "static":
        return [_apply_brightness(base_color(i), brightness) for i in range(led_count)]

    if mode == "breath":
        period_s = max(0.3, speed)
        phase = (now_s % period_s) / period_s
        # 0..1 smooth curve (cosine)
        wave = 0.15 + 0.85 * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
        factor = brightness * wave
        return [_apply_brightness(base_color(i), factor) for i in range(led_count)]

    if mode == "gradient":
        period_s = max(0.5, speed)
        phase = (now_s % period_s) / period_s
        # move gradient along strip
        offset01 = phase
        return [_apply_brightness(base_color(i, offset01), brightness) for i in range(led_count)]

    if mode == "chase":
        leds_per_s = max(1.0, speed)
        points = command.get("points", 3)
        try:
            points_n = int(points)
        except Exception:
            points_n = 3
        points_n = max(1, min(12, points_n))

        # Multi-point chase: several heads equally spaced.
        head = int((now_s * leds_per_s) % led_count)
        spacing = max(1, int(round(led_count / points_n)))

        head_color = colors[0]
        tail_color = colors[1] if len(colors) > 1 else colors[0]

        out: List[List[int]] = []
        for i in range(led_count):
            # Find nearest head in modular space.
            best_dist = led_count
            for k in range(points_n):
                h = (head + k * spacing) % led_count
                dist = (h - i) % led_count
                if dist < best_dist:
                    best_dist = dist

            if best_dist == 0:
                rgb = head_color
                factor = brightness
            elif best_dist == 1:
                rgb = tail_color
                factor = brightness * 0.7
            elif best_dist == 2:
                rgb = tail_color
                factor = brightness * 0.4
            elif best_dist == 3:
                rgb = tail_color
                factor = brightness * 0.18
            else:
                rgb = [0, 0, 0]
                factor = 1.0

            out.append(_apply_brightness(rgb, factor))

        return out

    # Unknown mode: fallback to static
    return [_apply_brightness(base_color(i), brightness) for i in range(led_count)]


def frame_to_raw_bytes(frame: Sequence[Sequence[int]]) -> bytes:
    buf = bytearray()
    for rgb in frame:
        if len(rgb) != 3:
            continue
        buf.extend((_clamp_int(int(rgb[0])), _clamp_int(int(rgb[1])), _clamp_int(int(rgb[2]))))
    return bytes(buf)
