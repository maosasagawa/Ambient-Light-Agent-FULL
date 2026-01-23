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
    - mode_options: optional dict for future extensions
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

    if mode == "pulse":
        options = command.get("mode_options") or {}
        period_s = max(0.2, float(options.get("period_s", speed)))
        duty = float(options.get("duty", 0.2))
        duty = max(0.05, min(0.95, duty))
        phase = (now_s % period_s) / period_s
        factor = brightness if phase < duty else 0.0
        return [_apply_brightness(base_color(i), factor) for i in range(led_count)]

    if mode == "breath":
        # Simulate natural breathing: Inhale (fast) -> Hold -> Exhale (slow) -> Pause
        # Cycle: 0.0 -> 1.0
        period_s = max(1.0, speed)
        t = (now_s % period_s) / period_s
        
        # Simple "natural" curve approximation
        # t: 0.0->0.4 (Inhale), 0.4->0.45 (Hold), 0.45->0.9 (Exhale), 0.9->1.0 (Pause)
        if t < 0.4:
            # Inhale: 0 -> 1 (Sine ease out)
            phase = t / 0.4
            wave = math.sin(phase * math.pi / 2)
        elif t < 0.45:
            # Hold: 1
            wave = 1.0
        elif t < 0.9:
            # Exhale: 1 -> 0 (Sine ease in-out)
            phase = (t - 0.45) / 0.45
            wave = 0.5 + 0.5 * math.cos(phase * math.pi)
        else:
            # Pause: 0
            wave = 0.0

        # Map 0..1 to min_brightness..1
        min_b = 0.05
        factor = brightness * (min_b + (1.0 - min_b) * wave)
        
        # If multiple colors, we can slowly rotate them too? 
        # For now, keep spatial gradient static, just breathe brightness.
        return [_apply_brightness(base_color(i), factor) for i in range(led_count)]

    if mode == "flow":
        # Smooth color flow: The entire strip changes color uniformly over time,
        # blending between the provided colors.
        period_s = max(2.0, speed)
        total_colors = len(colors)
        if total_colors < 2:
             # Static if only one color
             return [_apply_brightness(colors[0], brightness) for _ in range(led_count)]

        # t goes from 0..total_colors
        t = (now_s % period_s) / period_s * total_colors
        idx = int(t) % total_colors
        next_idx = (idx + 1) % total_colors
        frac = t - int(t)
        
        # Smooth blending
        current_rgb = _lerp_rgb(colors[idx], colors[next_idx], frac)
        return [_apply_brightness(current_rgb, brightness) for _ in range(led_count)]

    if mode == "gradient":
        # Moving spatial gradient (Aurora style)
        period_s = max(2.0, speed)
        # Slower, smoother movement
        phase = (now_s % period_s) / period_s
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


def frame_to_rgb565_bytes(frame: Sequence[Sequence[int]]) -> bytes:
    """Pack RGB into RGB565 (2 bytes per LED)."""
    buf = bytearray()
    for rgb in frame:
        if len(rgb) != 3:
            continue
        r = _clamp_int(int(rgb[0]))
        g = _clamp_int(int(rgb[1]))
        b = _clamp_int(int(rgb[2]))
        value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        buf.append((value >> 8) & 0xFF)
        buf.append(value & 0xFF)
    return bytes(buf)


def frame_to_rgb111_bytes(frame: Sequence[Sequence[int]]) -> bytes:
    """Pack RGB into 1-bit-per-channel (3 bits per LED)."""
    buf = bytearray()
    bit_pos = 0
    current = 0

    for rgb in frame:
        if len(rgb) != 3:
            continue
        bits = [1 if int(rgb[0]) >= 128 else 0, 1 if int(rgb[1]) >= 128 else 0, 1 if int(rgb[2]) >= 128 else 0]
        for bit in bits:
            current = (current << 1) | bit
            bit_pos += 1
            if bit_pos == 8:
                buf.append(current)
                current = 0
                bit_pos = 0

    if bit_pos:
        current = current << (8 - bit_pos)
        buf.append(current)

    return bytes(buf)
