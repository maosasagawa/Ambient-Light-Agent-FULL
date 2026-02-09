import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _clamp_int(n: int, *, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(n)))


def _clamp_float(n: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(n)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(t: float) -> float:
    """Smooth interpolation curve (Hermite interpolation)."""
    t = _clamp_float(t)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    """Even smoother interpolation curve (Ken Perlin's improved smoothstep)."""
    t = _clamp_float(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out for natural acceleration/deceleration."""
    t = _clamp_float(t)
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        p = 2.0 * t - 2.0
        return 1.0 + p * p * p / 2.0


def _ease_out_quad(t: float) -> float:
    """Quadratic ease-out for smooth deceleration."""
    t = _clamp_float(t)
    return 1.0 - (1.0 - t) * (1.0 - t)


def _rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV color space."""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    diff = max_c - min_c
    
    # Hue
    if diff == 0:
        h = 0.0
    elif max_c == r_norm:
        h = 60.0 * (((g_norm - b_norm) / diff) % 6.0)
    elif max_c == g_norm:
        h = 60.0 * (((b_norm - r_norm) / diff) + 2.0)
    else:
        h = 60.0 * (((r_norm - g_norm) / diff) + 4.0)
    
    # Saturation
    s = 0.0 if max_c == 0 else diff / max_c
    
    # Value
    v = max_c
    
    return (h, s, v)


def _hsv_to_rgb(h: float, s: float, v: float) -> List[int]:
    """Convert HSV to RGB color space."""
    h = h % 360.0
    s = _clamp_float(s)
    v = _clamp_float(v)
    
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return [
        _clamp_int(round((r + m) * 255)),
        _clamp_int(round((g + m) * 255)),
        _clamp_int(round((b + m) * 255))
    ]


def _lerp_rgb(c1: Sequence[int], c2: Sequence[int], t: float) -> List[int]:
    """Linear interpolation in RGB space."""
    return [
        _clamp_int(round(_lerp(c1[0], c2[0], t))),
        _clamp_int(round(_lerp(c1[1], c2[1], t))),
        _clamp_int(round(_lerp(c1[2], c2[2], t))),
    ]


def _lerp_rgb_smooth(c1: Sequence[int], c2: Sequence[int], t: float) -> List[int]:
    """Smooth interpolation in RGB space using smootherstep."""
    smooth_t = _smootherstep(t)
    return _lerp_rgb(c1, c2, smooth_t)


def _lerp_hsv(c1: Sequence[int], c2: Sequence[int], t: float) -> List[int]:
    """Interpolation in HSV color space for more natural color transitions."""
    h1, s1, v1 = _rgb_to_hsv(c1[0], c1[1], c1[2])
    h2, s2, v2 = _rgb_to_hsv(c2[0], c2[1], c2[2])
    
    # Handle hue wrapping (choose shorter path around color wheel)
    h_diff = h2 - h1
    if abs(h_diff) > 180:
        if h_diff > 0:
            h1 += 360.0
        else:
            h2 += 360.0
    
    smooth_t = _smootherstep(t)
    h = _lerp(h1, h2, smooth_t)
    s = _lerp(s1, s2, smooth_t)
    v = _lerp(v1, v2, smooth_t)
    
    return _hsv_to_rgb(h, s, v)


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


def _gradient_at(
    colors: List[List[int]],
    pos01: float,
    use_hsv: bool = False,
    loop: bool = False,
) -> List[int]:
    """Get color at position in gradient with optional HSV interpolation."""
    if len(colors) == 1:
        return colors[0]

    if loop:
        pos01 = pos01 % 1.0
        segments = len(colors)
        x = pos01 * segments
        i = int(math.floor(x)) % len(colors)
        j = (i + 1) % len(colors)
        t = x - math.floor(x)
    else:
        pos01 = max(0.0, min(1.0, pos01))
        segments = len(colors) - 1
        x = pos01 * segments
        i = int(math.floor(x))
        if i >= segments:
            return colors[-1]
        j = i + 1
        t = x - i

    if use_hsv:
        return _lerp_hsv(colors[i], colors[j], t)
    return _lerp_rgb_smooth(colors[i], colors[j], t)


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
    - mode: static | breath | chase | pulse | flow | wave | sparkle
    - colors: list of RGB or list of {rgb}
    - brightness: 0..1
    - speed: mode-dependent
      - breath/flow/wave/sparkle: seconds per cycle (larger=slower)
      - pulse: seconds per cycle (can be overridden by mode_options.period_s)
      - chase: LEDs per second (larger=faster)
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
        # Natural breathing effect with smooth sine wave
        period_s = max(2.0, speed * 2.0)
        t = (now_s % period_s) / period_s
        
        # Smooth breathing curve: longer inhale, shorter exhale
        # This feels more "alive"
        wave = math.pow(math.sin(t * math.pi), 2.5)
        
        # Map to brightness range with minimum floor
        min_b = 0.1
        factor = brightness * (min_b + (1.0 - min_b) * wave)
        
        # Slow color drift during breath
        color_shift = (t * 0.1) % 1.0
        return [
            _apply_brightness(
                _gradient_at(colors, (i / max(1, led_count - 1)) + color_shift, loop=True),
                factor,
            )
            for i in range(led_count)
        ]

    if mode == "flow":
        # Liquid color flow: colors move smoothly along the strip
        period_s = max(1.0, speed * 2.0)
        phase = (now_s % period_s) / period_s
        
        # We want a very smooth, constant velocity flow
        result = []
        for i in range(led_count):
            # pos is spatial, phase is temporal
            pos = i / max(1, led_count - 1)
            # Combine spatial and temporal for movement
            # We use a non-linear spatial mapping for "liquid" feel
            color_pos = (pos * 0.8 - phase) % 1.0
            
            # Use RGB blending for natural color fusion
            color = _gradient_at(colors, color_pos, use_hsv=False, loop=True)
            
            # Add a slow, subtle secondary wave for "shimmer"
            shimmer = math.sin(pos * 5.0 + phase * math.pi * 2.0) * 0.05 + 0.95
            result.append(_apply_brightness(color, brightness * shimmer))
        return result

    if mode == "wave":
        # Smooth wave effect with multiple frequencies
        period_s = max(2.0, speed)
        t = (now_s % period_s) / period_s * 2.0 * math.pi
        
        # Number of waves across the strip
        wave_count = command.get("wave_count", 2.0)
        try:
            wave_count = float(wave_count)
        except Exception:
            wave_count = 2.0
        wave_count = max(0.5, min(10.0, wave_count))
        
        result = []
        for i in range(led_count):
            pos = i / max(1, led_count - 1)
            
            # Primary wave
            wave1 = math.sin(pos * wave_count * 2.0 * math.pi - t)
            # Secondary wave for complexity
            wave2 = math.sin(pos * wave_count * 3.0 * math.pi - t * 1.5) * 0.3
            
            combined_wave = (wave1 + wave2) * 0.5 + 0.5
            combined_wave = _smoothstep(combined_wave)
            
            # Get color from gradient based on wave position
            color_pos = (pos + combined_wave * 0.2) % 1.0
            color = _gradient_at(colors, color_pos, use_hsv=True)
            
            # Brightness modulation by wave
            intensity = 0.3 + 0.7 * combined_wave
            result.append(_apply_brightness(color, brightness * intensity))
        
        return result
    
    if mode == "sparkle":
        # Random sparkle effect with smooth fade
        import random
        random.seed(int(now_s * 1000) % 10000)
        
        period_s = max(0.4, float(speed))
        t = now_s / period_s
        sparkle_prob = 0.02  # Probability per LED
        fade_time = 0.5  # Seconds for sparkle to fade
        
        base_brightness = 0.2
        
        result = []
        for i in range(led_count):
            # Deterministic "random" based on time and position
            noise_val = math.sin(t * 3.0 + i * 17.3) * math.cos(t * 5.7 + i * 23.1)
            
            if abs(noise_val) > (1.0 - sparkle_prob * 2.0):
                # Sparkle!
                phase = (abs(noise_val) - (1.0 - sparkle_prob * 2.0)) / (sparkle_prob * 2.0)
                sparkle_brightness = base_brightness + (1.0 - base_brightness) * _ease_out_quad(1.0 - phase)
            else:
                sparkle_brightness = base_brightness
            
            color = _gradient_at(colors, i / max(1, led_count - 1), use_hsv=True)
            result.append(_apply_brightness(color, brightness * sparkle_brightness))
        
        return result

    if mode == "chase":
        # Smooth meteor chase effect with natural tail fade
        leds_per_s = max(2.0, speed * 10.0)
        points = command.get("points", 2)
        try:
            points_n = int(points)
        except Exception:
            points_n = 2
        points_n = max(1, min(8, points_n))

        # Continuous position for smooth sub-pixel movement
        head_pos = (now_s * leds_per_s) % led_count
        spacing = led_count / points_n

        # Tail length for smooth fade
        tail_length = max(6.0, led_count / 10.0)
        
        out: List[List[int]] = []
        for i in range(led_count):
            max_intensity = 0.0
            best_color = [0, 0, 0]
            
            for k in range(points_n):
                head = (head_pos + k * spacing) % led_count
                
                # Wrapped distance calculation for smoothness
                dist = (head - i) % led_count
                
                if dist < tail_length:
                    # Normalized distance 0..1
                    d_norm = dist / tail_length
                    # Smooth exponential fade
                    intensity = math.exp(-d_norm * 4.0)
                    
                    # Add a small "glow" at the head for anti-aliasing feel
                    if dist < 1.0:
                        # Soften the head edge
                        intensity = max(intensity, _ease_out_quad(1.0 - dist))
                    
                    if intensity > max_intensity:
                        max_intensity = intensity
                        best_color = _gradient_at(colors, d_norm, use_hsv=True)
            
            # Ambient background color (very faint)
            bg_color = _apply_brightness(base_color(i), 0.05)
            final_color = _lerp_rgb(bg_color, best_color, max_intensity)
            
            out.append(_apply_brightness(final_color, brightness))

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
