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


def _gradient_at(colors: List[List[int]], pos01: float, use_hsv: bool = True) -> List[int]:
    """Get color at position in gradient with optional HSV interpolation."""
    if len(colors) == 1:
        return colors[0]

    pos01 = max(0.0, min(1.0, pos01))
    segments = len(colors) - 1
    x = pos01 * segments
    i = int(math.floor(x))
    if i >= segments:
        return colors[-1]
    t = x - i
    
    # Use HSV interpolation for more natural gradients
    if use_hsv:
        return _lerp_hsv(colors[i], colors[i + 1], t)
    else:
        return _lerp_rgb_smooth(colors[i], colors[i + 1], t)


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
        # Natural breathing effect with smooth sine wave
        period_s = max(1.0, speed)
        t = (now_s % period_s) / period_s
        
        # Smooth breathing curve using cosine
        # Creates a natural inhale-exhale pattern
        wave = (math.cos(t * 2.0 * math.pi) + 1.0) / 2.0
        
        # Apply easing for even smoother transitions
        wave = _ease_in_out_cubic(wave)
        
        # Map to brightness range with minimum floor
        min_b = 0.08
        factor = brightness * (min_b + (1.0 - min_b) * wave)
        
        # Optional: Slowly shift colors through the gradient
        color_shift = (t * 0.2) % 1.0  # Slow color rotation
        return [_apply_brightness(base_color(i, color_shift), factor) for i in range(led_count)]

    if mode == "flow":
        # Smooth color flow with HSV interpolation for natural color transitions
        period_s = max(2.0, speed)
        total_colors = len(colors)
        if total_colors < 2:
             # Static if only one color
             return [_apply_brightness(colors[0], brightness) for _ in range(led_count)]

        # Cycle through colors with smooth transitions
        t = (now_s % period_s) / period_s * total_colors
        idx = int(t) % total_colors
        next_idx = (idx + 1) % total_colors
        frac = t - int(t)
        
        # Use HSV interpolation for more natural color blending
        current_rgb = _lerp_hsv(colors[idx], colors[next_idx], frac)
        
        # Add subtle brightness wave for more dynamic effect
        wave = (math.sin(t * math.pi * 2.0) * 0.1 + 1.0)
        adjusted_brightness = brightness * wave
        
        return [_apply_brightness(current_rgb, adjusted_brightness) for _ in range(led_count)]

    if mode == "gradient":
        # Moving spatial gradient (Aurora style) with smooth motion
        period_s = max(2.0, speed)
        phase = (now_s % period_s) / period_s
        
        # Apply easing to the movement for smoother flow
        smooth_phase = _ease_in_out_cubic(phase)
        
        # Add subtle wave distortion for more organic feel
        wave_intensity = 0.05
        result = []
        for i in range(led_count):
            pos = i / max(1, led_count - 1)
            # Add wave distortion that moves with the gradient
            wave_offset = math.sin((pos * 3.0 + smooth_phase * 2.0) * math.pi) * wave_intensity
            color_pos = (pos + smooth_phase + wave_offset) % 1.0
            
            color = _gradient_at(colors, color_pos, use_hsv=True)
            
            # Subtle brightness variation for depth
            brightness_var = 1.0 + math.sin((pos * 2.0 + smooth_phase * 4.0) * math.pi) * 0.08
            result.append(_apply_brightness(color, brightness * brightness_var))
        
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
    
    if mode == "rainbow":
        # Smooth rainbow effect with natural color progression
        period_s = max(2.0, speed)
        t = (now_s % period_s) / period_s
        
        result = []
        for i in range(led_count):
            pos = i / max(1, led_count - 1)
            
            # Hue cycles through full spectrum
            hue = ((pos + t) * 360.0) % 360.0
            saturation = 1.0
            value = 1.0
            
            color = _hsv_to_rgb(hue, saturation, value)
            result.append(_apply_brightness(color, brightness))
        
        return result
    
    if mode == "fire":
        # Flickering fire effect with warm colors
        result = []
        base_time = now_s * 10.0  # Faster flicker
        
        for i in range(led_count):
            pos = i / max(1, led_count - 1)
            
            # Multiple noise frequencies for natural flicker
            flicker1 = math.sin(base_time + i * 0.5) * 0.5 + 0.5
            flicker2 = math.sin(base_time * 1.7 + i * 0.3) * 0.3 + 0.5
            flicker3 = math.sin(base_time * 2.3 + i * 0.8) * 0.2 + 0.5
            
            intensity = (flicker1 * 0.5 + flicker2 * 0.3 + flicker3 * 0.2)
            intensity = _clamp_float(intensity, low=0.3, high=1.0)
            
            # Fire colors: red-orange-yellow gradient
            if len(colors) >= 2:
                color = _lerp_hsv(colors[0], colors[1], flicker1)
            else:
                # Default fire colors
                hue = 15.0 + (flicker1 * 30.0)  # Orange to yellow
                saturation = 0.9 + (flicker2 * 0.1)
                value = intensity
                color = _hsv_to_rgb(hue, saturation, value)
            
            result.append(_apply_brightness(color, brightness * intensity))
        
        return result
    
    if mode == "sparkle":
        # Random sparkle effect with smooth fade
        import random
        random.seed(int(now_s * 1000) % 10000)
        
        sparkle_prob = 0.02  # Probability per LED
        fade_time = 0.5  # Seconds for sparkle to fade
        
        base_brightness = 0.2
        
        result = []
        for i in range(led_count):
            # Deterministic "random" based on time and position
            noise_val = math.sin(now_s * 3.0 + i * 17.3) * math.cos(now_s * 5.7 + i * 23.1)
            
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
        leds_per_s = max(1.0, speed * 8.0)  # Speed multiplier for visible movement
        points = command.get("points", 2)
        try:
            points_n = int(points)
        except Exception:
            points_n = 2
        points_n = max(1, min(8, points_n))

        # Continuous position for smooth sub-pixel movement
        head_pos = (now_s * leds_per_s) % led_count
        spacing = led_count / points_n

        head_color = colors[0]
        tail_color = colors[1] if len(colors) > 1 else colors[0]

        # Tail length for smooth fade
        tail_length = 8.0
        
        out: List[List[int]] = []
        for i in range(led_count):
            max_intensity = 0.0
            final_color = [0, 0, 0]
            
            # Check each meteor point
            for k in range(points_n):
                head = (head_pos + k * spacing) % led_count
                
                # Distance behind the head (in the direction of movement)
                dist = (head - i) % led_count
                
                if dist < tail_length:
                    # Smooth exponential fade for natural tail
                    intensity = math.exp(-dist / (tail_length * 0.4))
                    intensity = _ease_out_quad(intensity)  # Smooth the fade curve
                    
                    if intensity > max_intensity:
                        max_intensity = intensity
                        # Blend from head to tail color based on distance
                        color_t = dist / tail_length
                        final_color = _lerp_hsv(head_color, tail_color, color_t)
            
            # Apply brightness and intensity
            out.append(_apply_brightness(final_color, brightness * max_intensity))

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
