package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/model"
	"ambient-light-agent/internal/storage"
	"ambient-light-agent/internal/ws"
	"github.com/google/uuid"
)

type animationModelResponse struct {
	Summary string `json:"summary"`
	Code    string `json:"code"`
}

type AnimationService struct {
	cfg      config.Config
	store    *storage.Store
	hub      *ws.Hub
	aihubmix *AIHubMixClient
}

func NewAnimationService(cfg config.Config, store *storage.Store, hub *ws.Hub) *AnimationService {
	return &AnimationService{
		cfg:      cfg,
		store:    store,
		hub:      hub,
		aihubmix: NewAIHubMixClientWithTimeout(cfg, cfg.ModelAnimation, cfg.AnimationAITimeoutSec),
	}
}

func (s *AnimationService) Generate(userID, prompt string, fps, durationSec, width, height int, strict bool) (model.AnimationScript, []model.MatrixFrame, error) {
	if fps <= 0 {
		fps = s.cfg.SyncFPS
	}
	infiniteDuration := durationSec <= 0
	requestedDurationSec := durationSec
	if infiniteDuration {
		requestedDurationSec = 0
	}
	renderDurationSec := resolveAnimationRenderDurationSec(requestedDurationSec)
	if width <= 0 {
		width = s.cfg.MatrixWidth
	}
	if height <= 0 {
		height = s.cfg.MatrixHeight
	}

	code := fallbackAnimationCodeByPrompt(prompt)
	generated, genErr := s.generateScriptFromModel(prompt, fps, requestedDurationSec, renderDurationSec, infiniteDuration, width, height, code)
	if strict {
		if genErr != nil {
			return model.AnimationScript{}, nil, fmt.Errorf("model generation failed: %w", genErr)
		}
		if strings.TrimSpace(generated) == "" {
			return model.AnimationScript{}, nil, errors.New("model generation returned empty script")
		}
		code = ensureExecutableAnimationCode(generated)
		if err := s.validateGeneratedPythonSyntax(code); err != nil {
			return model.AnimationScript{}, nil, fmt.Errorf("generated script syntax invalid: %w", err)
		}
	} else {
		if genErr == nil && strings.TrimSpace(generated) != "" {
			if generatedScriptFitsPrompt(generated, prompt) {
				candidate := ensureExecutableAnimationCode(generated)
				if err := s.validateGeneratedPythonSyntax(candidate); err == nil {
					code = candidate
				}
			}
		}
	}
	script := model.AnimationScript{
		ID:          uuid.NewString(),
		UserID:      userID,
		Prompt:      prompt,
		Language:    model.ScriptPython,
		Code:        code,
		FPS:         fps,
		DurationSec: requestedDurationSec,
		Width:       width,
		Height:      height,
		CreatedAt:   time.Now().UnixMilli(),
	}

	frames, err := s.runSandbox(script, renderDurationSec, infiniteDuration)
	if err != nil {
		return model.AnimationScript{}, nil, err
	}

	if len(frames) > 0 {
		last := frames[len(frames)-1]
		m := model.Matrix{
			Width:      width,
			Height:     height,
			Pixels:     last.Pixels,
			Source:     "animation-script",
			CreatedBy:  userID,
			CreatedAt:  time.Now().UnixMilli(),
			Encoding:   "rgb24",
			FrameIndex: last.Meta.FrameIndex,
		}
		_ = s.store.SetLatestMatrix(m)
	}

	if err := s.store.SetFrames(userID, frames); err != nil {
		return model.AnimationScript{}, nil, err
	}

	s.hub.BroadcastEvent(model.Event{Type: "animation.generated", Payload: map[string]interface{}{
		"script":               script,
		"frames":               len(frames),
		"infinite_duration":    infiniteDuration,
		"preview_duration_sec": renderDurationSec,
	}, CreatedAt: time.Now().UnixMilli()})

	return script, frames, nil
}

func resolveAnimationRenderDurationSec(requested int) int {
	if requested > 0 {
		return requested
	}
	return 8
}

func (s *AnimationService) Favorite(script model.AnimationScript) error {
	return s.store.AddFavoriteScript(script)
}

func (s *AnimationService) ListFavorites() []model.AnimationScript {
	return s.store.ListFavoriteScripts()
}

func defaultPythonAnimationCode() string {
	return `import json, math

def frame(width, height, idx, total):
    out = []
    t = idx / max(total - 1, 1)
    for y in range(height):
        for x in range(width):
            fx = x / max(width - 1, 1)
            fy = y / max(height - 1, 1)
            r = int(40 + 120 * fx)
            g = int(30 + 70 * (1.0 - fy))
            b = int(90 + 100 * (0.5 + 0.5 * math.sin((fx + t) * math.pi)))
            out.append({"r": max(0,min(255,r)), "g": max(0,min(255,g)), "b": max(0,min(255,b))})
    return out

def generate(width, height, fps, duration):
    total = max(1, fps * duration)
    frames = []
    for i in range(total):
        frames.append({
            "meta": {
                "timestamp_unix_ms": 0,
                "frame_index": i,
                "width": width,
                "height": height,
                "encoding": "rgb24"
            },
            "pixels": frame(width, height, i, total)
        })
    return frames

if __name__ == "__main__":
    print(json.dumps(generate(ARGS_WIDTH, ARGS_HEIGHT, ARGS_FPS, ARGS_DURATION)))`
}

func flamePythonAnimationCode() string {
	return `import json, random, math

def clamp(v):
    return max(0, min(255, int(v)))

def _value_noise(x, y, t):
    n = math.sin(x * 12.9898 + y * 78.233 + t * 37.719) * 43758.5453
    return n - math.floor(n)

def _blur2d(src, width, height):
    out = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            total = 0.0
            weight = 0.0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    nx = min(width - 1, max(0, x + ox))
                    ny = min(height - 1, max(0, y + oy))
                    w = 2.0 if (ox == 0 and oy == 0) else 1.0
                    total += src[ny][nx] * w
                    weight += w
            out[y][x] = total / max(weight, 1.0)
    return out

def frame(width, height, idx, total):
    t = idx / max(total - 1, 1)
    heat = [[0.0 for _ in range(width)] for _ in range(height)]
    basePulse = 0.75 + 0.25 * math.sin(t * 2.0 * math.pi * 1.8)

    for x in range(width):
        nx = x / max(width - 1, 1)
        source = 0.55 + 0.35 * math.sin((nx * 2.4 + t * 1.9) * math.pi)
        turbulence = 0.25 * _value_noise(nx * 4.0, 0.0, t * 6.5)
        heat[height - 1][x] = max(0.0, source + turbulence) * basePulse

    for y in range(height - 2, -1, -1):
        ny = y / max(height - 1, 1)
        for x in range(width):
            wobble = int(round(((_value_noise(x * 0.35, y * 0.22, t * 4.2) - 0.5) * 2.4)))
            px = min(width - 1, max(0, x + wobble))
            carry = heat[y + 1][px] * (0.88 - 0.22 * ny)
            curl = 0.08 * math.sin((x * 0.33) + (t * 9.0) + (ny * 7.0))
            spark = 0.04 * _value_noise(x * 0.9, y * 0.7, t * 14.0)
            heat[y][x] = max(0.0, carry + curl + spark)

    heat = _blur2d(heat, width, height)

    out = []
    for y in range(height):
        for x in range(width):
            ny = y / max(height - 1, 1)
            h = max(0.0, min(1.35, heat[y][x]))
            ember = _value_noise(x * 0.5, y * 0.5, t * 10.0)
            core = max(0.0, min(1.0, h * 1.2))
            edge = max(0.0, min(1.0, (h - 0.2) * 1.4))
            r = clamp(130 + 125 * core + 20 * ember)
            g = clamp(18 + 145 * edge)
            b = clamp(4 + 52 * max(0.0, (h - 0.58) * 1.8))
            if ny < 0.22:
                smoke = (0.22 - ny) / 0.22
                r = clamp(r * (1.0 - 0.18 * smoke))
                g = clamp(g * (1.0 - 0.10 * smoke))
                b = clamp(b + 28 * smoke)
            out.append({"r": r, "g": g, "b": b})
    return out

def generate(width, height, fps, duration):
    total = max(1, fps * duration)
    frames = []
    for i in range(total):
        frames.append({
            "meta": {
                "timestamp_unix_ms": 0,
                "frame_index": i,
                "width": width,
                "height": height,
                "encoding": "rgb24"
            },
            "pixels": frame(width, height, i, total)
        })
    return frames

if __name__ == "__main__":
    print(json.dumps(generate(ARGS_WIDTH, ARGS_HEIGHT, ARGS_FPS, ARGS_DURATION)))`
}

func fallbackAnimationCodeByPrompt(prompt string) string {
	_ = prompt
	return defaultPythonAnimationCode()
}

func isFlamePrompt(prompt string) bool {
	lp := strings.ToLower(prompt)
	return strings.Contains(lp, "火焰") || strings.Contains(lp, "火") || strings.Contains(lp, "flame") || strings.Contains(lp, "fire") || strings.Contains(lp, "燃烧") || strings.Contains(lp, "烛") || strings.Contains(lp, "蜡烛") || strings.Contains(lp, "candle")
}

func scriptLooksFlame(code string) bool {
	lc := strings.ToLower(code)
	keywords := []string{"flame", "fire", "ember", "heat", "spark", "火", "焰", "燃"}
	for _, k := range keywords {
		if strings.Contains(lc, strings.ToLower(k)) {
			return true
		}
	}
	return false
}

func generatedScriptFitsPrompt(code, prompt string) bool {
	lc := strings.ToLower(code)
	complexSignals := 0
	if strings.Contains(lc, "numpy") || strings.Contains(lc, "np.") {
		complexSignals++
	}
	if strings.Contains(lc, "perlin") || strings.Contains(lc, "gaussian_filter") || strings.Contains(lc, "scipy") {
		complexSignals++
	}
	dynamicSignals := 0
	if strings.Contains(lc, "sin(") || strings.Contains(lc, "cos(") {
		dynamicSignals++
	}
	if strings.Contains(lc, "noise") || strings.Contains(lc, "perlin") {
		dynamicSignals++
	}
	if strings.Contains(lc, " t") || strings.Contains(lc, "t,") || strings.Contains(lc, " t=") {
		dynamicSignals++
	}
	subjectSignals := 0
	for _, k := range []string{"radius", "circle", "disc", "horizon", "cloud", "wave", "sun", "moon", "mount", "silhouette", "particle", "ray", "object"} {
		if strings.Contains(lc, k) {
			subjectSignals++
		}
	}
	if strings.Contains(lc, "render_frame") {
		complexSignals++
	}
	if complexSignals < 2 || dynamicSignals < 2 || subjectSignals < 1 {
		return false
	}
	if looksGenericFlowOnly(lc) {
		return false
	}
	_ = prompt
	return true
}

func looksGenericFlowOnly(codeLower string) bool {
	flowLike := 0
	for _, k := range []string{"gradient", "wave", "sin(", "cos(", "palette", "colorpos", "hsv"} {
		if strings.Contains(codeLower, k) {
			flowLike++
		}
	}
	entityLike := 0
	for _, k := range []string{"disc", "silhouette", "plume", "skyline", "ridge", "particle", "raindrop", "snow", "lightning", "subject", "object"} {
		if strings.Contains(codeLower, k) {
			entityLike++
		}
	}
	return flowLike >= 4 && entityLike == 0
}

func scriptMatchesPromptScene(codeLower, promptLower string) bool {
	needs := []struct {
		promptKeys []string
		codeKeys   []string
	}{
		{[]string{"fire", "flame", "candle", "火", "焰", "蜡烛", "烛"}, []string{"flame", "ember", "heat", "plume", "candle", "fire"}},
		{[]string{"ocean", "sea", "wave", "water", "海", "水", "浪"}, []string{"wave", "ocean", "water", "foam", "ripple"}},
		{[]string{"rain", "storm", "thunder", "雨", "雷", "暴"}, []string{"rain", "raindrop", "lightning", "storm", "thunder"}},
		{[]string{"snow", "blizzard", "雪", "冰"}, []string{"snow", "flake", "blizzard", "frost", "particle"}},
		{[]string{"city", "neon", "urban", "城市", "霓虹"}, []string{"city", "skyline", "neon", "building", "window"}},
		{[]string{"forest", "mountain", "hill", "山", "林"}, []string{"terrain", "ridge", "silhouette", "mount", "forest"}},
		{[]string{"sun", "moon", "dawn", "sunrise", "sunset", "日", "月"}, []string{"sun", "moon", "horizon", "disc", "halo"}},
	}

	hitScene := false
	for _, rule := range needs {
		matchedPrompt := false
		for _, pk := range rule.promptKeys {
			if strings.Contains(promptLower, pk) {
				matchedPrompt = true
				hitScene = true
				break
			}
		}
		if !matchedPrompt {
			continue
		}
		for _, ck := range rule.codeKeys {
			if strings.Contains(codeLower, ck) {
				return true
			}
		}
		return false
	}

	if !hitScene {
		genericSignals := 0
		for _, k := range []string{"subject", "silhouette", "object", "particle", "horizon", "disc", "plume", "ridge", "skyline"} {
			if strings.Contains(codeLower, k) {
				genericSignals++
			}
		}
		return genericSignals >= 2
	}
	return true
}

func inferSceneAnchors(prompt string) []string {
	lp := strings.ToLower(prompt)
	anchors := make([]string, 0, 3)
	appendOnce := func(v string) {
		for _, a := range anchors {
			if a == v {
				return
			}
		}
		anchors = append(anchors, v)
	}
	if strings.Contains(lp, "sun") || strings.Contains(lp, "日") || strings.Contains(lp, "晨") || strings.Contains(lp, "dawn") || strings.Contains(lp, "sunrise") {
		appendOnce("solar disk")
		appendOnce("horizon glow")
	}
	if strings.Contains(lp, "ocean") || strings.Contains(lp, "sea") || strings.Contains(lp, "wave") || strings.Contains(lp, "水") || strings.Contains(lp, "海") {
		appendOnce("wave field")
	}
	if strings.Contains(lp, "forest") || strings.Contains(lp, "mountain") || strings.Contains(lp, "山") || strings.Contains(lp, "林") {
		appendOnce("terrain silhouette")
	}
	if strings.Contains(lp, "city") || strings.Contains(lp, "neon") || strings.Contains(lp, "城市") || strings.Contains(lp, "霓虹") {
		appendOnce("city skyline")
	}
	if strings.Contains(lp, "fire") || strings.Contains(lp, "flame") || strings.Contains(lp, "火") {
		appendOnce("flame plume")
	}
	if len(anchors) == 0 {
		appendOnce("primary subject")
		appendOnce("mid-ground structure")
	}
	appendOnce("atmospheric light layer")
	if len(anchors) > 3 {
		anchors = anchors[:3]
	}
	return anchors
}

func sceneSpecificRequirements(prompt string) string {
	anchors := inferSceneAnchors(prompt)
	return fmt.Sprintf(`
- Extract scene anchors from instruction and implement explicit visual entities (not abstract gradient-only output).
- Required anchors for this prompt: %s.
- At least one anchor must be represented by an identifiable shape/structure (disc, silhouette, plume, skyline, wave front, etc.).
- Include foreground/mid/background separation and believable environmental evolution over time.
- If output lacks identifiable entities tied to the instruction, output is invalid.`, strings.Join(anchors, ", "))
}

func (s *AnimationService) generateScriptFromModel(prompt string, fps, requestedDurationSec, renderDurationSec int, infiniteDuration bool, width, height int, currentCode string) (string, error) {
	durationHint := fmt.Sprintf("%ds", requestedDurationSec)
	if infiniteDuration {
		durationHint = fmt.Sprintf("infinite loop (return seamless loop-ready animation). Use %ds as preview window.", renderDurationSec)
	}
	systemPrompt := fmt.Sprintf(`
# Role
You are an expert LED animation engineer specializing in fluid dynamics and generative art.

# Task
Create a highly realistic, physics-based Python animation for a %dx%d LED matrix.

# Available Tools
You MUST use these libraries for realism if applicable:
- numpy: vectorized grid operations, fading trails, and gradients (highly recommended)
- perlin_noise: natural textures (clouds, fire, water, smoke)
- scipy.ndimage: diffusion/blur effects (gaussian_filter)
- Pillow / OpenCV: optional post-process if needed
- math, random, colorsys, itertools

Code MUST gracefully fallback if optional packages are unavailable (try/except import + pure-python path).

# Requirements
- Define function: render_frame(t, width, height, duration=None) -> list[list[list[int]]].
- t is float seconds.
- Return height x width list of [R,G,B] (0..255).
- Performance: prefer numpy vectorization where possible to keep CPU fast.
- Style: smooth motion, natural decay, organic textures; avoid simplistic geometric bouncing unless requested.
- Generate complete executable code, including frame generation entrypoint that prints JSON list to stdout.
- Use ARGS_WIDTH/ARGS_HEIGHT/ARGS_FPS/ARGS_DURATION as read-only runtime constants; NEVER assign to ARGS_* names.
- If duration is infinite mode, ensure seamless looping behavior.

# Scene
Instruction: "%s"
Parameters: %dx%d, fps=%d, duration=%s

# Current Animation Code
%s

# Output JSON
{
  "summary": "Brief Chinese summary of the effect logic",
  "code": "python code string"
}

Return JSON only.
`, width, height, prompt, width, height, fps, durationHint, currentCode)

	request := "Generate improved animation based on instruction and current code."
	content, err := s.aihubmix.Chat(
		systemPrompt,
		request,
		0.4,
	)
	if err != nil {
		if isModelTimeoutErr(err) {
			retryContent, retryErr := s.aihubmix.Chat(systemPrompt, "Return compact valid JSON only with summary and code. No extra prose.", 0.2)
			if retryErr == nil {
				content = retryContent
			} else {
				return "", err
			}
		} else {
			return "", err
		}
	}
	code, ok := extractModelCodeCandidate(content)
	if ok && strings.TrimSpace(code) != "" {
		return strings.TrimSpace(code), nil
	}
	return "", errors.New("model output missing usable code field")
}

func isModelTimeoutErr(err error) bool {
	if err == nil {
		return false
	}
	e := strings.ToLower(err.Error())
	return strings.Contains(e, "context deadline exceeded") || strings.Contains(e, "client.timeout") || strings.Contains(e, "timeout")
}

func semanticScenePythonAnimationCode(prompt string) string {
	p := strings.ToLower(strings.TrimSpace(prompt))
	if p == "" {
		p = "ambient cinematic scene"
	}
	qPrompt := fmt.Sprintf("%q", p)
	tmpl := `import json, math

try:
    import numpy as np
except Exception:
    np = None

try:
    from perlin_noise import PerlinNoise
except Exception:
    PerlinNoise = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

PROMPT = __PROMPT__
LP = PROMPT.lower()
_noise = PerlinNoise(octaves=3) if PerlinNoise else None

def clamp(v):
    return max(0, min(255, int(v)))

def n3(x, y, t):
    if _noise is not None:
        return float(_noise([x, y, t]))
    return math.sin(x * 7.1 + y * 11.3 + t * 3.7) * 0.5

def scene_type():
    if any(k in LP for k in ["fire", "flame", "火", "燃"]):
        return "fire"
    if any(k in LP for k in ["candle", "烛", "蜡烛"]):
        return "candle"
    if any(k in LP for k in ["ocean", "sea", "wave", "water", "海", "水", "浪"]):
        return "ocean"
    if any(k in LP for k in ["rain", "storm", "thunder", "雨", "雷", "暴"]):
        return "rain"
    if any(k in LP for k in ["snow", "blizzard", "雪", "冰"]):
        return "snow"
    if any(k in LP for k in ["forest", "mountain", "hill", "山", "林", "雾"]):
        return "terrain"
    if any(k in LP for k in ["city", "neon", "urban", "城市", "霓虹", "街"]):
        return "city"
    if any(k in LP for k in ["sunrise", "sunset", "dawn", "sun", "moon", "日出", "日落", "太阳", "月亮"]):
        return "solar"
    return "cinematic"

def render_frame(t, width, height):
    kind = scene_type()
    prog = min(1.0, max(0.0, t / 4.5))
    sun_x = 0.5 + 0.06 * math.sin(t * 0.24)
    sun_y = 0.72 - 0.40 * prog
    sun_r = 0.11 + 0.02 * math.sin(t * 0.52)

    if np is not None:
        ys, xs = np.mgrid[0:height, 0:width]
        nx = xs / max(width - 1, 1)
        ny = ys / max(height - 1, 1)

        palettes = {
            "solar": ((20, 35, 80), (190, 110, 50)),
            "fire": ((8, 8, 16), (170, 58, 20)),
            "candle": ((10, 8, 14), (120, 76, 28)),
            "ocean": ((10, 28, 70), (12, 90, 150)),
            "rain": ((12, 22, 48), (35, 72, 120)),
            "snow": ((24, 34, 58), (96, 122, 150)),
            "terrain": ((14, 24, 42), (95, 115, 95)),
            "city": ((10, 10, 22), (45, 42, 65)),
            "cinematic": ((18, 24, 58), (110, 78, 92)),
        }
        (tr, tg, tb), (br, bg, bb) = palettes.get(kind, palettes["cinematic"])
        top_r = tr + 35 * prog
        top_g = tg + 42 * prog
        top_b = tb + 20 * prog
        bot_r = br + 34 * prog
        bot_g = bg + 26 * prog
        bot_b = bb + 14 * prog

        sky_mix = np.clip((ny - 0.05) / 0.9, 0.0, 1.0)
        r = top_r * (1.0 - sky_mix) + bot_r * sky_mix
        g = top_g * (1.0 - sky_mix) + bot_g * sky_mix
        b = top_b * (1.0 - sky_mix) + bot_b * sky_mix

        dist = np.sqrt((nx - sun_x) ** 2 + (ny - sun_y) ** 2)
        sun_disc = np.clip(1.0 - dist / max(sun_r, 1e-4), 0.0, 1.0) ** 1.7
        halo = np.exp(-((dist / max(sun_r * 3.2, 1e-4)) ** 2))
        horizon = np.clip(1.0 - np.abs(ny - 0.72) * 4.0, 0.0, 1.0)
        haze = np.clip(horizon * (0.45 + 0.55 * prog), 0.0, 1.0)

        cloud = np.zeros((height, width), dtype=float)
        for yy in range(height):
            for xx in range(width):
                cx = xx / max(width - 1, 1)
                cy = yy / max(height - 1, 1)
                cloud[yy, xx] = 0.5 + 0.5 * n3(cx * 1.8 + t * 0.035, cy * 2.7, t * 0.12)
        cloud = np.clip((cloud - 0.53) * 2.8, 0.0, 1.0)
        cloud *= np.clip(1.0 - np.abs(ny - 0.45) * 2.3, 0.0, 1.0)

        angle = np.arctan2(ny - sun_y, nx - sun_x)
        ray = (np.sin(angle * 10.0 + t * 1.4) * 0.5 + 0.5)
        ray *= np.clip(1.0 - dist / max(sun_r * 4.2, 1e-4), 0.0, 1.0)

        if kind == "fire":
            plume = np.clip(1.0 - ny * 1.25, 0.0, 1.0) * (0.55 + 0.45 * np.sin((nx * 3.4 + t * 2.1) * math.pi))
            r = r + 140 * plume + 40 * halo
            g = g + 72 * plume + 20 * haze
            b = b + 18 * plume
        elif kind == "candle":
            core_dx = nx - (0.5 + 0.03 * np.sin(t * 7.2))
            core_dy = ny - (0.7 + 0.02 * np.sin(t * 9.1))
            wick = np.clip(1.0 - np.sqrt(core_dx * core_dx + core_dy * core_dy) / 0.22, 0.0, 1.0)
            flicker = 0.65 + 0.35 * np.sin(t * 13.0 + nx * 8.0)
            glow = np.clip(1.0 - np.sqrt((nx - 0.5) ** 2 + (ny - 0.7) ** 2) / 0.55, 0.0, 1.0)
            r = r + 160 * wick * flicker + 75 * glow
            g = g + 98 * wick * flicker + 45 * glow
            b = b + 18 * wick + 8 * glow
        elif kind == "ocean":
            wave = 0.5 + 0.5 * np.sin((nx * 6.0 - t * 2.4) * math.pi + ny * 3.5)
            r = r + 20 * haze + 18 * ray
            g = g + 36 * wave + 20 * haze
            b = b + 70 * wave + 26 * halo
        elif kind == "rain":
            drops = np.clip(0.5 + 0.5 * np.sin((nx * 26.0 + ny * 8.0 - t * 10.5) * math.pi), 0.0, 1.0)
            streak = np.clip(1.0 - np.abs(np.mod(nx * 7.0 + t * 2.2, 1.0) - 0.5) * 7.5, 0.0, 1.0)
            rain_field = np.clip(drops * 0.7 + streak * 0.6, 0.0, 1.0)
            lightning = np.clip(np.sin(t * 1.9) * 0.5 + 0.5, 0.0, 1.0) ** 18
            r = r + 35 * rain_field + 90 * lightning
            g = g + 48 * rain_field + 100 * lightning
            b = b + 75 * rain_field + 120 * lightning
        elif kind == "snow":
            flakes = np.clip(0.5 + 0.5 * np.sin((nx * 19.0 + ny * 14.0 - t * 3.2) * math.pi), 0.0, 1.0)
            drift = np.clip(0.5 + 0.5 * np.sin((nx * 4.2 + t * 0.9) * math.pi), 0.0, 1.0)
            snow_field = np.clip(0.4 * flakes + 0.6 * drift, 0.0, 1.0)
            r = r + 85 * snow_field + 20 * haze
            g = g + 95 * snow_field + 24 * haze
            b = b + 105 * snow_field + 30 * haze
        elif kind == "terrain":
            ridge = np.clip(0.62 - ny + 0.08 * np.sin(nx * 7.0 + t * 0.5), 0.0, 1.0)
            r = r + 30 * haze + 18 * sun_disc - 60 * ridge
            g = g + 32 * haze + 15 * sun_disc + 20 * ridge
            b = b + 10 * haze - 48 * ridge
        elif kind == "city":
            bars = np.mod(np.floor(nx * max(width // 2, 4)), 2)
            skyline = np.clip((0.88 - ny) * 5.0, 0.0, 1.0) * bars
            neon = 0.5 + 0.5 * np.sin(nx * 20.0 + t * 6.0)
            r = r + 30 * skyline + 24 * neon
            g = g + 24 * skyline + 18 * neon
            b = b + 58 * skyline + 26 * neon
        else:
            r = r + 145 * sun_disc + 90 * halo + 65 * haze + 28 * ray
            g = g + 110 * sun_disc + 75 * halo + 45 * haze + 18 * ray
            b = b + 35 * sun_disc + 28 * halo + 10 * haze

        occl = 1.0 - 0.42 * cloud
        r, g, b = r * occl, g * occl, b * occl

        if gaussian_filter is not None:
            r = gaussian_filter(r, sigma=0.55)
            g = gaussian_filter(g, sigma=0.55)
            b = gaussian_filter(b, sigma=0.55)

        rgb = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1)
        return rgb.astype(int).tolist()

    out = []
    for y in range(height):
        row = []
        ny = y / max(height - 1, 1)
        for x in range(width):
            nx = x / max(width - 1, 1)
            base = {
                "solar": ((20, 35, 80), (190, 110, 50)),
                "fire": ((8, 8, 16), (170, 58, 20)),
                "candle": ((10, 8, 14), (120, 76, 28)),
                "ocean": ((10, 28, 70), (12, 90, 150)),
                "rain": ((12, 22, 48), (35, 72, 120)),
                "snow": ((24, 34, 58), (96, 122, 150)),
                "terrain": ((14, 24, 42), (95, 115, 95)),
                "city": ((10, 10, 22), (45, 42, 65)),
                "cinematic": ((18, 24, 58), (110, 78, 92)),
            }
            (tr, tg, tb), (br, bg, bb) = base.get(kind, base["cinematic"])
            top_r = tr + 35 * prog
            top_g = tg + 42 * prog
            top_b = tb + 20 * prog
            bot_r = br + 34 * prog
            bot_g = bg + 26 * prog
            bot_b = bb + 14 * prog

            sky_mix = max(0.0, min(1.0, (ny - 0.05) / 0.9))
            r = top_r * (1.0 - sky_mix) + bot_r * sky_mix
            g = top_g * (1.0 - sky_mix) + bot_g * sky_mix
            b = top_b * (1.0 - sky_mix) + bot_b * sky_mix

            dx, dy = nx - sun_x, ny - sun_y
            dist = math.sqrt(dx * dx + dy * dy)
            sun_disc = max(0.0, min(1.0, 1.0 - dist / max(sun_r, 1e-4))) ** 1.7
            halo = math.exp(-((dist / max(sun_r * 3.2, 1e-4)) ** 2))
            horizon = max(0.0, 1.0 - abs(ny - 0.72) * 4.0)
            haze = horizon * (0.45 + 0.55 * prog)

            cloud = 0.5 + 0.5 * n3(nx * 1.8 + t * 0.035, ny * 2.7, t * 0.12)
            cloud = max(0.0, min(1.0, (cloud - 0.53) * 2.8))
            cloud *= max(0.0, 1.0 - abs(ny - 0.45) * 2.3)

            angle = math.atan2(dy, dx)
            ray = (math.sin(angle * 10.0 + t * 1.4) * 0.5 + 0.5)
            ray *= max(0.0, 1.0 - dist / max(sun_r * 4.2, 1e-4))

            if kind == "fire":
                plume = max(0.0, min(1.0, 1.0 - ny * 1.25)) * (0.55 + 0.45 * math.sin((nx * 3.4 + t * 2.1) * math.pi))
                r = r + 140 * plume + 40 * halo
                g = g + 72 * plume + 20 * haze
                b = b + 18 * plume
            elif kind == "candle":
                core_dx = nx - (0.5 + 0.03 * math.sin(t * 7.2))
                core_dy = ny - (0.7 + 0.02 * math.sin(t * 9.1))
                wick = max(0.0, min(1.0, 1.0 - math.sqrt(core_dx * core_dx + core_dy * core_dy) / 0.22))
                flicker = 0.65 + 0.35 * math.sin(t * 13.0 + nx * 8.0)
                glow = max(0.0, min(1.0, 1.0 - math.sqrt((nx - 0.5) ** 2 + (ny - 0.7) ** 2) / 0.55))
                r = r + 160 * wick * flicker + 75 * glow
                g = g + 98 * wick * flicker + 45 * glow
                b = b + 18 * wick + 8 * glow
            elif kind == "ocean":
                wave = 0.5 + 0.5 * math.sin((nx * 6.0 - t * 2.4) * math.pi + ny * 3.5)
                r = r + 20 * haze + 18 * ray
                g = g + 36 * wave + 20 * haze
                b = b + 70 * wave + 26 * halo
            elif kind == "rain":
                drops = max(0.0, min(1.0, 0.5 + 0.5 * math.sin((nx * 26.0 + ny * 8.0 - t * 10.5) * math.pi)))
                streak = max(0.0, min(1.0, 1.0 - abs(((nx * 7.0 + t * 2.2) % 1.0) - 0.5) * 7.5))
                rain_field = max(0.0, min(1.0, drops * 0.7 + streak * 0.6))
                lightning = max(0.0, min(1.0, math.sin(t * 1.9) * 0.5 + 0.5)) ** 18
                r = r + 35 * rain_field + 90 * lightning
                g = g + 48 * rain_field + 100 * lightning
                b = b + 75 * rain_field + 120 * lightning
            elif kind == "snow":
                flakes = max(0.0, min(1.0, 0.5 + 0.5 * math.sin((nx * 19.0 + ny * 14.0 - t * 3.2) * math.pi)))
                drift = max(0.0, min(1.0, 0.5 + 0.5 * math.sin((nx * 4.2 + t * 0.9) * math.pi)))
                snow_field = max(0.0, min(1.0, 0.4 * flakes + 0.6 * drift))
                r = r + 85 * snow_field + 20 * haze
                g = g + 95 * snow_field + 24 * haze
                b = b + 105 * snow_field + 30 * haze
            elif kind == "terrain":
                ridge = max(0.0, min(1.0, 0.62 - ny + 0.08 * math.sin(nx * 7.0 + t * 0.5)))
                r = r + 30 * haze + 18 * sun_disc - 60 * ridge
                g = g + 32 * haze + 15 * sun_disc + 20 * ridge
                b = b + 10 * haze - 48 * ridge
            elif kind == "city":
                bars = int(nx * max(width // 2, 4)) % 2
                skyline = max(0.0, min(1.0, (0.88 - ny) * 5.0)) * bars
                neon = 0.5 + 0.5 * math.sin(nx * 20.0 + t * 6.0)
                r = r + 30 * skyline + 24 * neon
                g = g + 24 * skyline + 18 * neon
                b = b + 58 * skyline + 26 * neon
            else:
                r = r + 145 * sun_disc + 90 * halo + 65 * haze + 28 * ray
                g = g + 110 * sun_disc + 75 * halo + 45 * haze + 18 * ray
                b = b + 35 * sun_disc + 28 * halo + 10 * haze
            r *= (1.0 - 0.42 * cloud)
            g *= (1.0 - 0.42 * cloud)
            b *= (1.0 - 0.42 * cloud)

            row.append([clamp(r), clamp(g), clamp(b)])
        out.append(row)
    return out

def _normalize_pixels(frame_pixels, width, height):
    out = []
    for y in range(height):
        for x in range(width):
            p = frame_pixels[y][x]
            out.append({"r": clamp(p[0]), "g": clamp(p[1]), "b": clamp(p[2])})
    return out

def generate(width, height, fps, duration):
    total = max(1, int(fps * duration))
    frames = []
    for i in range(total):
        t = i / max(fps, 1)
        pixels = render_frame(t, width, height)
        frames.append({
            "meta": {
                "timestamp_unix_ms": 0,
                "frame_index": i,
                "width": width,
                "height": height,
                "encoding": "rgb24"
            },
            "pixels": _normalize_pixels(pixels, width, height)
        })
    return frames

if __name__ == "__main__":
    print(json.dumps(generate(ARGS_WIDTH, ARGS_HEIGHT, ARGS_FPS, ARGS_DURATION)))`
	return strings.ReplaceAll(tmpl, "__PROMPT__", qPrompt)
}

func (s *AnimationService) runSandbox(script model.AnimationScript, renderDurationSec int, infiniteDuration bool) ([]model.MatrixFrame, error) {
	tmpDir, err := os.MkdirTemp("", "ambient-script-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	pyFile := filepath.Join(tmpDir, "anim.py")
	loopFlag := 0
	if infiniteDuration {
		loopFlag = 1
	}
	prelude := fmt.Sprintf("ARGS_WIDTH=%d\nARGS_HEIGHT=%d\nARGS_FPS=%d\nARGS_DURATION=%d\nARGS_LOOP=%d\n", script.Width, script.Height, script.FPS, renderDurationSec, loopFlag)
	code := prelude + bytes.NewBufferString(script.Code).String()

	if err := os.WriteFile(pyFile, []byte(code), 0o600); err != nil {
		return nil, err
	}

	timeoutSec := resolveSandboxTimeoutSec(s.cfg.ScriptTimeoutSec, script.Width, script.Height, script.FPS, renderDurationSec)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, s.cfg.ScriptSandboxPython, "-I", pyFile)
	cmd.Dir = tmpDir
	cmd.Env = []string{"PATH=/usr/bin:/bin"}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("sandbox execution timed out after %ds", timeoutSec)
		}
		return nil, fmt.Errorf("sandbox run failed: %w: %s", err, truncate(stderr.String(), 300))
	}

	var frames []model.MatrixFrame
	if err := json.Unmarshal(stdout.Bytes(), &frames); err != nil {
		return nil, fmt.Errorf("invalid sandbox output: %w", err)
	}

	now := time.Now().UnixMilli()
	for i := range frames {
		frames[i].Meta.TimestampUnixMS = now + int64((1000/script.FPS)*i)
		if len(frames[i].Pixels) != script.Width*script.Height {
			return nil, errors.New("invalid frame pixel count")
		}
	}
	return frames, nil
}

func resolveSandboxTimeoutSec(base, width, height, fps, durationSec int) int {
	if base <= 0 {
		base = 8
	}
	if base < 8 {
		base = 8
	}
	frameCount := fps * durationSec
	if frameCount <= 0 {
		frameCount = 1
	}
	pixelWork := width * height * frameCount
	adaptive := 6 + pixelWork/25000
	if adaptive < base {
		adaptive = base
	}
	if adaptive > 60 {
		adaptive = 60
	}
	return adaptive
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max]
}

func stripCodeFence(s string) string {
	t := strings.TrimSpace(s)
	t = strings.TrimPrefix(t, "```python")
	t = strings.TrimPrefix(t, "```")
	t = strings.TrimSuffix(t, "```")
	return strings.TrimSpace(t)
}

func parseAnimationModelResponse(content string) (animationModelResponse, bool) {
	t := strings.TrimSpace(content)
	t = strings.TrimPrefix(t, "```json")
	t = strings.TrimPrefix(t, "```")
	t = strings.TrimSuffix(t, "```")
	t = strings.TrimSpace(t)

	obj := extractJSONObjectString(t)
	var out animationModelResponse
	if err := json.Unmarshal([]byte(obj), &out); err != nil {
		return animationModelResponse{}, false
	}
	return out, true
}

var looseCodeFieldPattern = regexp.MustCompile(`"code"\s*:\s*"((?:\\.|[^"\\])*)"`)

func extractCodeFieldFromLooseJSON(content string) (string, bool) {
	t := strings.TrimSpace(content)
	if t == "" {
		return "", false
	}
	m := looseCodeFieldPattern.FindStringSubmatch(t)
	if len(m) < 2 {
		obj := extractJSONObjectString(t)
		m = looseCodeFieldPattern.FindStringSubmatch(obj)
		if len(m) < 2 {
			return "", false
		}
	}
	unquoted, err := strconv.Unquote(`"` + m[1] + `"`)
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(unquoted), true
}

func extractCodeFieldByScanner(content string) (string, bool) {
	idx := strings.Index(content, `"code"`)
	if idx < 0 {
		return "", false
	}
	rest := content[idx+len(`"code"`):]
	colon := strings.Index(rest, ":")
	if colon < 0 {
		return "", false
	}
	rest = strings.TrimSpace(rest[colon+1:])
	if rest == "" || rest[0] != '"' {
		return "", false
	}
	var b strings.Builder
	escaped := false
	for i := 1; i < len(rest); i++ {
		ch := rest[i]
		if escaped {
			b.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' {
			escaped = true
			b.WriteByte(ch)
			continue
		}
		if ch == '"' {
			raw := b.String()
			if uq, err := strconv.Unquote(`"` + raw + `"`); err == nil {
				return strings.TrimSpace(uq), true
			}
			raw = strings.ReplaceAll(raw, `\\n`, "\n")
			raw = strings.ReplaceAll(raw, `\\t`, "\t")
			raw = strings.ReplaceAll(raw, `\\"`, `"`)
			return strings.TrimSpace(raw), true
		}
		b.WriteByte(ch)
	}
	return "", false
}

func cleanupModelCodeCandidate(code string) string {
	t := strings.TrimSpace(code)
	if t == "" {
		return ""
	}
	if strings.HasPrefix(t, `{"`) && strings.Contains(t, `"code"`) {
		return ""
	}
	if len(t) >= 2 && t[0] == '"' && t[len(t)-1] == '"' {
		if uq, err := strconv.Unquote(t); err == nil {
			t = strings.TrimSpace(uq)
		}
	}
	t = strings.ReplaceAll(t, `print(json.dumps(output))"`, `print(json.dumps(output))`)
	t = strings.ReplaceAll(t, `print(json.dumps(_generate_frames(ARGS_WIDTH, ARGS_HEIGHT, ARGS_FPS, ARGS_DURATION)))"`, `print(json.dumps(_generate_frames(ARGS_WIDTH, ARGS_HEIGHT, ARGS_FPS, ARGS_DURATION)))`)
	return strings.TrimSpace(t)
}

func extractModelCodeCandidate(content string) (string, bool) {
	if parsed, ok := parseAnimationModelResponse(content); ok && strings.TrimSpace(parsed.Code) != "" {
		if c := cleanupModelCodeCandidate(parsed.Code); c != "" {
			return c, true
		}
	}
	if extracted, ok := extractCodeFieldFromLooseJSON(content); ok && strings.TrimSpace(extracted) != "" {
		if c := cleanupModelCodeCandidate(extracted); c != "" {
			return c, true
		}
	}
	if scanned, ok := extractCodeFieldByScanner(content); ok && strings.TrimSpace(scanned) != "" {
		if c := cleanupModelCodeCandidate(scanned); c != "" {
			return c, true
		}
	}
	fallback := cleanupModelCodeCandidate(stripCodeFence(content))
	if fallback != "" && !strings.Contains(fallback, `"code"`) {
		return fallback, true
	}
	return "", false
}

func (s *AnimationService) validateGeneratedPythonSyntax(code string) error {
	tmpDir, err := os.MkdirTemp("", "ambient-script-validate-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)
	pyFile := filepath.Join(tmpDir, "check.py")
	if err := os.WriteFile(pyFile, []byte(code), 0o600); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, s.cfg.ScriptSandboxPython, "-m", "py_compile", pyFile)
	cmd.Env = []string{"PATH=/usr/bin:/bin"}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return errors.New("python syntax check timed out")
		}
		return fmt.Errorf("%w: %s", err, truncate(stderr.String(), 300))
	}
	return nil
}

func extractJSONObjectString(s string) string {
	start := -1
	depth := 0
	inString := false
	escaped := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '"' {
			inString = true
			continue
		}
		if ch == '{' {
			if depth == 0 {
				start = i
			}
			depth++
			continue
		}
		if ch == '}' && depth > 0 {
			depth--
			if depth == 0 && start >= 0 {
				return s[start : i+1]
			}
		}
	}
	return s
}

func ensureExecutableAnimationCode(code string) string {
	trimmed := strings.TrimSpace(code)
	if strings.Contains(trimmed, "ARGS_WIDTH") && strings.Contains(trimmed, "print(json.dumps") {
		return trimmed
	}
	adapter := `

import json

def _normalize_pixels(frame_pixels, width, height):
    out = []
    for y in range(height):
        for x in range(width):
            p = frame_pixels[y][x]
            out.append({"r": max(0,min(255,int(p[0]))), "g": max(0,min(255,int(p[1]))), "b": max(0,min(255,int(p[2])))})
    return out

def _generate_frames(width, height, fps, duration):
    total = max(1, int(fps * duration))
    frames = []
    for i in range(total):
        t = i / max(fps, 1)
        try:
            pixels = render_frame(t, width, height, duration)
        except TypeError:
            pixels = render_frame(t, width, height)
        frames.append({
            "meta": {
                "timestamp_unix_ms": 0,
                "frame_index": i,
                "width": width,
                "height": height,
                "encoding": "rgb24"
            },
            "pixels": _normalize_pixels(pixels, width, height)
        })
    return frames

if __name__ == "__main__":
    print(json.dumps(_generate_frames(ARGS_WIDTH, ARGS_HEIGHT, ARGS_FPS, ARGS_DURATION)))
`
	return trimmed + "\n" + adapter
}
