package service

import (
	"errors"
	"math"
	"strings"
	"time"

	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/model"
	"ambient-light-agent/internal/storage"
	"ambient-light-agent/internal/ws"
)

type StripService struct {
	cfg       config.Config
	store     *storage.Store
	hub       *ws.Hub
	knowledge *KnowledgeService
}

func NewStripService(cfg config.Config, store *storage.Store, hub *ws.Hub, knowledge *KnowledgeService) *StripService {
	return &StripService{cfg: cfg, store: store, hub: hub, knowledge: knowledge}
}

func (s *StripService) Recommend(userID, prompt string, mode model.StripMode, brightness, speed float64) (model.StripCmd, error) {
	if _, ok := model.SupportedStripModes[mode]; !ok {
		return model.StripCmd{}, errors.New("unsupported strip mode")
	}
	if brightness < 0 || brightness > 1 {
		return model.StripCmd{}, errors.New("brightness must be in [0,1]")
	}
	if speed < 0 {
		return model.StripCmd{}, errors.New("speed must be >= 0")
	}

	base := model.RGB{R: 90, G: 120, B: 150}
	last := s.store.GetLatestStrip()
	if last != nil {
		base = last.Color
	}
	hints := s.knowledge.FindHints(prompt)
	color := applyHints(base, hints, prompt)

	cmd := model.StripCmd{
		Mode:       mode,
		Color:      color,
		Brightness: brightness,
		Speed:      speed,
		LedCount:   s.cfg.StripLEDCount,
		Reason:     reasonFromHints(hints, prompt),
		CreatedAt:  time.Now().UnixMilli(),
	}
	if err := s.store.SetLatestStrip(cmd); err != nil {
		return model.StripCmd{}, err
	}

	pref := s.store.GetPreference(userID)
	brightnessPref := brightness
	if pref != nil && brightness == 0 {
		brightnessPref = pref.PreferredBrightness
	}
	_ = s.store.UpsertPreference(model.UserPreference{
		UserID:              userID,
		PreferredStripMode:  string(mode),
		PreferredBrightness: brightnessPref,
		LastPrompt:          prompt,
		UpdatedAt:           time.Now().UnixMilli(),
	})

	s.hub.BroadcastEvent(model.Event{Type: "strip.updated", Payload: cmd, CreatedAt: time.Now().UnixMilli()})
	if frame, err := s.GenerateFrame(prompt, cmd, 0, "clockwise"); err == nil {
		s.hub.BroadcastEvent(model.Event{Type: "strip.frame.updated", Payload: frame, CreatedAt: time.Now().UnixMilli()})
	}
	return cmd, nil
}

func (s *StripService) GenerateFrame(prompt string, cmd model.StripCmd, phase float64, direction string) (model.StripFrame, error) {
	if cmd.LedCount <= 0 {
		cmd.LedCount = s.cfg.StripLEDCount
	}
	if cmd.Brightness < 0 || cmd.Brightness > 1 {
		return model.StripFrame{}, errors.New("brightness must be in [0,1]")
	}
	if direction == "" {
		direction = "clockwise"
	}
	palette := deriveFramePalette(cmd.Color, prompt)

	pixels := make([]model.RGB, cmd.LedCount)
	motionSpeed := calibratedStripMotionSpeed(cmd.Speed)
	switch cmd.Mode {
	case model.ModeStatic:
		for i := range pixels {
			position := float64(i) / float64(maxInt(cmd.LedCount-1, 1))
			pixels[i] = applyBrightness(samplePalette(palette, position), cmd.Brightness)
		}
	case model.ModeBreath, model.ModePulse:
		baseWave := 0.5 + 0.5*math.Sin((phase*motionSpeed)*2*math.Pi)
		factor := 0.5 + 0.5*baseWave
		for i := range pixels {
			position := float64(i) / float64(maxInt(cmd.LedCount-1, 1))
			spatial := 0.9 + 0.1*(0.5+0.5*math.Sin((position+phase*0.35)*2*math.Pi))
			pixels[i] = applyBrightness(samplePalette(palette, position), cmd.Brightness*factor*spatial)
		}
	case model.ModeChase, model.ModeFlow, model.ModeWave, model.ModeSparkle, model.ModeSurround:
		for i := range pixels {
			position := float64(i) / float64(maxInt(cmd.LedCount, 1))
			idxPhase := phase*motionSpeed + position
			if direction == "counterclockwise" {
				idxPhase = phase*motionSpeed - position
			}
			wave := 0.5 + 0.5*math.Sin((idxPhase*1.2)*2*math.Pi)
			trail := 0.42 + 0.58*wave
			if cmd.Mode == model.ModeSparkle {
				seed := 0.5 + 0.5*math.Sin((idxPhase*11.5+float64(i)*0.73)*2*math.Pi)
				trail = 0.28 + 0.72*seed
			}
			if cmd.Mode == model.ModeChase {
				head := 0.5 + 0.5*math.Sin((idxPhase*2.4)*2*math.Pi)
				trail = 0.18 + 0.82*math.Pow(head, 2.2)
			}
			if cmd.Mode == model.ModeWave {
				wave2 := 0.5 + 0.5*math.Sin((idxPhase*0.65+phase*0.4)*2*math.Pi)
				trail = 0.35 + 0.65*(0.65*wave+0.35*wave2)
			}
			if cmd.Mode == model.ModeSurround {
				center := math.Abs(position - 0.5)
				shell := 1 - clampFloat(center*1.9, 0, 1)
				trail = 0.28 + 0.72*(0.55*wave+0.45*shell)
			}
			colorPos := math.Mod(position+phase*motionSpeed*0.35+wave*0.08, 1)
			pixels[i] = applyBrightness(samplePalette(palette, colorPos), cmd.Brightness*trail)
		}
	default:
		return model.StripFrame{}, errors.New("unsupported strip mode")
	}

	frame := model.StripFrame{
		Mode:       cmd.Mode,
		LedCount:   cmd.LedCount,
		FrameIndex: 0,
		Pixels:     pixels,
		Phase:      phase,
		Brightness: cmd.Brightness,
		Speed:      cmd.Speed,
		Direction:  direction,
		Reason:     "已根据当前模式、相位与方向生成灯带逐点帧。",
		CreatedAt:  time.Now().UnixMilli(),
	}
	if err := s.store.SetLatestStripFrame(frame); err != nil {
		return model.StripFrame{}, err
	}
	s.hub.BroadcastEvent(model.Event{Type: "strip.frame.updated", Payload: frame, CreatedAt: time.Now().UnixMilli()})
	_ = prompt
	return frame, nil
}

func calibratedStripMotionSpeed(speed float64) float64 {
	v := clampFloat(speed, 0, 4)
	return v * 0.28
}

func deriveFramePalette(base model.RGB, prompt string) []model.RGB {
	base = ensureLuminousColor(base)
	if explicitSingleColorRequested(prompt) {
		return []model.RGB{base}
	}
	h, s, v := rgbToHSV(base)
	c1 := base
	c2 := hsvToRGB(math.Mod(h+24, 360), clampFloat(s*0.9, 0.35, 1), clampFloat(v*1.08, 0.24, 1))
	c3 := hsvToRGB(math.Mod(h+334, 360), clampFloat(s*1.02, 0.35, 1), clampFloat(v*0.92, 0.22, 1))
	return ensureLuminousPalette([]model.RGB{c1, c2, c3})
}

func samplePalette(palette []model.RGB, t float64) model.RGB {
	if len(palette) == 0 {
		return model.RGB{R: 90, G: 120, B: 150}
	}
	if len(palette) == 1 {
		return palette[0]
	}
	t = math.Mod(t+1, 1)
	if t < 0 {
		t += 1
	}
	segments := len(palette) - 1
	segment := t * float64(segments)
	idx := int(math.Floor(segment))
	if idx >= segments {
		return palette[len(palette)-1]
	}
	local := segment - float64(idx)
	holdPortion := 0.84
	blendPortion := 1 - holdPortion
	if local <= holdPortion {
		return palette[idx]
	}
	blendT := (local - holdPortion) / math.Max(blendPortion, 1e-6)
	return blendRGB(palette[idx], palette[idx+1], clampFloat(blendT, 0, 1))
}

func blendRGB(a, b model.RGB, t float64) model.RGB {
	t = clampFloat(t, 0, 1)
	return model.RGB{
		R: uint8(clampInt(int(math.Round(float64(a.R)+(float64(b.R)-float64(a.R))*t)), 0, 255)),
		G: uint8(clampInt(int(math.Round(float64(a.G)+(float64(b.G)-float64(a.G))*t)), 0, 255)),
		B: uint8(clampInt(int(math.Round(float64(a.B)+(float64(b.B)-float64(a.B))*t)), 0, 255)),
	}
}

func (s *StripService) PatchPositions(mode model.StripMode, brightness, speed float64, updates map[int]model.RGB) (model.StripFrame, error) {
	ledCount := s.cfg.StripLEDCount
	base := s.store.GetLatestStripFrame()
	pixels := make([]model.RGB, ledCount)
	for i := range pixels {
		pixels[i] = model.RGB{R: 0, G: 0, B: 0}
	}
	if base != nil && len(base.Pixels) == ledCount {
		copy(pixels, base.Pixels)
	}
	for idx, c := range updates {
		if idx < 0 || idx >= ledCount {
			return model.StripFrame{}, errors.New("position index out of range")
		}
		pixels[idx] = applyBrightness(c, brightness)
	}
	frame := model.StripFrame{
		Mode:       mode,
		LedCount:   ledCount,
		FrameIndex: 0,
		Pixels:     pixels,
		Phase:      0,
		Brightness: brightness,
		Speed:      speed,
		Direction:  "manual",
		Reason:     "已应用逐点灯珠手动控制更新。",
		CreatedAt:  time.Now().UnixMilli(),
	}
	if err := s.store.SetLatestStripFrame(frame); err != nil {
		return model.StripFrame{}, err
	}
	s.hub.BroadcastEvent(model.Event{Type: "strip.frame.updated", Payload: frame, CreatedAt: time.Now().UnixMilli()})
	return frame, nil
}

func (s *StripService) BuildSurroundFlow(prompt string, brightness, speed float64, phase float64, direction string) (model.StripFrame, error) {
	if brightness < 0 || brightness > 1 {
		return model.StripFrame{}, errors.New("brightness must be in [0,1]")
	}
	cmd, err := s.Recommend("anon", prompt, model.ModeSurround, brightness, speed)
	if err != nil {
		return model.StripFrame{}, err
	}
	frame, err := s.GenerateFrame(prompt, cmd, phase, direction)
	if err != nil {
		return model.StripFrame{}, err
	}
	frame.Reason = "已识别用户情绪并生成环绕流动效果，支持全灯带逐点控制。"
	_ = s.store.SetLatestStripFrame(frame)
	s.hub.BroadcastEvent(model.Event{Type: "strip.surround_flow.updated", Payload: frame, CreatedAt: time.Now().UnixMilli()})
	return frame, nil
}

func (s *StripService) GenerateSequence(prompt string, cmd model.StripCmd, fps, durationSec int, direction string) ([]model.StripFrame, error) {
	if fps <= 0 {
		fps = s.cfg.SyncFPS
	}
	if durationSec <= 0 {
		durationSec = 2
	}
	total := fps * durationSec
	if total < 1 {
		total = 1
	}
	frames := make([]model.StripFrame, 0, total)
	for i := 0; i < total; i++ {
		phase := float64(i) / float64(total)
		f, err := s.GenerateFrame(prompt, cmd, phase, direction)
		if err != nil {
			return nil, err
		}
		f.FrameIndex = i
		f.CreatedAt = time.Now().UnixMilli() + int64(i*(1000/maxInt(fps, 1)))
		frames = append(frames, f)
	}
	return frames, nil
}

func applyHints(base model.RGB, hints []string, prompt string) model.RGB {
	out := base
	lp := strings.ToLower(prompt)
	if shouldUseOptimisticFallback(prompt) {
		out = optimisticCoreColor()
	}
	if strings.Contains(lp, "warm") || strings.Contains(lp, "暖") {
		out = model.RGB{R: 220, G: 140, B: 70}
	}
	if strings.Contains(lp, "cool") || strings.Contains(lp, "冷") {
		out = model.RGB{R: 60, G: 130, B: 220}
	}
	for _, h := range hints {
		lh := strings.ToLower(h)
		switch {
		case strings.Contains(lh, "amber"):
			out = model.RGB{R: 240, G: 150, B: 40}
		case strings.Contains(lh, "ocean"):
			out = model.RGB{R: 35, G: 120, B: 200}
		case strings.Contains(lh, "sunset"):
			out = model.RGB{R: 210, G: 95, B: 85}
		}
	}
	return ensureLuminousColor(out)
}

func reasonFromHints(hints []string, prompt string) string {
	emotion := detectEmotion(prompt)
	if shouldUseOptimisticFallback(prompt) {
		return "检测到你当前情绪偏低且未指定颜色，我优先采用偏乐观的暖亮色，帮助情绪平稳提振并保持驾驶舒适。"
	}
	if len(hints) > 0 {
		return "识别到" + emotion + "情绪意图，并结合知识库色彩线索与历史状态，推荐更连贯的灯光方案。"
	}
	return "识别到" + emotion + "情绪语义，已按该氛围选择主色与动态节奏，并保持与当前灯带状态的连续性。"
}

func applyBrightness(c model.RGB, b float64) model.RGB {
	if b < 0 {
		b = 0
	}
	if b > 1 {
		b = 1
	}
	return model.RGB{
		R: uint8(math.Round(float64(c.R) * b)),
		G: uint8(math.Round(float64(c.G) * b)),
		B: uint8(math.Round(float64(c.B) * b)),
	}
}
