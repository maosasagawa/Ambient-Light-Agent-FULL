package service

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"regexp"
	"strings"
	"time"
	"unicode"

	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/model"
	"ambient-light-agent/internal/storage"
)

type IntentService struct {
	cfg       config.Config
	store     *storage.Store
	aihubmix  *AIHubMixClient
	knowledge *KnowledgeService
}

func NewIntentService(cfg config.Config, store *storage.Store, knowledge *KnowledgeService) *IntentService {
	return &IntentService{cfg: cfg, store: store, aihubmix: NewAIHubMixClient(cfg, cfg.ModelIntent), knowledge: knowledge}
}

func (s *IntentService) Plan(userID, prompt, language string) (model.IntentPlan, error) {
	target := detectTarget(prompt)
	variant := s.pickVariant(userID)
	scene, broadcast := sceneAndCopy(prompt, language, target, variant)
	reasoning := "已使用启发式语义规划（关键词与用户偏好）生成结果。"
	var suggestedStrip *model.StripCmd
	var suggestedMatrix *model.Matrix

	currentStateJSON := s.currentStateJSON()
	kbContext := "(none)"
	if s.knowledge != nil {
		kbContext = s.knowledge.RenderContext(10)
	}

	systemPrompt := s.resolveSystemPromptTemplate(variant)
	userPrompt := s.mergedUserPrompt(prompt, currentStateJSON, kbContext)

	if content, err := s.aihubmix.Chat(systemPrompt, userPrompt, 0.2); err == nil {
		parsed, parseErr := parsePlanResponse(content)
		if parseErr == nil {
			s.applyPlanResponse(&target, &scene, &broadcast, &suggestedStrip, &suggestedMatrix, parsed, prompt)
			reasoning = "已使用 AIHubMix 合并提示词规划，并启用结构化兜底保护。"
		} else {
			if repaired, retryErr := s.retryRepairPlanJSON(content); retryErr == nil {
				s.applyPlanResponse(&target, &scene, &broadcast, &suggestedStrip, &suggestedMatrix, repaired, prompt)
				reasoning = "已使用 AIHubMix 合并提示词规划，并通过一次 JSON 修复流程完成解析。"
			} else {
				reasoning = "AIHubMix 返回成功但结构不符合约定，已回退到启发式规划。"
			}
		}
	}
	broadcast = strings.TrimSpace(broadcast)
	scene = strings.TrimSpace(scene)
	if suggestedStrip == nil && target != model.TargetMatrix {
		if h := heuristicStripFromPrompt(prompt, s.cfg.StripLEDCount); h != nil {
			suggestedStrip = h
		}
	}
	if strings.EqualFold(language, "zh") {
		broadcast = s.localizeReasonTextIfNeeded(broadcast, language)
		if suggestedStrip != nil {
			suggestedStrip.Reason = s.localizeReasonTextIfNeeded(suggestedStrip.Reason, language)
		}
	}

	plan := model.IntentPlan{
		UserID:            userID,
		Prompt:            prompt,
		Language:          language,
		Target:            target,
		SceneSuggestion:   scene,
		BroadcastCopy:     broadcast,
		SuggestedStrip:    suggestedStrip,
		SuggestedMatrix:   suggestedMatrix,
		TemplateID:        "default-intent-template",
		TemplateVariant:   variant,
		CreatedAtUnixMS:   time.Now().UnixMilli(),
		ReasoningSnapshot: reasoning,
	}

	pref := s.store.GetPreference(userID)
	if pref != nil && pref.PreferredStripMode != "" {
		plan.ReasoningSnapshot += " 检测到用户历史偏好并已纳入规划。"
	}

	if err := s.store.SetPlan(userID, plan); err != nil {
		return model.IntentPlan{}, err
	}

	newPref := model.UserPreference{
		UserID:              userID,
		PreferredStripMode:  prefModeFromPrompt(prompt, pref),
		PreferredBrightness: prefBrightnessFromPrompt(prompt, pref),
		LastPrompt:          prompt,
		UpdatedAt:           time.Now().UnixMilli(),
	}
	if err := s.store.UpsertPreference(newPref); err != nil {
		return model.IntentPlan{}, err
	}

	return plan, nil
}

func detectTarget(prompt string) model.Target {
	lp := strings.ToLower(prompt)
	mentionsMatrix := strings.Contains(lp, "矩阵") || strings.Contains(lp, "matrix") || strings.Contains(lp, "像素")
	mentionsStrip := strings.Contains(lp, "灯带") || strings.Contains(lp, "strip") || strings.Contains(lp, "flow") || strings.Contains(lp, "breath")
	if mentionsMatrix && mentionsStrip {
		return model.TargetBoth
	}
	if mentionsMatrix {
		return model.TargetMatrix
	}
	if mentionsStrip {
		return model.TargetStrip
	}
	return model.TargetBoth
}

func sceneAndCopy(prompt, lang string, target model.Target, variant string) (string, string) {
	lp := strings.ToLower(prompt)
	style := "soft haze"
	emotion := detectEmotion(prompt)
	emotionZH := emotionLabelZH(emotion)
	optimisticFallback := shouldUseOptimisticFallback(prompt)
	if strings.Contains(lp, "雨") || strings.Contains(lp, "rain") {
		style = "misty rain glow"
	}
	if strings.Contains(lp, "夜") || strings.Contains(lp, "night") {
		style = "calm midnight gradient"
	}
	if strings.Contains(lp, "火") || strings.Contains(lp, "火焰") || strings.Contains(lp, "flame") || strings.Contains(lp, "fire") || strings.Contains(lp, "燃烧") {
		style = "flickering ember flame"
	}
	if optimisticFallback {
		style = "optimistic sunrise gradient"
	}

	if lang == "zh" || strings.Contains(prompt, "，") || strings.Contains(prompt, "。") {
		scene := "我识别到你偏向" + emotionZH + "情绪，建议采用" + style + "场景，让颜色层次更贴合当前氛围。"
		copy := "我根据你的语义和情绪倾向，选择了更匹配的色彩与节奏，整体氛围更连贯。"
		if optimisticFallback {
			scene = "我识别到你偏向" + emotionZH + "情绪，在你未指定颜色时，建议采用" + style + "，用更积极明快的色调托住情绪。"
			copy = "你没有指定颜色，我优先推荐偏乐观的暖亮配色，帮助情绪从低压状态平稳过渡到更舒展的氛围。"
		}
		if target == model.TargetStrip {
			scene = "我识别到你偏向" + emotionZH + "情绪，建议灯带使用" + style + "配色并保持中低速变化，兼顾氛围与舒适度。"
			if optimisticFallback {
				scene = "我识别到你偏向" + emotionZH + "情绪，在你未指定颜色时，建议灯带使用" + style + "并保持中低速变化，兼顾提振感与舒适度。"
			}
		}
		if variant == "B" {
			copy = "我做了偏保守的亮度与节奏推荐，让色彩表达更稳定，也更适合长时间驾驶。"
			if optimisticFallback {
				copy = "在你未指定颜色时，我采用了偏乐观但不过激的暖亮配色，并保持保守亮度与节奏，避免疲劳与分心。"
			}
		}
		return scene, copy
	}

	scene := "I detected a " + emotion + " mood and recommend a " + style + " scene for more coherent color storytelling."
	copy := "I mapped your intent and mood cues to color and motion choices, so the atmosphere feels deliberate and consistent."
	if optimisticFallback {
		scene = "I detected a " + emotion + " mood; since no color was specified, I recommend an optimistic sunrise gradient to gently lift the overall tone."
		copy = "Because you did not specify colors, I prioritized a brighter optimistic palette to support emotional recovery while keeping the scene comfortable for driving."
	}
	if variant == "B" {
		copy = "I tuned brightness and pacing to a safer baseline so the scene stays expressive without becoming distracting."
		if optimisticFallback {
			copy = "I kept the optimistic palette but tuned brightness and pacing to a safer baseline, balancing uplift with low distraction."
		}
	}
	return scene, copy
}

func heuristicStripFromPrompt(prompt string, ledCount int) *model.StripCmd {
	lp := strings.ToLower(prompt)
	if strings.Contains(lp, "火") || strings.Contains(lp, "火焰") || strings.Contains(lp, "flame") || strings.Contains(lp, "fire") || strings.Contains(lp, "燃烧") {
		return &model.StripCmd{
			Mode:       model.ModeWave,
			Color:      ensureLuminousColor(model.RGB{R: 242, G: 112, B: 36}),
			Brightness: 0.72,
			Speed:      1.3,
			LedCount:   ledCount,
			Reason:     "你提到火焰，我用暖橙作为主色并加入更有跳动感的波浪节奏，让火光氛围更真实。",
			CreatedAt:  time.Now().UnixMilli(),
		}
	}
	return nil
}

func detectEmotion(prompt string) string {
	lp := strings.ToLower(prompt)
	if strings.Contains(lp, "calm") || strings.Contains(lp, "放松") || strings.Contains(lp, "平静") || strings.Contains(lp, "柔") {
		return "calm"
	}
	if strings.Contains(lp, "happy") || strings.Contains(lp, "开心") || strings.Contains(lp, "愉快") || strings.Contains(lp, "活力") {
		return "uplifting"
	}
	if strings.Contains(lp, "sad") || strings.Contains(lp, "down") || strings.Contains(lp, "upset") || strings.Contains(lp, "anxious") || strings.Contains(lp, "stressed") || strings.Contains(lp, "depressed") || strings.Contains(lp, "tired") || strings.Contains(lp, "难过") || strings.Contains(lp, "低落") || strings.Contains(lp, "烦") || strings.Contains(lp, "焦虑") || strings.Contains(lp, "压抑") || strings.Contains(lp, "沮丧") || strings.Contains(lp, "疲惫") || strings.Contains(lp, "糟糕") {
		return "soothing"
	}
	if strings.Contains(lp, "激情") || strings.Contains(lp, "energetic") || strings.Contains(lp, "兴奋") {
		return "energetic"
	}
	return "balanced"
}

func emotionLabelZH(emotion string) string {
	switch emotion {
	case "calm":
		return "平静"
	case "uplifting":
		return "愉悦"
	case "soothing":
		return "安抚"
	case "energetic":
		return "活力"
	default:
		return "均衡"
	}
}

func shouldUseOptimisticFallback(prompt string) bool {
	return detectEmotion(prompt) == "soothing" && !hasExplicitColorPreference(prompt)
}

func hasExplicitColorPreference(prompt string) bool {
	lp := strings.ToLower(prompt)
	if explicitSingleColorRequested(prompt) {
		return true
	}
	colorKeys := []string{
		"red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black", "cyan", "magenta", "amber", "gold",
		"warm", "cool", "cold", "neon", "pastel",
		"红", "蓝", "绿", "黄", "橙", "紫", "粉", "白", "黑", "青", "洋红", "琥珀", "金",
		"暖色", "冷色", "霓虹", "马卡龙",
	}
	for _, key := range colorKeys {
		if strings.Contains(lp, key) {
			return true
		}
	}
	hexColorPattern := regexp.MustCompile(`#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\\b`)
	if hexColorPattern.MatchString(prompt) {
		return true
	}
	rgbPattern := regexp.MustCompile(`rgb\\s*\\(\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*\\)`)
	return rgbPattern.MatchString(lp)
}

func optimisticCoreColor() model.RGB {
	return model.RGB{R: 255, G: 170, B: 72}
}

func (s *IntentService) pickVariant(userID string) string {
	h := fnv.New32a()
	_, _ = h.Write([]byte(userID))
	if h.Sum32()%2 == 0 {
		return "A"
	}
	return "B"
}

func prefModeFromPrompt(prompt string, prev *model.UserPreference) string {
	lp := strings.ToLower(prompt)
	for mode := range model.SupportedStripModes {
		if strings.Contains(lp, string(mode)) {
			return string(mode)
		}
	}
	if prev != nil && prev.PreferredStripMode != "" {
		return prev.PreferredStripMode
	}
	return string(model.ModeFlow)
}

func prefBrightnessFromPrompt(prompt string, prev *model.UserPreference) float64 {
	lp := strings.ToLower(prompt)
	if strings.Contains(lp, "low") || strings.Contains(lp, "柔") || strings.Contains(lp, "soft") {
		return 0.35
	}
	if strings.Contains(lp, "bright") || strings.Contains(lp, "亮") {
		return 0.75
	}
	if prev != nil && prev.PreferredBrightness > 0 {
		return prev.PreferredBrightness
	}
	return 0.5
}

type planResponse struct {
	Target string `json:"target"`
	Strip  struct {
		Theme      string   `json:"theme"`
		Mode       string   `json:"mode"`
		Speed      *float64 `json:"speed"`
		Brightness *float64 `json:"brightness"`
		Colors     []struct {
			Name string `json:"name"`
			RGB  []int  `json:"rgb"`
		} `json:"colors"`
		Reason string `json:"reason"`
	} `json:"strip"`
	Matrix struct {
		ScenePrompt string `json:"scene_prompt"`
		Reason      string `json:"reason"`
	} `json:"matrix"`
	SpeakableReason string `json:"speakable_reason"`
	StripZones      []struct {
		ZoneID     string   `json:"zone_id"`
		Location   string   `json:"location"`
		Mode       string   `json:"mode"`
		Speed      *float64 `json:"speed"`
		Brightness *float64 `json:"brightness"`
		Colors     []struct {
			Name string `json:"name"`
			RGB  []int  `json:"rgb"`
		} `json:"colors"`
		Reason string `json:"reason"`
	} `json:"strip_zones"`
}

func parsePlanResponse(content string) (planResponse, error) {
	var out planResponse
	if err := json.Unmarshal([]byte(extractJSONObject(content)), &out); err != nil {
		return planResponse{}, err
	}
	return out, nil
}

func (s *IntentService) PlanVehicle(userID, vehicleID, prompt, language string, zones []model.VehicleStripZone) (model.IntentPlan, model.VehicleStripPlan, error) {
	basePlan, err := s.Plan(userID, prompt, language)
	if err != nil {
		return model.IntentPlan{}, model.VehicleStripPlan{}, err
	}

	if len(zones) == 0 {
		zones = s.store.GetVehicleZones(vehicleID)
	}
	if len(zones) == 0 {
		zones = defaultVehicleZones()
		_ = s.store.UpsertVehicleZones(vehicleID, zones)
	}

	decisions := s.heuristicZoneDecisions(basePlan, zones)

	zonePrompt := s.zoneAwareSystemPrompt()
	zoneUser := s.zoneAwareUserPrompt(prompt, language, vehicleID, zones)
	if content, aiErr := s.aihubmix.Chat(zonePrompt, zoneUser, 0.2); aiErr == nil {
		if parsed, parseErr := parsePlanResponse(content); parseErr == nil {
			if len(parsed.StripZones) > 0 {
				decisions = mergeZoneDecisions(decisions, parsed.StripZones)
			}
			if strings.TrimSpace(parsed.SpeakableReason) != "" {
				basePlan.BroadcastCopy = strings.TrimSpace(parsed.SpeakableReason)
			}
		}
	}
	decisions = normalizeZonePalettes(decisions, prompt)
	if strings.EqualFold(language, "zh") {
		for i := range decisions {
			decisions[i].Reason = s.localizeReasonTextIfNeeded(decisions[i].Reason, language)
		}
		basePlan.BroadcastCopy = s.localizeReasonTextIfNeeded(basePlan.BroadcastCopy, language)
	}

	vPlan := model.VehicleStripPlan{
		VehicleID: vehicleID,
		Target:    basePlan.Target,
		Zones:     decisions,
		Reason:    "已完成面向车辆分区与硬件通道的灯带联动规划。",
		CreatedAt: time.Now().UnixMilli(),
	}

	for _, z := range decisions {
		primary := model.RGB{R: 90, G: 120, B: 150}
		if len(z.Colors) > 0 {
			primary = z.Colors[0]
		}
		_ = s.store.SetVehicleZoneCmd(vehicleID, z.ZoneID, model.StripCmd{
			Mode:       z.Mode,
			Color:      primary,
			Brightness: z.Brightness,
			Speed:      z.Speed,
			LedCount:   zoneLEDCount(zones, z.ZoneID),
			Reason:     z.Reason,
			CreatedAt:  time.Now().UnixMilli(),
		})
	}

	return basePlan, vPlan, nil
}

func (s *IntentService) heuristicZoneDecisions(base model.IntentPlan, zones []model.VehicleStripZone) []model.ZoneStripDecision {
	mode := model.ModeFlow
	brightness := 0.5
	speed := 0.6
	primary := model.RGB{R: 90, G: 120, B: 150}
	baseReason := ""
	if base.SuggestedStrip != nil {
		mode = base.SuggestedStrip.Mode
		brightness = base.SuggestedStrip.Brightness
		speed = base.SuggestedStrip.Speed
		primary = base.SuggestedStrip.Color
		baseReason = strings.TrimSpace(base.SuggestedStrip.Reason)
	}
	out := make([]model.ZoneStripDecision, 0, len(zones))
	wantsSingle := explicitSingleColorRequested(base.Prompt)
	emotionZH := emotionLabelZH(detectEmotion(base.Prompt))
	for idx, z := range zones {
		localB := brightness
		if strings.Contains(strings.ToLower(z.Location), "footwell") {
			localB = clampFloat(brightness*0.8, 0.1, 1)
		}
		if strings.Contains(strings.ToLower(z.Location), "dashboard") {
			localB = clampFloat(brightness*0.9, 0.1, 1)
		}
		palette := buildIntentPalette(primary, idx, wantsSingle)
		reason := baseReason
		if reason == "" {
			reason = buildIntentAwareZoneReason(base.Prompt, emotionZH, mode, z.Location, palette)
		}
		out = append(out, model.ZoneStripDecision{
			ZoneID:     z.ZoneID,
			Location:   z.Location,
			Mode:       mode,
			Speed:      speed,
			Brightness: localB,
			Colors:     palette,
			Reason:     reason,
		})
	}
	return out
}

func buildIntentAwareZoneReason(prompt, emotionZH string, mode model.StripMode, location string, palette []model.RGB) string {
	mainColor := "主色"
	if len(palette) > 0 {
		mainColor = colorNameSimpleZH(palette[0])
	}
	modeZH := stripModeNameZH(mode)
	loc := strings.TrimSpace(location)
	if loc == "" {
		loc = "当前区域"
	}
	if shouldUseOptimisticFallback(prompt) {
		return "考虑到你当前偏" + emotionZH + "的状态，我在" + loc + "优先用更明亮的" + mainColor + "并配合" + modeZH + "节奏，既提振情绪也保持舒适。"
	}
	return "结合你当前意图，我在" + loc + "以" + mainColor + "为主并搭配协调辅色，配合" + modeZH + "动效，让氛围更贴合你想要的感觉。"
}

func stripModeNameZH(mode model.StripMode) string {
	switch mode {
	case model.ModeStatic:
		return "静态"
	case model.ModeBreath:
		return "呼吸"
	case model.ModeFlow:
		return "流动"
	case model.ModeChase:
		return "追逐"
	case model.ModePulse:
		return "脉冲"
	case model.ModeWave:
		return "波浪"
	case model.ModeSparkle:
		return "闪烁"
	case model.ModeSurround:
		return "环绕流动"
	default:
		return "动态"
	}
}

func colorNameSimpleZH(c model.RGB) string {
	r := int(c.R)
	g := int(c.G)
	b := int(c.B)
	if r >= g && r >= b {
		if g > 150 && b < 130 {
			return "暖橙色"
		}
		if b > 150 {
			return "玫红色"
		}
		return "红橙色"
	}
	if g >= r && g >= b {
		if b > 150 {
			return "青绿色"
		}
		return "绿色"
	}
	if r > 150 {
		return "紫蓝色"
	}
	return "蓝色"
}

func normalizeZonePalettes(decisions []model.ZoneStripDecision, prompt string) []model.ZoneStripDecision {
	wantsSingle := explicitSingleColorRequested(prompt)
	for i := range decisions {
		base := model.RGB{R: 90, G: 120, B: 150}
		if len(decisions[i].Colors) > 0 {
			base = decisions[i].Colors[0]
		}
		if wantsSingle {
			decisions[i].Colors = []model.RGB{base}
			continue
		}
		if len(decisions[i].Colors) >= 2 && len(decisions[i].Colors) <= 3 {
			continue
		}
		decisions[i].Colors = buildIntentPalette(base, i, false)
	}
	return decisions
}

func explicitSingleColorRequested(prompt string) bool {
	lp := strings.ToLower(prompt)
	keys := []string{
		"single color", "single-color", "one color", "solid color", "monochrome",
		"纯色", "单色", "一个颜色", "一种颜色", "同一种颜色",
	}
	for _, k := range keys {
		if strings.Contains(lp, k) {
			return true
		}
	}
	return false
}

func buildIntentPalette(primary model.RGB, zoneIndex int, single bool) []model.RGB {
	primary = ensureLuminousColor(primary)
	if single {
		return []model.RGB{primary}
	}
	h, s, v := rgbToHSV(primary)
	accentHue := math.Mod(h+18+float64((zoneIndex*7)%22), 360)
	deepHue := math.Mod(h+334-float64((zoneIndex*5)%18)+360, 360)
	c1 := primary
	c2 := hsvToRGB(accentHue, clampFloat(s*0.92, 0.35, 1), clampFloat(v*1.08, 0.25, 1))
	c3 := hsvToRGB(deepHue, clampFloat(s*1.05, 0.35, 1), clampFloat(v*0.92, 0.24, 1))
	return ensureLuminousPalette([]model.RGB{c1, c2, c3})
}

func rgbToHSV(c model.RGB) (float64, float64, float64) {
	r := float64(c.R) / 255
	g := float64(c.G) / 255
	b := float64(c.B) / 255
	maxV := math.Max(r, math.Max(g, b))
	minV := math.Min(r, math.Min(g, b))
	delta := maxV - minV
	h := 0.0
	if delta != 0 {
		switch maxV {
		case r:
			h = 60 * math.Mod((g-b)/delta, 6)
		case g:
			h = 60 * (((b - r) / delta) + 2)
		default:
			h = 60 * (((r - g) / delta) + 4)
		}
	}
	if h < 0 {
		h += 360
	}
	s := 0.0
	if maxV > 0 {
		s = delta / maxV
	}
	return h, s, maxV
}

func hsvToRGB(h, s, v float64) model.RGB {
	h = math.Mod(h+360, 360)
	s = clampFloat(s, 0, 1)
	v = clampFloat(v, 0, 1)
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := v - c
	r1, g1, b1 := 0.0, 0.0, 0.0
	switch {
	case h < 60:
		r1, g1, b1 = c, x, 0
	case h < 120:
		r1, g1, b1 = x, c, 0
	case h < 180:
		r1, g1, b1 = 0, c, x
	case h < 240:
		r1, g1, b1 = 0, x, c
	case h < 300:
		r1, g1, b1 = x, 0, c
	default:
		r1, g1, b1 = c, 0, x
	}
	return model.RGB{
		R: uint8(clampInt(int(math.Round((r1+m)*255)), 0, 255)),
		G: uint8(clampInt(int(math.Round((g1+m)*255)), 0, 255)),
		B: uint8(clampInt(int(math.Round((b1+m)*255)), 0, 255)),
	}
}

func mergeZoneDecisions(base []model.ZoneStripDecision, llmZones []struct {
	ZoneID     string   `json:"zone_id"`
	Location   string   `json:"location"`
	Mode       string   `json:"mode"`
	Speed      *float64 `json:"speed"`
	Brightness *float64 `json:"brightness"`
	Colors     []struct {
		Name string `json:"name"`
		RGB  []int  `json:"rgb"`
	} `json:"colors"`
	Reason string `json:"reason"`
}) []model.ZoneStripDecision {
	idx := map[string]int{}
	for i := range base {
		idx[base[i].ZoneID] = i
	}
	for _, z := range llmZones {
		i, ok := idx[z.ZoneID]
		if !ok {
			continue
		}
		mode := model.StripMode(strings.ToLower(z.Mode))
		if _, ok := model.SupportedStripModes[mode]; ok {
			base[i].Mode = mode
		}
		if z.Speed != nil && *z.Speed > 0 {
			base[i].Speed = *z.Speed
		}
		if z.Brightness != nil && *z.Brightness >= 0 && *z.Brightness <= 1 {
			base[i].Brightness = *z.Brightness
		}
		if len(z.Colors) > 0 {
			colors := make([]model.RGB, 0, len(z.Colors))
			for _, c := range z.Colors {
				if len(c.RGB) < 3 {
					continue
				}
				colors = append(colors, model.RGB{R: uint8(clampInt(c.RGB[0], 0, 255)), G: uint8(clampInt(c.RGB[1], 0, 255)), B: uint8(clampInt(c.RGB[2], 0, 255))})
			}
			if len(colors) > 0 {
				base[i].Colors = colors
			}
		}
		if strings.TrimSpace(z.Reason) != "" {
			base[i].Reason = strings.TrimSpace(z.Reason)
		}
	}
	return base
}

func zoneLEDCount(zones []model.VehicleStripZone, zoneID string) int {
	for _, z := range zones {
		if z.ZoneID == zoneID {
			return z.LEDCount
		}
	}
	return 0
}

func defaultVehicleZones() []model.VehicleStripZone {
	return []model.VehicleStripZone{
		{ZoneID: "dashboard", Location: "dashboard", LEDCount: 60, ChannelID: "ch1"},
		{ZoneID: "door_left", Location: "front-left door", LEDCount: 45, ChannelID: "ch2"},
		{ZoneID: "door_right", Location: "front-right door", LEDCount: 45, ChannelID: "ch3"},
		{ZoneID: "footwell_left", Location: "front-left footwell", LEDCount: 30, ChannelID: "ch4"},
		{ZoneID: "footwell_right", Location: "front-right footwell", LEDCount: 30, ChannelID: "ch5"},
	}
}

func (s *IntentService) zoneAwareSystemPrompt() string {
	return s.mergedSystemPrompt() + "\nAdditionally return strip_zones[] with zone_id/location/mode/speed/brightness/colors/reason for multi-strip vehicle planning."
}

func (s *IntentService) zoneAwareUserPrompt(prompt, language, vehicleID string, zones []model.VehicleStripZone) string {
	b, _ := json.Marshal(zones)
	current := s.store.GetVehicleZoneCmds(vehicleID)
	cb, _ := json.Marshal(current)
	return fmt.Sprintf("Language: %s\nVehicleID: %s\nZones: %s\nCurrentZoneState: %s\nUserInstruction: %s", language, vehicleID, string(b), string(cb), prompt)
}

func (s *IntentService) retryRepairPlanJSON(raw string) (planResponse, error) {
	repairSystem := "You are a strict JSON repair assistant. Return only valid JSON with the exact same schema requested by planner."
	repairUser := "Fix this model output into valid JSON only, no markdown, no extra keys: \n" + raw
	content, err := s.aihubmix.Chat(repairSystem, repairUser, 0)
	if err != nil {
		return planResponse{}, err
	}
	return parsePlanResponse(content)
}

func (s *IntentService) applyPlanResponse(
	target *model.Target,
	scene *string,
	broadcast *string,
	suggestedStrip **model.StripCmd,
	suggestedMatrix **model.Matrix,
	parsed planResponse,
	prompt string,
) {
	switch model.Target(strings.ToLower(parsed.Target)) {
	case model.TargetMatrix, model.TargetStrip, model.TargetBoth:
		*target = model.Target(strings.ToLower(parsed.Target))
	}
	if strings.TrimSpace(parsed.SpeakableReason) != "" {
		*broadcast = strings.TrimSpace(parsed.SpeakableReason)
	}
	if strings.TrimSpace(parsed.Matrix.ScenePrompt) != "" {
		sceneValue := strings.TrimSpace(parsed.Matrix.ScenePrompt)
		if strings.TrimSpace(parsed.Matrix.Reason) != "" {
			sceneValue += " | " + strings.TrimSpace(parsed.Matrix.Reason)
		}
		*scene = sceneValue
		*suggestedMatrix = &model.Matrix{
			Width:      s.cfg.MatrixWidth,
			Height:     s.cfg.MatrixHeight,
			Pixels:     nil,
			Source:     "llm-scene-prompt",
			CreatedBy:  "intent-planner",
			CreatedAt:  time.Now().UnixMilli(),
			Encoding:   "rgb24",
			FrameIndex: 0,
		}
	}
	if hasStripUpdate(parsed) {
		baseMode := model.ModeFlow
		baseColor := model.RGB{R: 90, G: 120, B: 150}
		baseBrightness := 0.5
		baseSpeed := 0.6
		baseReason := "已按你的意图与安全优先级规划灯带方案。"
		if latest := s.store.GetLatestStrip(); latest != nil {
			if _, ok := model.SupportedStripModes[latest.Mode]; ok {
				baseMode = latest.Mode
			}
			baseColor = latest.Color
			if latest.Brightness >= 0 && latest.Brightness <= 1 {
				baseBrightness = latest.Brightness
			}
			if latest.Speed > 0 {
				baseSpeed = latest.Speed
			}
			if strings.TrimSpace(latest.Reason) != "" {
				baseReason = strings.TrimSpace(latest.Reason)
			}
		}

		mode := baseMode
		if m := strings.TrimSpace(parsed.Strip.Mode); m != "" {
			candidate := model.StripMode(strings.ToLower(m))
			if _, ok := model.SupportedStripModes[candidate]; ok {
				mode = candidate
			}
		}

		color := baseColor
		if len(parsed.Strip.Colors) > 0 && len(parsed.Strip.Colors[0].RGB) >= 3 {
			color = model.RGB{
				R: uint8(clampInt(parsed.Strip.Colors[0].RGB[0], 0, 255)),
				G: uint8(clampInt(parsed.Strip.Colors[0].RGB[1], 0, 255)),
				B: uint8(clampInt(parsed.Strip.Colors[0].RGB[2], 0, 255)),
			}
		}
		if shouldUseOptimisticFallback(prompt) {
			color = optimisticCoreColor()
		}
		color = ensureLuminousColor(color)

		brightness := baseBrightness
		if parsed.Strip.Brightness != nil {
			brightness = clampFloat(*parsed.Strip.Brightness, 0, 1)
		}

		speed := baseSpeed
		if parsed.Strip.Speed != nil && *parsed.Strip.Speed > 0 {
			speed = *parsed.Strip.Speed
		}

		reason := baseReason
		if r := strings.TrimSpace(parsed.Strip.Reason); r != "" {
			reason = r
		}
		if shouldUseOptimisticFallback(prompt) {
			reason = "检测到你情绪偏低且未指定颜色，我优先采用乐观暖亮主色，在安全优先的前提下做温和提振。"
		}

		*suggestedStrip = &model.StripCmd{
			Mode:       mode,
			Color:      color,
			Brightness: brightness,
			Speed:      speed,
			LedCount:   s.cfg.StripLEDCount,
			Reason:     reason,
			CreatedAt:  time.Now().UnixMilli(),
		}
	}
}

func hasStripUpdate(parsed planResponse) bool {
	if strings.TrimSpace(parsed.Strip.Theme) != "" {
		return true
	}
	if strings.TrimSpace(parsed.Strip.Mode) != "" {
		return true
	}
	if parsed.Strip.Speed != nil || parsed.Strip.Brightness != nil {
		return true
	}
	if len(parsed.Strip.Colors) > 0 {
		return true
	}
	if strings.TrimSpace(parsed.Strip.Reason) != "" {
		return true
	}
	return false
}

func (s *IntentService) currentStateJSON() string {
	state := map[string]interface{}{}
	if cmd := s.store.GetLatestStrip(); cmd != nil {
		state["latest_strip_cmd"] = cmd
	}
	if frame := s.store.GetLatestStripFrame(); frame != nil {
		state["latest_strip_frame"] = frame
	}
	b, err := json.Marshal(state)
	if err != nil {
		return "{}"
	}
	if len(b) == 2 {
		return "{}"
	}
	return string(b)
}

func (s *IntentService) mergedSystemPrompt() string {
	return strings.TrimSpace(`
# Role
You are a gentle, empathetic lighting designer with a poetic soul.
Analyze the user's request and plan the lighting effect for a 16x16 Pixel Matrix and an LED Strip.

# Task
1. Determine Intent: control matrix, strip, or both.
2. Plan Strip if involved:
- Supported modes: static, breath, flow, chase, pulse, wave, sparkle, surround_flow.
- Choose mode + numeric speed (>0).
- Current State is authoritative for incremental edits.
- Incremental Update Rules (CRITICAL):
  - If user asks partial adjustment (e.g., faster/slower/change color/brighter/dimmer), modify ONLY explicitly requested fields.
  - Copy ALL unspecified strip fields exactly from Current State (mode/colors/speed/brightness/theme behavior).
- Speed standard: for all dynamic modes (breath/flow/chase/pulse/wave/sparkle/surround_flow), larger speed means faster; smaller speed means slower.
- For slower: decrease speed value; for faster: increase speed value.
- Practical reference: around 0.2~0.7 is slow, around 0.8~1.6 is medium, >=1.8 is fast.
  - For brighter/dimmer: adjust brightness in [0.0,1.0].
- Treat as full redesign only when user clearly asks a new scene/mood/theme.
- Prefer 2~3 harmonious colors unless user explicitly requests single-color.
3. Plan Matrix if involved: generate scene_prompt describing scene/image content for 16x16 matrix (not generic lighting jargon).
4. Speakable reason: one Chinese poetic line, under 60 Chinese characters, must explicitly mention chosen color/light atmosphere.
5. IMPORTANT: For strip-related Chinese copy, do NOT mention frosted glass/diffusion. Explain only color/scene selection based on detected intent/emotion and safety.
6. For strip.reason and speakable_reason: return natural Chinese generated for this specific request; do NOT use fixed template sentences.
7. strip.reason should focus on WHY these colors were chosen (core color + accent relation + mood/safety intent).

# Selection Rules
1. Safety first: when driving safety/health concern appears, prioritize alertness-supportive colors and explain safety priority in reason.
2. User intent priority: explicit user command overrides inferred state.
3. Priority chain: safety > user intent > system inference.

# Color Rules
1. Colors should be vivid and saturated when the intent does not request dim/muted style.
2. Keep luminous feel; for dark scenes keep hue while reducing brightness.
3. Multi-color sets must be harmonious (complementary or contrast with aesthetic coherence).
4. RGB must be integer 0~255.
5. The first color should be the theme core color.
6. If user mood appears negative (e.g., sad/anxious/stressed/down) and user did not explicitly specify colors, default to an optimistic uplifting palette and explain this reason clearly.
7. Avoid dull or dim-looking strip recommendations by keeping core and accent colors perceptibly luminous.

# Output Format (JSON only)
{
  "target": "matrix" | "strip" | "both",
  "strip": {
    "theme": "Short Theme Name",
    "mode": "static | breath | flow | chase | pulse | wave | sparkle | surround_flow",
    "speed": 2.0,
    "brightness": 1.0,
    "colors": [{"name": "Color Name", "rgb": [R, G, B]}],
    "reason": "Internal logic explanation"
  },
  "matrix": {
    "scene_prompt": "pixel art scene prompt for 16x16",
    "reason": "Why this prompt suits user's intent"
  },
  "speakable_reason": "一句可口播的温柔解释"
}

You MUST base your full plan (target, strip, matrix.scene_prompt, speakable_reason) on User Instruction.
Do NOT ignore, replace, or dilute explicit user intent with a default theme.
Only return valid JSON. No markdown.
`)
}

func (s *IntentService) resolveSystemPromptTemplate(variant string) string {
	for _, t := range s.store.ListTemplates() {
		if t.ID != "default-intent-template" {
			continue
		}
		if t.Status != "" && strings.ToLower(t.Status) != "active" {
			continue
		}
		if v, ok := t.Variants[variant]; ok && strings.TrimSpace(v) != "" {
			return v
		}
		if v, ok := t.Variants["default"]; ok && strings.TrimSpace(v) != "" {
			return v
		}
	}
	return s.mergedSystemPrompt()
}

func (s *IntentService) mergedUserPrompt(instruction, currentState, kbContext string) string {
	return fmt.Sprintf("Current State: %s\n\nKB Context:\n%s\n\nUser Instruction: %s\n\nYou MUST ground all planning on this instruction and keep incremental updates faithful to Current State.", currentState, kbContext, instruction)
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func clampFloat(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func containsHan(s string) bool {
	for _, ch := range s {
		if unicode.Is(unicode.Han, ch) {
			return true
		}
	}
	return false
}

func (s *IntentService) localizeReasonTextIfNeeded(text, language string) string {
	t := strings.TrimSpace(text)
	if t == "" {
		return t
	}
	if !strings.EqualFold(language, "zh") || containsHan(t) {
		return t
	}
	system := "You rewrite short lighting recommendation reasons into natural Chinese. Keep meaning and color-choice rationale. Do not use rigid templates. Return JSON only: {\"text\":\"...\"}."
	user := "Rewrite this into natural Chinese:\n" + t
	content, err := s.aihubmix.Chat(system, user, 0)
	if err != nil {
		return t
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(content)), &out); err != nil {
		return t
	}
	localized := strings.TrimSpace(out.Text)
	if localized == "" {
		return t
	}
	return localized
}

func extractJSONObject(s string) string {
	t := strings.TrimSpace(s)
	t = strings.TrimPrefix(t, "```json")
	t = strings.TrimPrefix(t, "```")
	t = strings.TrimSuffix(t, "```")
	t = strings.TrimSpace(t)

	start := -1
	depth := 0
	inString := false
	escaped := false
	for i := 0; i < len(t); i++ {
		ch := t[i]
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
		if ch == '}' {
			if depth > 0 {
				depth--
				if depth == 0 && start >= 0 {
					return t[start : i+1]
				}
			}
		}
	}
	return t
}
