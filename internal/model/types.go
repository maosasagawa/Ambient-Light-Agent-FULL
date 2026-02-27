package model

import "time"

type Target string

const (
	TargetMatrix Target = "matrix"
	TargetStrip  Target = "strip"
	TargetBoth   Target = "both"
)

type StripMode string

const (
	ModeStatic   StripMode = "static"
	ModeBreath   StripMode = "breath"
	ModeChase    StripMode = "chase"
	ModePulse    StripMode = "pulse"
	ModeFlow     StripMode = "flow"
	ModeWave     StripMode = "wave"
	ModeSparkle  StripMode = "sparkle"
	ModeSurround StripMode = "surround_flow"
)

var SupportedStripModes = map[StripMode]struct{}{
	ModeStatic:   {},
	ModeBreath:   {},
	ModeChase:    {},
	ModePulse:    {},
	ModeFlow:     {},
	ModeWave:     {},
	ModeSparkle:  {},
	ModeSurround: {},
}

type RGB struct {
	R uint8 `json:"r"`
	G uint8 `json:"g"`
	B uint8 `json:"b"`
}

type IntentPlan struct {
	UserID            string    `json:"user_id"`
	Prompt            string    `json:"prompt"`
	Language          string    `json:"language"`
	Target            Target    `json:"target"`
	SceneSuggestion   string    `json:"scene_suggestion"`
	BroadcastCopy     string    `json:"broadcast_copy"`
	TemplateID        string    `json:"template_id"`
	TemplateVariant   string    `json:"template_variant"`
	SuggestedStrip    *StripCmd `json:"suggested_strip,omitempty"`
	SuggestedMatrix   *Matrix   `json:"suggested_matrix,omitempty"`
	CreatedAtUnixMS   int64     `json:"created_at_unix_ms"`
	ReasoningSnapshot string    `json:"reasoning_snapshot"`
}

type Matrix struct {
	Width      int    `json:"width"`
	Height     int    `json:"height"`
	Pixels     []RGB  `json:"pixels"`
	Source     string `json:"source"`
	CreatedBy  string `json:"created_by"`
	CreatedAt  int64  `json:"created_at_unix_ms"`
	Encoding   string `json:"encoding"`
	FrameIndex int    `json:"frame_index"`
}

type FrameMeta struct {
	TimestampUnixMS int64  `json:"timestamp_unix_ms"`
	FrameIndex      int    `json:"frame_index"`
	Width           int    `json:"width"`
	Height          int    `json:"height"`
	Encoding        string `json:"encoding"`
}

type MatrixFrame struct {
	Meta   FrameMeta `json:"meta"`
	Pixels []RGB     `json:"pixels"`
}

type StripCmd struct {
	Mode       StripMode `json:"mode"`
	Color      RGB       `json:"color"`
	Brightness float64   `json:"brightness"`
	Speed      float64   `json:"speed"`
	LedCount   int       `json:"led_count"`
	Reason     string    `json:"reason"`
	CreatedAt  int64     `json:"created_at_unix_ms"`
}

type StripFrame struct {
	Mode       StripMode `json:"mode"`
	LedCount   int       `json:"led_count"`
	FrameIndex int       `json:"frame_index"`
	Pixels     []RGB     `json:"pixels"`
	Phase      float64   `json:"phase"`
	Brightness float64   `json:"brightness"`
	Speed      float64   `json:"speed"`
	Direction  string    `json:"direction"`
	Reason     string    `json:"reason"`
	CreatedAt  int64     `json:"created_at_unix_ms"`
}

type VehicleStripZone struct {
	ZoneID    string `json:"zone_id"`
	Location  string `json:"location"`
	LEDCount  int    `json:"led_count"`
	ChannelID string `json:"channel_id"`
}

type ZoneStripDecision struct {
	ZoneID     string    `json:"zone_id"`
	Location   string    `json:"location"`
	Mode       StripMode `json:"mode"`
	Speed      float64   `json:"speed"`
	Brightness float64   `json:"brightness"`
	Colors     []RGB     `json:"colors"`
	Reason     string    `json:"reason"`
}

type VehicleStripPlan struct {
	VehicleID string              `json:"vehicle_id"`
	Target    Target              `json:"target"`
	Zones     []ZoneStripDecision `json:"zones"`
	Reason    string              `json:"reason"`
	CreatedAt int64               `json:"created_at_unix_ms"`
}

type HardwareZonePayload struct {
	ZoneID     string `json:"zone_id"`
	ChannelID  string `json:"channel_id"`
	Encoding   string `json:"encoding"`
	PayloadB64 string `json:"payload_b64"`
	Meta       string `json:"meta"`
}

type HardwareEnvelope struct {
	VehicleID string                `json:"vehicle_id"`
	CreatedAt int64                 `json:"created_at_unix_ms"`
	Zones     []HardwareZonePayload `json:"zones"`
}

type ScriptLanguage string

const (
	ScriptPython ScriptLanguage = "python"
)

type AnimationScript struct {
	ID          string         `json:"id"`
	UserID      string         `json:"user_id"`
	Prompt      string         `json:"prompt"`
	Language    ScriptLanguage `json:"language"`
	Code        string         `json:"code"`
	FPS         int            `json:"fps"`
	DurationSec int            `json:"duration_sec"`
	Width       int            `json:"width"`
	Height      int            `json:"height"`
	CreatedAt   int64          `json:"created_at_unix_ms"`
}

type UserPreference struct {
	UserID              string  `json:"user_id"`
	PreferredStripMode  string  `json:"preferred_strip_mode"`
	PreferredBrightness float64 `json:"preferred_brightness"`
	LastPrompt          string  `json:"last_prompt"`
	UpdatedAt           int64   `json:"updated_at_unix_ms"`
}

type PromptTemplate struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Variants map[string]string `json:"variants"`
	Status   string            `json:"status"`
	Updated  int64             `json:"updated_unix_ms"`
}

type StoredState struct {
	LatestMatrix      *Matrix                        `json:"latest_matrix,omitempty"`
	LatestStripCmd    *StripCmd                      `json:"latest_strip_cmd,omitempty"`
	LatestStripFrame  *StripFrame                    `json:"latest_strip_frame,omitempty"`
	FavoriteScripts   []AnimationScript              `json:"favorite_scripts"`
	PromptTemplates   map[string]PromptTemplate      `json:"prompt_templates"`
	UserPreferences   map[string]UserPreference      `json:"user_preferences"`
	LastPlanByUser    map[string]IntentPlan          `json:"last_plan_by_user"`
	LastFramesByUser  map[string][]MatrixFrame       `json:"last_frames_by_user"`
	VehicleZones      map[string][]VehicleStripZone  `json:"vehicle_zones"`
	VehicleZoneCmds   map[string]map[string]StripCmd `json:"vehicle_zone_cmds"`
	HardwareOutbox    map[string]HardwareEnvelope    `json:"hardware_outbox"`
	LastUpdatedUnixMS int64                          `json:"last_updated_unix_ms"`
	CreatedAt         time.Time                      `json:"created_at"`
}

type Event struct {
	Type      string      `json:"type"`
	Payload   interface{} `json:"payload"`
	CreatedAt int64       `json:"created_at_unix_ms"`
}
