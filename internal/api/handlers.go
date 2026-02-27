package api

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/model"
	"ambient-light-agent/internal/service"
	"ambient-light-agent/internal/storage"
	"ambient-light-agent/internal/ws"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

type Handler struct {
	cfg         config.Config
	store       *storage.Store
	hub         *ws.Hub
	hardwareHub *ws.HardwareHub
	intentSvc   *service.IntentService
	matrixSvc   *service.MatrixService
	stripSvc    *service.StripService
	animSvc     *service.AnimationService
	upgrader    websocket.Upgrader
}

type apiError struct {
	Error string `json:"error"`
}

func (h *Handler) Healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handler) WebSocket(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, errors.New("websocket requires GET"))
		return
	}
	if !websocket.IsWebSocketUpgrade(r) {
		writeErr(w, http.StatusBadRequest, errors.New("websocket upgrade required"))
		return
	}
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws upgrade failed: remote=%s host=%s uri=%s err=%v", r.RemoteAddr, r.Host, r.RequestURI, err)
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	client := ws.NewClient(h.hub, conn)
	h.hub.BroadcastEvent(model.Event{Type: "ws.client_connected", Payload: map[string]string{"id": uuid.NewString()}, CreatedAt: time.Now().UnixMilli()})
	h.hub.Register(client)
	go client.WritePump()
	go client.ReadPump()
}

func (h *Handler) HardwareWebSocket(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, errors.New("websocket requires GET"))
		return
	}
	if !websocket.IsWebSocketUpgrade(r) {
		writeErr(w, http.StatusBadRequest, errors.New("websocket upgrade required"))
		return
	}
	vehicleID := strings.TrimSpace(r.URL.Query().Get("vehicle_id"))
	if vehicleID == "" {
		writeErr(w, http.StatusBadRequest, errors.New("vehicle_id required"))
		return
	}
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("hardware ws upgrade failed: remote=%s host=%s uri=%s err=%v", r.RemoteAddr, r.Host, r.RequestURI, err)
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	client := h.hardwareHub.Register(vehicleID, conn)
	if env := h.store.GetHardwareEnvelope(vehicleID); env != nil {
		h.hardwareHub.PushEnvelope(*env)
	}
	go client.WritePump()
	go client.ReadPump()
}

func (h *Handler) PlanIntent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		UserID   string `json:"user_id"`
		Prompt   string `json:"prompt"`
		Language string `json:"language"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.UserID == "" {
		req.UserID = userIDFromRequest(r)
	}
	if strings.TrimSpace(req.Prompt) == "" {
		writeErr(w, http.StatusBadRequest, errors.New("prompt required"))
		return
	}
	if req.Language == "" {
		req.Language = "zh"
	}

	plan, err := h.intentSvc.Plan(req.UserID, req.Prompt, req.Language)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	h.hub.BroadcastEvent(model.Event{Type: "intent.planned", Payload: plan, CreatedAt: time.Now().UnixMilli()})
	if plan.SuggestedStrip != nil {
		_ = h.store.SetLatestStrip(*plan.SuggestedStrip)
		h.hub.BroadcastEvent(model.Event{Type: "strip.updated", Payload: *plan.SuggestedStrip, CreatedAt: time.Now().UnixMilli()})
		if frame, frameErr := h.stripSvc.GenerateFrame(req.Prompt, *plan.SuggestedStrip, 0, "clockwise"); frameErr == nil {
			h.hub.BroadcastEvent(model.Event{Type: "strip.frame.updated", Payload: frame, CreatedAt: time.Now().UnixMilli()})
		}
	}
	writeJSON(w, http.StatusOK, plan)
}

func (h *Handler) DownsampleMatrix(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	if err := r.ParseMultipartForm(h.cfg.MaxUploadSizeBytes); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	userID := firstOr(r.FormValue("user_id"), userIDFromRequest(r))
	width := atoiDefault(r.FormValue("width"), h.cfg.MatrixWidth)
	height := atoiDefault(r.FormValue("height"), h.cfg.MatrixHeight)

	file, fileHeader, err := r.FormFile("image")
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	defer file.Close()

	if err := validateImageUpload(fileHeader); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	b, err := io.ReadAll(file)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}

	m, err := h.matrixSvc.DownsampleImage(userID, b, width, height)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}

func (h *Handler) GenerateStaticMatrix(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		UserID string `json:"user_id"`
		Prompt string `json:"prompt"`
		Width  int    `json:"width"`
		Height int    `json:"height"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.UserID == "" {
		req.UserID = userIDFromRequest(r)
	}
	m, err := h.matrixSvc.GenerateStaticFromPrompt(req.UserID, req.Prompt, req.Width, req.Height)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}

func (h *Handler) GenerateAnimation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		UserID      string `json:"user_id"`
		Prompt      string `json:"prompt"`
		FPS         int    `json:"fps"`
		DurationSec int    `json:"duration_sec"`
		Width       int    `json:"width"`
		Height      int    `json:"height"`
		Strict      bool   `json:"strict"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.UserID == "" {
		req.UserID = userIDFromRequest(r)
	}
	script, frames, err := h.animSvc.Generate(req.UserID, req.Prompt, req.FPS, req.DurationSec, req.Width, req.Height, req.Strict)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	previewDuration := script.DurationSec
	if previewDuration <= 0 {
		previewDuration = 8
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"script":               script,
		"frames":               frames,
		"infinite_duration":    script.DurationSec <= 0,
		"preview_duration_sec": previewDuration,
	})
}

func (h *Handler) GetLatestMatrix(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	m := h.store.GetLatestMatrix()
	if m == nil {
		writeErr(w, http.StatusNotFound, errors.New("no matrix data"))
		return
	}
	writeJSON(w, http.StatusOK, m)
}

func (h *Handler) GetLatestMatrixRGB24(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	m := h.store.GetLatestMatrix()
	if m == nil {
		writeErr(w, http.StatusNotFound, errors.New("no matrix data"))
		return
	}
	b := service.EncodeRGB24(m.Pixels)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Width", itoa(m.Width))
	w.Header().Set("X-Height", itoa(m.Height))
	_, _ = w.Write(b)
}

func (h *Handler) RecommendStrip(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		UserID     string          `json:"user_id"`
		Prompt     string          `json:"prompt"`
		Mode       model.StripMode `json:"mode"`
		Brightness float64         `json:"brightness"`
		Speed      float64         `json:"speed"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Mode == "" {
		req.Mode = model.ModeFlow
	}
	if req.UserID == "" {
		req.UserID = userIDFromRequest(r)
	}
	if req.Brightness == 0 {
		req.Brightness = 0.5
	}
	if req.Speed == 0 {
		req.Speed = 0.6
	}
	cmd, err := h.stripSvc.Recommend(req.UserID, req.Prompt, req.Mode, req.Brightness, req.Speed)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, cmd)
}

func (h *Handler) GenerateStripFrame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Prompt     string          `json:"prompt"`
		Mode       model.StripMode `json:"mode"`
		Brightness float64         `json:"brightness"`
		Speed      float64         `json:"speed"`
		Phase      float64         `json:"phase"`
		Direction  string          `json:"direction"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Mode == "" {
		req.Mode = model.ModeFlow
	}
	if req.Brightness == 0 {
		req.Brightness = 0.5
	}
	if req.Speed == 0 {
		req.Speed = 0.6
	}
	cmd, err := h.stripSvc.Recommend(userIDFromRequest(r), req.Prompt, req.Mode, req.Brightness, req.Speed)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	frame, err := h.stripSvc.GenerateFrame(req.Prompt, cmd, req.Phase, req.Direction)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, frame)
}

func (h *Handler) GenerateStripAnimation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Prompt      string          `json:"prompt"`
		Mode        model.StripMode `json:"mode"`
		Brightness  float64         `json:"brightness"`
		Speed       float64         `json:"speed"`
		Direction   string          `json:"direction"`
		FPS         int             `json:"fps"`
		DurationSec int             `json:"duration_sec"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Mode == "" {
		req.Mode = model.ModeFlow
	}
	if req.Brightness == 0 {
		req.Brightness = 0.5
	}
	if req.Speed == 0 {
		req.Speed = 0.6
	}
	effectiveFPS := req.FPS
	if effectiveFPS <= 0 {
		effectiveFPS = h.cfg.SyncFPS
	}
	if effectiveFPS > 60 {
		effectiveFPS = 60
	}
	effectiveDuration := req.DurationSec
	if effectiveDuration <= 0 {
		effectiveDuration = 2
	}
	if effectiveDuration > 12 {
		effectiveDuration = 12
	}
	maxFrames := 600
	if effectiveFPS*effectiveDuration > maxFrames {
		effectiveDuration = maxFrames / effectiveFPS
		if effectiveDuration < 1 {
			effectiveDuration = 1
		}
	}
	cmd, err := h.stripSvc.Recommend(userIDFromRequest(r), req.Prompt, req.Mode, req.Brightness, req.Speed)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	frames, err := h.stripSvc.GenerateSequence(req.Prompt, cmd, effectiveFPS, effectiveDuration, req.Direction)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"sequence_id":  uuid.NewString(),
		"mode":         cmd.Mode,
		"fps":          effectiveFPS,
		"duration_sec": effectiveDuration,
		"frames":       frames,
	})
}

func (h *Handler) PatchStripPositions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Mode       model.StripMode `json:"mode"`
		Brightness float64         `json:"brightness"`
		Speed      float64         `json:"speed"`
		Updates    []struct {
			Index int       `json:"index"`
			Color model.RGB `json:"color"`
		} `json:"updates"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Mode == "" {
		req.Mode = model.ModeStatic
	}
	if req.Brightness == 0 {
		req.Brightness = 1.0
	}
	updates := make(map[int]model.RGB, len(req.Updates))
	for _, u := range req.Updates {
		updates[u.Index] = u.Color
	}
	frame, err := h.stripSvc.PatchPositions(req.Mode, req.Brightness, req.Speed, updates)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, frame)
}

func (h *Handler) GenerateSurroundFlow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Prompt     string  `json:"prompt"`
		Brightness float64 `json:"brightness"`
		Speed      float64 `json:"speed"`
		Phase      float64 `json:"phase"`
		Direction  string  `json:"direction"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Brightness == 0 {
		req.Brightness = 0.5
	}
	if req.Speed == 0 {
		req.Speed = 0.8
	}
	frame, err := h.stripSvc.BuildSurroundFlow(req.Prompt, req.Brightness, req.Speed, req.Phase, req.Direction)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, frame)
}

func (h *Handler) EncodeStrip(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Encoding string      `json:"encoding"`
		Pixels   []model.RGB `json:"pixels"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var payload []byte
	switch strings.ToLower(req.Encoding) {
	case "rgb24":
		payload = service.EncodeRGB24(req.Pixels)
	case "rgb565":
		payload = service.EncodeRGB565(req.Pixels)
	case "rgb111":
		payload = service.EncodeRGB111(req.Pixels)
	default:
		writeErr(w, http.StatusBadRequest, errors.New("unsupported encoding"))
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Encoding", strings.ToLower(req.Encoding))
	_, _ = w.Write(payload)
}

func (h *Handler) Templates(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, h.store.ListTemplates())
	case http.MethodPost:
		var req model.PromptTemplate
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		if req.ID == "" {
			req.ID = uuid.NewString()
		}
		if req.Status == "" {
			req.Status = "active"
		}
		req.Updated = time.Now().UnixMilli()
		if err := h.store.UpsertTemplate(req); err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, req)
	default:
		methodNotAllowed(w)
	}
}

func (h *Handler) FavoriteScripts(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, h.animSvc.ListFavorites())
	case http.MethodPost:
		var script model.AnimationScript
		if err := json.NewDecoder(r.Body).Decode(&script); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		if script.ID == "" {
			script.ID = uuid.NewString()
		}
		script.CreatedAt = time.Now().UnixMilli()
		if err := h.animSvc.Favorite(script); err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, script)
	default:
		methodNotAllowed(w)
	}
}

func (h *Handler) VehicleZones(w http.ResponseWriter, r *http.Request) {
	vehicleID := strings.TrimSpace(r.URL.Query().Get("vehicle_id"))
	if vehicleID == "" {
		vehicleID = "vehicle-default"
	}
	switch r.Method {
	case http.MethodGet:
		zones := h.store.GetVehicleZones(vehicleID)
		if len(zones) == 0 {
			zones = []model.VehicleStripZone{
				{ZoneID: "dashboard", Location: "dashboard", LEDCount: 60, ChannelID: "ch1"},
				{ZoneID: "door_left", Location: "front-left door", LEDCount: 45, ChannelID: "ch2"},
				{ZoneID: "door_right", Location: "front-right door", LEDCount: 45, ChannelID: "ch3"},
				{ZoneID: "footwell_left", Location: "front-left footwell", LEDCount: 30, ChannelID: "ch4"},
				{ZoneID: "footwell_right", Location: "front-right footwell", LEDCount: 30, ChannelID: "ch5"},
			}
			_ = h.store.UpsertVehicleZones(vehicleID, zones)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"vehicle_id": vehicleID, "zones": zones})
	case http.MethodPost:
		var req struct {
			VehicleID string                   `json:"vehicle_id"`
			Zones     []model.VehicleStripZone `json:"zones"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		if req.VehicleID == "" {
			req.VehicleID = vehicleID
		}
		if len(req.Zones) == 0 {
			writeErr(w, http.StatusBadRequest, errors.New("zones required"))
			return
		}
		if err := h.store.UpsertVehicleZones(req.VehicleID, req.Zones); err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"vehicle_id": req.VehicleID, "zones": req.Zones})
	default:
		methodNotAllowed(w)
	}
}

func (h *Handler) VoiceCommand(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	req := struct {
		UserID     string                   `json:"user_id"`
		VehicleID  string                   `json:"vehicle_id"`
		Language   string                   `json:"language"`
		Prompt     string                   `json:"prompt"`
		Transcript string                   `json:"transcript"`
		Zones      []model.VehicleStripZone `json:"zones"`
	}{}
	if r.Method == http.MethodGet {
		req.UserID = strings.TrimSpace(r.URL.Query().Get("user_id"))
		req.VehicleID = strings.TrimSpace(r.URL.Query().Get("vehicle_id"))
		req.Language = strings.TrimSpace(r.URL.Query().Get("language"))
		req.Prompt = strings.TrimSpace(r.URL.Query().Get("prompt"))
		req.Transcript = strings.TrimSpace(r.URL.Query().Get("transcript"))
	} else {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
	}
	text := strings.TrimSpace(req.Prompt)
	if text == "" {
		text = strings.TrimSpace(req.Transcript)
	}
	resp, err := h.applyScene(text, req.UserID, req.VehicleID, req.Language, req.Zones, "voice")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) AppCommand(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if r.Method == http.MethodGet {
		userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
		vehicleID := strings.TrimSpace(r.URL.Query().Get("vehicle_id"))
		language := strings.TrimSpace(r.URL.Query().Get("language"))
		prompt := strings.TrimSpace(r.URL.Query().Get("prompt"))
		text := strings.TrimSpace(r.URL.Query().Get("text"))
		if prompt == "" {
			prompt = text
		}
		resp, err := h.applyScene(prompt, userID, vehicleID, language, nil, "app")
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, resp)
		return
	}

	contentType := strings.ToLower(strings.TrimSpace(r.Header.Get("Content-Type")))
	if strings.Contains(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(h.cfg.MaxUploadSizeBytes); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		op := strings.ToLower(strings.TrimSpace(firstOr(r.FormValue("op"), r.FormValue("operation"))))
		if op == "" {
			op = "matrix_upload"
		}
		if op != "matrix_upload" {
			writeErr(w, http.StatusBadRequest, errors.New("multipart supports only matrix_upload"))
			return
		}
		userID := firstOr(r.FormValue("user_id"), userIDFromRequest(r))
		width := atoiDefault(r.FormValue("width"), h.cfg.MatrixWidth)
		height := atoiDefault(r.FormValue("height"), h.cfg.MatrixHeight)
		file, fileHeader, err := r.FormFile("image")
		if err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		defer file.Close()
		if err := validateImageUpload(fileHeader); err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		imageBytes, err := io.ReadAll(file)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		matrix, err := h.matrixSvc.DownsampleImage(userID, imageBytes, width, height)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"operation": "matrix_upload", "matrix": matrix})
		return
	}

	req := struct {
		Operation   string                   `json:"operation"`
		Op          string                   `json:"op"`
		UserID      string                   `json:"user_id"`
		VehicleID   string                   `json:"vehicle_id"`
		Language    string                   `json:"language"`
		Prompt      string                   `json:"prompt"`
		Text        string                   `json:"text"`
		Mode        model.StripMode          `json:"mode"`
		Brightness  float64                  `json:"brightness"`
		Speed       float64                  `json:"speed"`
		Color       *model.RGB               `json:"color"`
		LEDCount    int                      `json:"led_count"`
		Width       int                      `json:"width"`
		Height      int                      `json:"height"`
		FPS         int                      `json:"fps"`
		DurationSec int                      `json:"duration_sec"`
		Strict      bool                     `json:"strict"`
		ImageB64    string                   `json:"image_b64"`
		Zones       []model.VehicleStripZone `json:"zones"`
	}{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}

	operation := strings.ToLower(strings.TrimSpace(firstOr(req.Operation, req.Op)))
	if operation == "" {
		operation = "scene"
	}
	userID := firstOr(req.UserID, userIDFromRequest(r))
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		prompt = strings.TrimSpace(req.Text)
	}

	switch operation {
	case "scene":
		resp, err := h.applyScene(prompt, userID, req.VehicleID, req.Language, req.Zones, "app")
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, resp)
	case "manual_strip":
		resp, err := h.applyManualStrip(userID, req.VehicleID, req.Mode, req.Brightness, req.Speed, req.Color, req.LEDCount)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		writeJSON(w, http.StatusOK, resp)
	case "matrix_latest":
		matrix := h.store.GetLatestMatrix()
		if matrix == nil {
			writeErr(w, http.StatusNotFound, errors.New("no matrix data"))
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"operation": "matrix_latest", "matrix": matrix})
	case "matrix_static":
		matrix, err := h.matrixSvc.GenerateStaticFromPrompt(userID, prompt, req.Width, req.Height)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"operation": "matrix_static", "matrix": matrix})
	case "matrix_animate":
		script, frames, err := h.animSvc.Generate(userID, prompt, req.FPS, req.DurationSec, req.Width, req.Height, req.Strict)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		previewDuration := script.DurationSec
		if previewDuration <= 0 {
			previewDuration = 8
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"operation":            "matrix_animate",
			"script":               script,
			"frames":               frames,
			"infinite_duration":    script.DurationSec <= 0,
			"preview_duration_sec": previewDuration,
		})
	case "matrix_upload":
		if strings.TrimSpace(req.ImageB64) == "" {
			writeErr(w, http.StatusBadRequest, errors.New("image_b64 required for matrix_upload json mode"))
			return
		}
		imageBytes, err := base64.StdEncoding.DecodeString(req.ImageB64)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err)
			return
		}
		matrix, err := h.matrixSvc.DownsampleImage(userID, imageBytes, req.Width, req.Height)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"operation": "matrix_upload", "matrix": matrix})
	default:
		writeErr(w, http.StatusBadRequest, errors.New("unsupported operation"))
	}
}

func (h *Handler) applyManualStrip(userID, vehicleID string, mode model.StripMode, brightness, speed float64, color *model.RGB, ledCount int) (map[string]interface{}, error) {
	if userID == "" {
		userID = "anon"
	}
	if vehicleID == "" {
		vehicleID = "vehicle-default"
	}
	if mode == "" {
		mode = model.ModeFlow
	}
	if _, ok := model.SupportedStripModes[mode]; !ok {
		return nil, errors.New("unsupported strip mode")
	}
	if brightness <= 0 {
		brightness = 0.6
	}
	if brightness > 1 {
		brightness = 1
	}
	if speed <= 0 {
		speed = 0.6
	}
	if speed > 4 {
		speed = 4
	}
	baseColor := model.RGB{R: 90, G: 120, B: 150}
	if color != nil {
		baseColor = *color
	}
	if ledCount <= 0 {
		ledCount = 24
	}

	zones := h.store.GetVehicleZones(vehicleID)
	if len(zones) == 0 {
		zones = defaultVehicleZonesForManual()
		_ = h.store.UpsertVehicleZones(vehicleID, zones)
	}

	planZones := make([]model.ZoneStripDecision, 0, len(zones))
	env := model.HardwareEnvelope{VehicleID: vehicleID, CreatedAt: time.Now().UnixMilli(), Zones: make([]model.HardwareZonePayload, 0, len(zones))}
	for _, z := range zones {
		colors := []model.RGB{baseColor}
		planZones = append(planZones, model.ZoneStripDecision{
			ZoneID:     z.ZoneID,
			Location:   z.Location,
			Mode:       mode,
			Speed:      speed,
			Brightness: brightness,
			Colors:     colors,
			Reason:     "已应用手动参数。",
		})
		pixels := make([]model.RGB, 0, ledCount)
		for i := 0; i < ledCount; i++ {
			pixels = append(pixels, baseColor)
		}
		env.Zones = append(env.Zones, model.HardwareZonePayload{
			ZoneID:     z.ZoneID,
			ChannelID:  z.ChannelID,
			Encoding:   "rgb24",
			PayloadB64: base64.StdEncoding.EncodeToString(service.EncodeRGB24(pixels)),
			Meta:       "mode=" + string(mode) + ";source=manual",
		})
	}
	preview := buildStripFrame(mode, []model.RGB{baseColor}, brightness, speed, ledCount, "已应用手动参数。")

	if err := h.store.SetLatestStrip(model.StripCmd{
		Mode:       mode,
		Color:      baseColor,
		Brightness: brightness,
		Speed:      speed,
		LedCount:   ledCount,
		Reason:     "已应用手动参数。",
		CreatedAt:  time.Now().UnixMilli(),
	}); err != nil {
		return nil, err
	}
	if err := h.store.SetLatestStripFrame(preview); err != nil {
		return nil, err
	}
	if err := h.store.SetHardwareEnvelope(vehicleID, env); err != nil {
		return nil, err
	}
	h.hardwareHub.PushEnvelope(env)
	h.hub.BroadcastEvent(model.Event{Type: "strip.updated", Payload: model.StripCmd{Mode: mode, Color: baseColor, Brightness: brightness, Speed: speed, LedCount: ledCount, Reason: "已应用手动参数。", CreatedAt: time.Now().UnixMilli()}, CreatedAt: time.Now().UnixMilli()})
	h.hub.BroadcastEvent(model.Event{Type: "strip.frame.updated", Payload: preview, CreatedAt: time.Now().UnixMilli()})
	h.hub.BroadcastEvent(model.Event{Type: "app.manual_strip_applied", Payload: map[string]interface{}{"vehicle_id": vehicleID, "zones": len(env.Zones)}, CreatedAt: time.Now().UnixMilli()})

	vehiclePlan := model.VehicleStripPlan{
		VehicleID: vehicleID,
		Target:    model.TargetStrip,
		Zones:     planZones,
		Reason:    "手动灯带参数已生效。",
		CreatedAt: time.Now().UnixMilli(),
	}
	return map[string]interface{}{
		"operation":         "manual_strip",
		"source":            "app_manual",
		"vehicle_plan":      vehiclePlan,
		"hardware_envelope": env,
		"manual": map[string]interface{}{
			"user_id":    userID,
			"mode":       mode,
			"brightness": brightness,
			"speed":      speed,
			"color":      baseColor,
			"led_count":  ledCount,
		},
	}, nil
}

func defaultVehicleZonesForManual() []model.VehicleStripZone {
	return []model.VehicleStripZone{
		{ZoneID: "dashboard", Location: "dashboard", LEDCount: 60, ChannelID: "ch1"},
		{ZoneID: "door_left", Location: "front-left door", LEDCount: 45, ChannelID: "ch2"},
		{ZoneID: "door_right", Location: "front-right door", LEDCount: 45, ChannelID: "ch3"},
		{ZoneID: "footwell_left", Location: "front-left footwell", LEDCount: 30, ChannelID: "ch4"},
		{ZoneID: "footwell_right", Location: "front-right footwell", LEDCount: 30, ChannelID: "ch5"},
	}
}

func buildStripFrame(mode model.StripMode, colors []model.RGB, brightness, speed float64, ledCount int, reason string) model.StripFrame {
	if len(colors) == 0 {
		colors = []model.RGB{{R: 90, G: 120, B: 150}}
	}
	if ledCount <= 0 {
		ledCount = 24
	}
	pixels := make([]model.RGB, 0, ledCount)
	for i := 0; i < ledCount; i++ {
		pixels = append(pixels, colors[i%len(colors)])
	}
	return model.StripFrame{
		Mode:       mode,
		LedCount:   ledCount,
		FrameIndex: 0,
		Pixels:     pixels,
		Phase:      0,
		Brightness: brightness,
		Speed:      speed,
		Direction:  "clockwise",
		Reason:     reason,
		CreatedAt:  time.Now().UnixMilli(),
	}
}

func (h *Handler) STTPlan(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		UserID     string                   `json:"user_id"`
		VehicleID  string                   `json:"vehicle_id"`
		Language   string                   `json:"language"`
		Transcript string                   `json:"transcript"`
		Zones      []model.VehicleStripZone `json:"zones"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.UserID == "" {
		req.UserID = userIDFromRequest(r)
	}
	if req.VehicleID == "" {
		req.VehicleID = "vehicle-default"
	}
	if req.Language == "" {
		req.Language = "zh"
	}
	if strings.TrimSpace(req.Transcript) == "" {
		writeErr(w, http.StatusBadRequest, errors.New("transcript required"))
		return
	}
	resp, err := h.applyScene(req.Transcript, req.UserID, req.VehicleID, req.Language, req.Zones, "stt")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) AppApplyScene(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		UserID    string                   `json:"user_id"`
		VehicleID string                   `json:"vehicle_id"`
		Language  string                   `json:"language"`
		Prompt    string                   `json:"prompt"`
		Zones     []model.VehicleStripZone `json:"zones"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.UserID == "" {
		req.UserID = userIDFromRequest(r)
	}
	if req.VehicleID == "" {
		req.VehicleID = "vehicle-default"
	}
	if req.Language == "" {
		req.Language = "zh"
	}
	if strings.TrimSpace(req.Prompt) == "" {
		writeErr(w, http.StatusBadRequest, errors.New("prompt required"))
		return
	}
	resp, err := h.applyScene(req.Prompt, req.UserID, req.VehicleID, req.Language, req.Zones, "app")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) applyScene(prompt, userID, vehicleID, language string, zones []model.VehicleStripZone, source string) (map[string]interface{}, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("prompt required")
	}
	if userID == "" {
		userID = "anon"
	}
	if vehicleID == "" {
		vehicleID = "vehicle-default"
	}
	if language == "" {
		language = "zh"
	}

	plan, vehiclePlan, err := h.intentSvc.PlanVehicle(userID, vehicleID, prompt, language, zones)
	if err != nil {
		return nil, err
	}

	effectiveZones := zones
	if len(effectiveZones) == 0 {
		effectiveZones = h.store.GetVehicleZones(vehicleID)
	}
	env := model.HardwareEnvelope{VehicleID: vehicleID, CreatedAt: time.Now().UnixMilli(), Zones: make([]model.HardwareZonePayload, 0, len(vehiclePlan.Zones))}
	for _, z := range vehiclePlan.Zones {
		pixels := []model.RGB{}
		for i := 0; i < 16; i++ {
			if len(z.Colors) > 0 {
				pixels = append(pixels, z.Colors[i%len(z.Colors)])
			}
		}
		payload := service.EncodeRGB24(pixels)
		env.Zones = append(env.Zones, model.HardwareZonePayload{
			ZoneID:     z.ZoneID,
			ChannelID:  findChannelID(effectiveZones, z.ZoneID),
			Encoding:   "rgb24",
			PayloadB64: base64.StdEncoding.EncodeToString(payload),
			Meta:       "mode=" + string(z.Mode),
		})
	}

	if err := h.store.SetHardwareEnvelope(vehicleID, env); err != nil {
		return nil, err
	}
	h.hardwareHub.PushEnvelope(env)
	if len(vehiclePlan.Zones) > 0 {
		z := vehiclePlan.Zones[0]
		color := model.RGB{R: 90, G: 120, B: 150}
		if len(z.Colors) > 0 {
			color = z.Colors[0]
		}
		cmd := model.StripCmd{
			Mode:       z.Mode,
			Color:      color,
			Brightness: z.Brightness,
			Speed:      z.Speed,
			LedCount:   24,
			Reason:     z.Reason,
			CreatedAt:  time.Now().UnixMilli(),
		}
		frame := buildStripFrame(z.Mode, z.Colors, z.Brightness, z.Speed, 24, z.Reason)
		if err := h.store.SetLatestStrip(cmd); err != nil {
			return nil, err
		}
		if err := h.store.SetLatestStripFrame(frame); err != nil {
			return nil, err
		}
		h.hub.BroadcastEvent(model.Event{Type: "strip.updated", Payload: cmd, CreatedAt: time.Now().UnixMilli()})
		h.hub.BroadcastEvent(model.Event{Type: "strip.frame.updated", Payload: frame, CreatedAt: time.Now().UnixMilli()})
	}
	h.hub.BroadcastEvent(model.Event{Type: source + ".scene_applied", Payload: map[string]interface{}{"vehicle_id": vehicleID, "zones": len(vehiclePlan.Zones)}, CreatedAt: time.Now().UnixMilli()})

	resp := map[string]interface{}{
		"intent_plan":       plan,
		"vehicle_plan":      vehiclePlan,
		"hardware_envelope": env,
		"source":            source,
		"target":            plan.Target,
		"template_variant":  plan.TemplateVariant,
	}
	if plan.SuggestedStrip != nil {
		resp["suggested_strip"] = *plan.SuggestedStrip
	} else if len(vehiclePlan.Zones) > 0 {
		z := vehiclePlan.Zones[0]
		c := model.RGB{R: 90, G: 120, B: 150}
		if len(z.Colors) > 0 {
			c = z.Colors[0]
		}
		resp["suggested_strip"] = model.StripCmd{
			Mode:       z.Mode,
			Color:      c,
			Brightness: z.Brightness,
			Speed:      z.Speed,
			LedCount:   24,
			Reason:     z.Reason,
			CreatedAt:  time.Now().UnixMilli(),
		}
	}
	if m := h.store.GetLatestMatrix(); m != nil {
		resp["matrix_hint"] = map[string]interface{}{
			"path":   "/v1/hardware/matrix/latest",
			"width":  m.Width,
			"height": m.Height,
		}
	}
	return resp, nil
}

func (h *Handler) HardwarePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	vehicleID := strings.TrimSpace(r.URL.Query().Get("vehicle_id"))
	if vehicleID == "" {
		vehicleID = "vehicle-default"
	}
	env := h.store.GetHardwareEnvelope(vehicleID)
	if env == nil {
		writeErr(w, http.StatusNotFound, errors.New("no hardware envelope"))
		return
	}
	writeJSON(w, http.StatusOK, env)
}

func findChannelID(zones []model.VehicleStripZone, zoneID string) string {
	for _, z := range zones {
		if z.ZoneID == zoneID {
			return z.ChannelID
		}
	}
	return zoneID
}

func validateImageUpload(header *multipart.FileHeader) error {
	ext := strings.ToLower(filepath.Ext(header.Filename))
	switch ext {
	case ".png", ".jpg", ".jpeg", ".gif":
		return nil
	default:
		return errors.New("unsupported image format")
	}
}

func (h *Handler) RegisterClient(c *ws.Client) {
	h.hub.Register(c)
}

func writeJSON(w http.ResponseWriter, code int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(data)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, apiError{Error: err.Error()})
}

func methodNotAllowed(w http.ResponseWriter) {
	writeErr(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
}

func firstOr(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return v
}

func userIDFromRequest(r *http.Request) string {
	v := strings.TrimSpace(r.Header.Get("X-User-ID"))
	if v != "" {
		return v
	}
	return "anon"
}

func atoiDefault(v string, d int) int {
	v = strings.TrimSpace(v)
	if v == "" {
		return d
	}
	n := 0
	for _, ch := range v {
		if ch < '0' || ch > '9' {
			return d
		}
		n = n*10 + int(ch-'0')
	}
	if n <= 0 {
		return d
	}
	return n
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	buf := make([]byte, 0, 16)
	for v > 0 {
		d := v % 10
		buf = append([]byte{byte('0' + d)}, buf...)
		v /= 10
	}
	return string(buf)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
