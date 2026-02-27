package api

import (
	"net/http"

	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/service"
	"ambient-light-agent/internal/storage"
	"ambient-light-agent/internal/ws"
	"github.com/gorilla/websocket"
)

func NewRouter(
	cfg config.Config,
	store *storage.Store,
	hub *ws.Hub,
	hardwareHub *ws.HardwareHub,
	intentSvc *service.IntentService,
	matrixSvc *service.MatrixService,
	stripSvc *service.StripService,
	animSvc *service.AnimationService,
) http.Handler {
	h := &Handler{
		cfg:         cfg,
		store:       store,
		hub:         hub,
		hardwareHub: hardwareHub,
		intentSvc:   intentSvc,
		matrixSvc:   matrixSvc,
		stripSvc:    stripSvc,
		animSvc:     animSvc,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", h.Healthz)
	mux.HandleFunc("/debug", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/debug/", http.StatusTemporaryRedirect)
	})
	mux.Handle("/debug/", http.StripPrefix("/debug/", http.FileServer(http.Dir("./web/debug"))))
	mux.HandleFunc("/v1/ws", h.WebSocket)
	mux.HandleFunc("/v1/voice/command", h.VoiceCommand)
	mux.HandleFunc("/v1/app/command", h.AppCommand)
	mux.HandleFunc("/v1/hardware/ws", h.HardwareWebSocket)
	mux.HandleFunc("/v1/hardware/pull", h.HardwarePull)
	mux.HandleFunc("/v1/agent/plan", h.VoiceCommand)
	mux.HandleFunc("/v1/stt/plan", h.VoiceCommand)
	mux.HandleFunc("/v1/app/scene/apply", h.AppCommand)
	mux.HandleFunc("/v1/hw/ws", h.HardwareWebSocket)
	mux.HandleFunc("/v1/hw/pull", h.HardwarePull)
	mux.HandleFunc("/v1/matrix/downsample", h.DownsampleMatrix)
	mux.HandleFunc("/v1/matrix/generate-static", h.GenerateStaticMatrix)
	mux.HandleFunc("/v1/matrix/animate", h.GenerateAnimation)
	mux.HandleFunc("/v1/matrix/latest", h.GetLatestMatrix)
	mux.HandleFunc("/v1/strip/recommend", h.RecommendStrip)
	mux.HandleFunc("/v1/strip/frame", h.GenerateStripFrame)
	mux.HandleFunc("/v1/strip/animate", h.GenerateStripAnimation)

	return limitBody(cfg.MaxUploadSizeBytes, mux)
}

func limitBody(maxSize int64, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxSize)
		next.ServeHTTP(w, r)
	})
}
