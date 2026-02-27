package config

import (
	"errors"
	"os"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	ListenAddr            string
	DataPath              string
	KnowledgeBasePath     string
	MatrixWidth           int
	MatrixHeight          int
	StripLEDCount         int
	SyncFPS               int
	ModelIntent           string
	ModelImage            string
	ModelAnimation        string
	AIHubMixTimeoutSec    int
	AnimationAITimeoutSec int
	AIHubMixBaseURL       string
	AIHubMixAPIKey        string
	GeminiAPIKey          string
	BFLAPIKey             string
	ScriptTimeoutSec      int
	MaxUploadSizeBytes    int64
	ScriptSandboxPython   string
}

func Load() (Config, error) {
	_ = godotenv.Load()

	cfg := Config{
		ListenAddr:            getEnv("LISTEN_ADDR", ":8080"),
		DataPath:              getEnv("DATA_PATH", "./data/state.json"),
		KnowledgeBasePath:     getEnv("KNOWLEDGE_BASE_PATH", "./data/knowledge_colors.txt"),
		MatrixWidth:           getEnvInt("MATRIX_WIDTH", 16),
		MatrixHeight:          getEnvInt("MATRIX_HEIGHT", 16),
		StripLEDCount:         getEnvInt("STRIP_LED_COUNT", 120),
		SyncFPS:               getEnvInt("SYNC_FPS", 30),
		ModelIntent:           getEnv("MODEL_INTENT", "gemini-2.0-flash"),
		ModelImage:            getEnv("MODEL_IMAGE", "imagen"),
		ModelAnimation:        getEnv("MODEL_ANIMATION", "gemini-3-flash-preview"),
		AIHubMixTimeoutSec:    getEnvInt("AIHUBMIX_TIMEOUT_SEC", 12),
		AnimationAITimeoutSec: getEnvInt("ANIMATION_AI_TIMEOUT_SEC", 90),
		AIHubMixBaseURL:       strings.TrimRight(getEnv("AIHUBMIX_BASE_URL", "https://api.aihubmix.example"), "/"),
		AIHubMixAPIKey:        getEnv("AIHUBMIX_API_KEY", ""),
		GeminiAPIKey:          getEnv("GEMINI_API_KEY", ""),
		BFLAPIKey:             getEnv("BFL_API_KEY", ""),
		ScriptTimeoutSec:      getEnvInt("SCRIPT_TIMEOUT_SEC", 8),
		MaxUploadSizeBytes:    getEnvInt64("MAX_UPLOAD_SIZE_BYTES", 8*1024*1024),
		ScriptSandboxPython:   getEnv("SCRIPT_SANDBOX_PYTHON", "python3"),
	}

	if cfg.MatrixWidth <= 0 || cfg.MatrixHeight <= 0 {
		return Config{}, errors.New("matrix width/height must be > 0")
	}
	if cfg.SyncFPS <= 0 {
		return Config{}, errors.New("sync fps must be > 0")
	}
	if cfg.StripLEDCount <= 0 {
		return Config{}, errors.New("strip led count must be > 0")
	}
	if cfg.ScriptTimeoutSec <= 0 {
		return Config{}, errors.New("script timeout sec must be > 0")
	}
	if cfg.AIHubMixTimeoutSec <= 0 {
		return Config{}, errors.New("aihubmix timeout sec must be > 0")
	}
	if cfg.AnimationAITimeoutSec <= 0 {
		return Config{}, errors.New("animation ai timeout sec must be > 0")
	}

	return cfg, nil
}

func getEnv(key, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func getEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func getEnvInt64(key string, fallback int64) int64 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return fallback
	}
	return n
}
