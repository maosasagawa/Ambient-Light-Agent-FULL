package service

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"ambient-light-agent/internal/config"
)

var ErrNoAIHubMixKey = errors.New("aihubmix api key not configured")

type AIHubMixClient struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client
}

func NewAIHubMixClient(cfg config.Config, model string) *AIHubMixClient {
	return NewAIHubMixClientWithTimeout(cfg, model, cfg.AIHubMixTimeoutSec)
}

func NewAIHubMixClientWithTimeout(cfg config.Config, model string, timeoutSec int) *AIHubMixClient {
	if timeoutSec <= 0 {
		timeoutSec = cfg.AIHubMixTimeoutSec
		if timeoutSec <= 0 {
			timeoutSec = 12
		}
	}
	return &AIHubMixClient{
		baseURL: cfg.AIHubMixBaseURL,
		apiKey:  cfg.AIHubMixAPIKey,
		model:   model,
		http: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
	}
}

func (c *AIHubMixClient) Chat(system, user string, temperature float64) (string, error) {
	if strings.TrimSpace(c.apiKey) == "" {
		return "", ErrNoAIHubMixKey
	}

	reqBody := map[string]interface{}{
		"model": c.model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"temperature": temperature,
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest(http.MethodPost, c.chatCompletionsURL(), bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error interface{} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}

	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("aihubmix status=%d error=%v", resp.StatusCode, out.Error)
	}
	if len(out.Choices) == 0 {
		return "", errors.New("aihubmix returned empty choices")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

func (c *AIHubMixClient) chatCompletionsURL() string {
	if strings.HasSuffix(c.baseURL, "/v1") {
		return c.baseURL + "/chat/completions"
	}
	return c.baseURL + "/v1/chat/completions"
}
