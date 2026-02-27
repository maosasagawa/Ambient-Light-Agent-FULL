package service

import (
	"bytes"
	"errors"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"time"

	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/model"
	"ambient-light-agent/internal/storage"
	"ambient-light-agent/internal/ws"
	"github.com/disintegration/imaging"
)

type MatrixService struct {
	cfg   config.Config
	store *storage.Store
	hub   *ws.Hub
}

func NewMatrixService(cfg config.Config, store *storage.Store, hub *ws.Hub) *MatrixService {
	return &MatrixService{cfg: cfg, store: store, hub: hub}
}

func (s *MatrixService) DownsampleImage(userID string, imageBytes []byte, width, height int) (model.Matrix, error) {
	if width <= 0 {
		width = s.cfg.MatrixWidth
	}
	if height <= 0 {
		height = s.cfg.MatrixHeight
	}
	img, _, err := image.Decode(bytes.NewReader(imageBytes))
	if err != nil {
		return model.Matrix{}, err
	}
	resized := imaging.Resize(img, width, height, imaging.Lanczos)

	pixels := make([]model.RGB, 0, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			pixels = append(pixels, model.RGB{R: uint8(r >> 8), G: uint8(g >> 8), B: uint8(b >> 8)})
		}
	}

	m := model.Matrix{
		Width:      width,
		Height:     height,
		Pixels:     pixels,
		Source:     "upload-downsample",
		CreatedBy:  userID,
		CreatedAt:  time.Now().UnixMilli(),
		Encoding:   "rgb24",
		FrameIndex: 0,
	}
	if err := s.store.SetLatestMatrix(m); err != nil {
		return model.Matrix{}, err
	}
	s.hub.BroadcastEvent(model.Event{Type: "matrix.updated", Payload: m, CreatedAt: time.Now().UnixMilli()})
	return m, nil
}

func (s *MatrixService) GenerateStaticFromPrompt(userID, prompt string, width, height int) (model.Matrix, error) {
	if width <= 0 {
		width = s.cfg.MatrixWidth
	}
	if height <= 0 {
		height = s.cfg.MatrixHeight
	}
	if width*height > 4096 {
		return model.Matrix{}, errors.New("matrix too large")
	}

	pixels := make([]model.RGB, 0, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := softColorFromPrompt(prompt, x, y, width, height)
			pixels = append(pixels, c)
		}
	}

	m := model.Matrix{
		Width:      width,
		Height:     height,
		Pixels:     pixels,
		Source:     "prompt-static-fallback",
		CreatedBy:  userID,
		CreatedAt:  time.Now().UnixMilli(),
		Encoding:   "rgb24",
		FrameIndex: 0,
	}
	if err := s.store.SetLatestMatrix(m); err != nil {
		return model.Matrix{}, err
	}
	s.hub.BroadcastEvent(model.Event{Type: "matrix.updated", Payload: m, CreatedAt: time.Now().UnixMilli()})
	return m, nil
}

func softColorFromPrompt(prompt string, x, y, width, height int) model.RGB {
	_ = prompt
	fx := float64(x) / float64(maxInt(width-1, 1))
	fy := float64(y) / float64(maxInt(height-1, 1))
	r := uint8(30 + int(120*fx))
	g := uint8(20 + int(80*(1.0-fy)))
	b := uint8(60 + int(110*(1.0-fx*0.5)))
	return model.RGB{R: r, G: g, B: b}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func EncodeRGB24(pixels []model.RGB) []byte {
	out := make([]byte, 0, len(pixels)*3)
	for _, p := range pixels {
		out = append(out, p.R, p.G, p.B)
	}
	return out
}

func EncodeRGB565(pixels []model.RGB) []byte {
	out := make([]byte, 0, len(pixels)*2)
	for _, p := range pixels {
		v := (uint16(p.R&0xF8) << 8) | (uint16(p.G&0xFC) << 3) | (uint16(p.B) >> 3)
		out = append(out, byte(v>>8), byte(v&0xFF))
	}
	return out
}

func EncodeRGB111(pixels []model.RGB) []byte {
	out := make([]byte, 0, len(pixels))
	for _, p := range pixels {
		var v byte
		if p.R > 127 {
			v |= 0b100
		}
		if p.G > 127 {
			v |= 0b010
		}
		if p.B > 127 {
			v |= 0b001
		}
		out = append(out, v)
	}
	return out
}
