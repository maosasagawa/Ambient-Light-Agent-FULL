package service

import (
	"testing"

	"ambient-light-agent/internal/model"
)

func TestEncodeRGB24(t *testing.T) {
	in := []model.RGB{{R: 1, G: 2, B: 3}, {R: 4, G: 5, B: 6}}
	out := EncodeRGB24(in)
	if len(out) != 6 {
		t.Fatalf("unexpected length: %d", len(out))
	}
	if out[0] != 1 || out[1] != 2 || out[2] != 3 || out[3] != 4 || out[4] != 5 || out[5] != 6 {
		t.Fatalf("unexpected payload: %v", out)
	}
}

func TestEncodeRGB565Length(t *testing.T) {
	in := []model.RGB{{R: 255, G: 255, B: 255}, {R: 0, G: 0, B: 0}}
	out := EncodeRGB565(in)
	if len(out) != 4 {
		t.Fatalf("unexpected length: %d", len(out))
	}
}

func TestEncodeRGB111Length(t *testing.T) {
	in := []model.RGB{{R: 255, G: 0, B: 255}}
	out := EncodeRGB111(in)
	if len(out) != 1 {
		t.Fatalf("unexpected length: %d", len(out))
	}
	if out[0] != 0b101 {
		t.Fatalf("unexpected value: %08b", out[0])
	}
}
