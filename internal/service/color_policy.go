package service

import (
	"math"

	"ambient-light-agent/internal/model"
)

func ensureLuminousColor(c model.RGB) model.RGB {
	r := float64(c.R)
	g := float64(c.G)
	b := float64(c.B)

	luma := 0.2126*r + 0.7152*g + 0.0722*b
	maxCh := math.Max(r, math.Max(g, b))

	if luma >= 96 && maxCh >= 130 {
		return c
	}

	targetLuma := 112.0
	factor := targetLuma / math.Max(luma, 1)
	if factor < 1 {
		factor = 1
	}
	r *= factor
	g *= factor
	b *= factor

	maxAfter := math.Max(r, math.Max(g, b))
	if maxAfter < 138 {
		boost := 138 / math.Max(maxAfter, 1)
		r *= boost
		g *= boost
		b *= boost
	}

	return model.RGB{
		R: uint8(clampColorInt(int(math.Round(r)), 0, 255)),
		G: uint8(clampColorInt(int(math.Round(g)), 0, 255)),
		B: uint8(clampColorInt(int(math.Round(b)), 0, 255)),
	}
}

func ensureLuminousPalette(colors []model.RGB) []model.RGB {
	if len(colors) == 0 {
		return colors
	}
	out := make([]model.RGB, 0, len(colors))
	for _, c := range colors {
		out = append(out, ensureLuminousColor(c))
	}
	return out
}

func clampColorInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
