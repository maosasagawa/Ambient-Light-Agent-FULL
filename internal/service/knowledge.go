package service

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

type KnowledgeService struct {
	entries []string
}

func NewKnowledgeService(path string) (*KnowledgeService, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	entries := make([]string, 0, 64)
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		entries = append(entries, strings.ToLower(line))
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return &KnowledgeService{entries: entries}, nil
}

func (k *KnowledgeService) FindHints(prompt string) []string {
	lp := strings.ToLower(prompt)
	out := make([]string, 0, 8)
	for _, e := range k.entries {
		parts := strings.Split(e, ":")
		if len(parts) < 2 {
			continue
		}
		keywords := strings.Split(parts[0], ",")
		for _, kw := range keywords {
			kw = strings.TrimSpace(kw)
			if kw != "" && strings.Contains(lp, kw) {
				out = append(out, parts[1])
				break
			}
		}
	}
	return out
}

func (k *KnowledgeService) RenderContext(limit int) string {
	if limit <= 0 {
		limit = 8
	}
	if len(k.entries) == 0 {
		return "(none)"
	}
	if limit > len(k.entries) {
		limit = len(k.entries)
	}
	lines := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		parts := strings.SplitN(k.entries[i], ":", 2)
		if len(parts) == 2 {
			lines = append(lines, fmt.Sprintf("- %s => %s", strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])))
			continue
		}
		lines = append(lines, "- "+k.entries[i])
	}
	return strings.Join(lines, "\n")
}
