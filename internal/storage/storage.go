package storage

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"time"

	"ambient-light-agent/internal/model"
)

type Store struct {
	path  string
	mu    sync.RWMutex
	state model.StoredState
}

func NewStore(path string) (*Store, error) {
	if path == "" {
		return nil, errors.New("store path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	s := &Store{path: path}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	b, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			s.state = defaultState()
			return s.saveLocked()
		}
		return err
	}
	if len(b) == 0 {
		s.state = defaultState()
		return s.saveLocked()
	}

	var state model.StoredState
	if err := json.Unmarshal(b, &state); err != nil {
		return err
	}
	mergeDefaults(&state)
	s.state = state
	return nil
}

func defaultState() model.StoredState {
	now := time.Now().UTC()
	return model.StoredState{
		FavoriteScripts:  []model.AnimationScript{},
		PromptTemplates:  map[string]model.PromptTemplate{},
		UserPreferences:  map[string]model.UserPreference{},
		LastPlanByUser:   map[string]model.IntentPlan{},
		LastFramesByUser: map[string][]model.MatrixFrame{},
		VehicleZones:     map[string][]model.VehicleStripZone{},
		VehicleZoneCmds:  map[string]map[string]model.StripCmd{},
		HardwareOutbox:   map[string]model.HardwareEnvelope{},
		CreatedAt:        now,
	}
}

func mergeDefaults(state *model.StoredState) {
	if state.FavoriteScripts == nil {
		state.FavoriteScripts = []model.AnimationScript{}
	}
	if state.PromptTemplates == nil {
		state.PromptTemplates = map[string]model.PromptTemplate{}
	}
	if state.UserPreferences == nil {
		state.UserPreferences = map[string]model.UserPreference{}
	}
	if state.LastPlanByUser == nil {
		state.LastPlanByUser = map[string]model.IntentPlan{}
	}
	if state.LastFramesByUser == nil {
		state.LastFramesByUser = map[string][]model.MatrixFrame{}
	}
	if state.VehicleZones == nil {
		state.VehicleZones = map[string][]model.VehicleStripZone{}
	}
	if state.VehicleZoneCmds == nil {
		state.VehicleZoneCmds = map[string]map[string]model.StripCmd{}
	}
	if state.HardwareOutbox == nil {
		state.HardwareOutbox = map[string]model.HardwareEnvelope{}
	}
	if state.CreatedAt.IsZero() {
		state.CreatedAt = time.Now().UTC()
	}
}

func (s *Store) saveLocked() error {
	s.state.LastUpdatedUnixMS = time.Now().UnixMilli()
	b, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, b, 0o600)
}

func (s *Store) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saveLocked()
}

func (s *Store) Snapshot() model.StoredState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	b, _ := json.Marshal(s.state)
	var cloned model.StoredState
	_ = json.Unmarshal(b, &cloned)
	return cloned
}

func (s *Store) UpsertTemplate(t model.PromptTemplate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.PromptTemplates[t.ID] = t
	return s.saveLocked()
}

func (s *Store) ListTemplates() []model.PromptTemplate {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]model.PromptTemplate, 0, len(s.state.PromptTemplates))
	for _, t := range s.state.PromptTemplates {
		out = append(out, t)
	}
	return out
}

func (s *Store) SetLatestMatrix(m model.Matrix) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.LatestMatrix = &m
	return s.saveLocked()
}

func (s *Store) GetLatestMatrix() *model.Matrix {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.state.LatestMatrix == nil {
		return nil
	}
	m := *s.state.LatestMatrix
	return &m
}

func (s *Store) SetLatestStrip(cmd model.StripCmd) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.LatestStripCmd = &cmd
	return s.saveLocked()
}

func (s *Store) SetLatestStripFrame(frame model.StripFrame) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.LatestStripFrame = &frame
	return s.saveLocked()
}

func (s *Store) GetLatestStripFrame() *model.StripFrame {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.state.LatestStripFrame == nil {
		return nil
	}
	f := *s.state.LatestStripFrame
	return &f
}

func (s *Store) GetLatestStrip() *model.StripCmd {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.state.LatestStripCmd == nil {
		return nil
	}
	c := *s.state.LatestStripCmd
	return &c
}

func (s *Store) AddFavoriteScript(script model.AnimationScript) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.FavoriteScripts = append(s.state.FavoriteScripts, script)
	return s.saveLocked()
}

func (s *Store) ListFavoriteScripts() []model.AnimationScript {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]model.AnimationScript, len(s.state.FavoriteScripts))
	copy(out, s.state.FavoriteScripts)
	return out
}

func (s *Store) UpsertPreference(pref model.UserPreference) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.UserPreferences[pref.UserID] = pref
	return s.saveLocked()
}

func (s *Store) GetPreference(userID string) *model.UserPreference {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p, ok := s.state.UserPreferences[userID]
	if !ok {
		return nil
	}
	cp := p
	return &cp
}

func (s *Store) SetPlan(userID string, plan model.IntentPlan) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.LastPlanByUser[userID] = plan
	return s.saveLocked()
}

func (s *Store) SetFrames(userID string, frames []model.MatrixFrame) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.LastFramesByUser[userID] = frames
	return s.saveLocked()
}

func (s *Store) UpsertVehicleZones(vehicleID string, zones []model.VehicleStripZone) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.VehicleZones[vehicleID] = zones
	return s.saveLocked()
}

func (s *Store) GetVehicleZones(vehicleID string) []model.VehicleStripZone {
	s.mu.RLock()
	defer s.mu.RUnlock()
	z := s.state.VehicleZones[vehicleID]
	out := make([]model.VehicleStripZone, len(z))
	copy(out, z)
	return out
}

func (s *Store) SetVehicleZoneCmd(vehicleID, zoneID string, cmd model.StripCmd) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.state.VehicleZoneCmds[vehicleID]; !ok {
		s.state.VehicleZoneCmds[vehicleID] = map[string]model.StripCmd{}
	}
	s.state.VehicleZoneCmds[vehicleID][zoneID] = cmd
	return s.saveLocked()
}

func (s *Store) GetVehicleZoneCmds(vehicleID string) map[string]model.StripCmd {
	s.mu.RLock()
	defer s.mu.RUnlock()
	in := s.state.VehicleZoneCmds[vehicleID]
	out := make(map[string]model.StripCmd, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func (s *Store) SetHardwareEnvelope(vehicleID string, env model.HardwareEnvelope) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.HardwareOutbox[vehicleID] = env
	return s.saveLocked()
}

func (s *Store) GetHardwareEnvelope(vehicleID string) *model.HardwareEnvelope {
	s.mu.RLock()
	defer s.mu.RUnlock()
	env, ok := s.state.HardwareOutbox[vehicleID]
	if !ok {
		return nil
	}
	cp := env
	return &cp
}
