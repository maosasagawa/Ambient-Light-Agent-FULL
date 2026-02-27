package ws

import (
	"encoding/json"
	"sync"

	"ambient-light-agent/internal/model"
	"github.com/gorilla/websocket"
)

type HardwareHub struct {
	mu      sync.RWMutex
	clients map[string]map[*Client]struct{}
}

func NewHardwareHub() *HardwareHub {
	return &HardwareHub{clients: map[string]map[*Client]struct{}{}}
}

func (h *HardwareHub) Register(vehicleID string, conn *websocket.Conn) *Client {
	var c *Client
	c = NewClientWithClose(conn, func() { h.Unregister(vehicleID, c) })
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.clients[vehicleID]; !ok {
		h.clients[vehicleID] = map[*Client]struct{}{}
	}
	h.clients[vehicleID][c] = struct{}{}
	return c
}

func (h *HardwareHub) Unregister(vehicleID string, c *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if m, ok := h.clients[vehicleID]; ok {
		if _, exist := m[c]; exist {
			delete(m, c)
			close(c.send)
		}
		if len(m) == 0 {
			delete(h.clients, vehicleID)
		}
	}
}

func (h *HardwareHub) PushEnvelope(env model.HardwareEnvelope) {
	b, err := json.Marshal(map[string]interface{}{
		"type":       "hardware.envelope",
		"created_at": env.CreatedAt,
		"payload":    env,
	})
	if err != nil {
		return
	}

	h.mu.RLock()
	clients := h.clients[env.VehicleID]
	h.mu.RUnlock()
	for c := range clients {
		select {
		case c.send <- b:
		default:
			h.Unregister(env.VehicleID, c)
		}
	}
}
