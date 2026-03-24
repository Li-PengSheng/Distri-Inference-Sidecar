// Package vramguard implements a GPU VRAM circuit-breaker.
//
// A Guard polls nvidia-smi at a configurable interval. When VRAM utilisation
// rises above OOMThresholdPct the circuit opens: IsOpen returns true and the
// batcher stops enqueuing new requests. The circuit closes automatically once
// utilisation drops back below the threshold.
package vramguard

import (
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Config holds the tuning parameters for the VRAM guard.
type Config struct {
	// PollIntervalMs is how often (in milliseconds) nvidia-smi is queried.
	PollIntervalMs int
	// OOMThresholdPct is the VRAM utilisation percentage at which the
	// circuit-breaker opens. Must be in the range (0, 100].
	OOMThresholdPct float64
}

// Guard monitors GPU VRAM and exposes a circuit-breaker that opens when usage
// exceeds the configured threshold. All fields accessed concurrently use
// atomic operations to avoid data races.
type Guard struct {
	cfg         Config
	circuitOpen atomic.Bool
	// UsedMB holds the most recent used-VRAM reading (float64) from nvidia-smi.
	UsedMB atomic.Value
	// TotalMB holds the most recent total-VRAM reading (float64) from nvidia-smi.
	TotalMB atomic.Value
}

// New allocates a Guard with the given configuration and initialises the VRAM
// counters to zero. Call Start (typically in a goroutine) to begin polling.
func New(cfg Config) *Guard {
	g := &Guard{cfg: cfg}
	g.UsedMB.Store(float64(0))
	g.TotalMB.Store(float64(0))
	return g
}

// IsOpen reports whether the circuit-breaker is currently open (true = VRAM
// pressure is too high; new requests should be rejected).
func (g *Guard) IsOpen() bool {
	return g.circuitOpen.Load()
}

// GetUsage returns the most recent used and total VRAM readings in megabytes.
func (g *Guard) GetUsage() (float64, float64) {
	return g.UsedMB.Load().(float64), g.TotalMB.Load().(float64)
}

// Start polls nvidia-smi at the configured interval and updates the
// circuit-breaker state accordingly. It runs indefinitely; call it via
// go g.Start().
func (g *Guard) Start() {
	ticker := time.NewTicker(time.Duration(g.cfg.PollIntervalMs) * time.Millisecond)
	for range ticker.C {
		used, total, err := queryVRAM()
		if err != nil {
			slog.Error("nvidia-smi query failed", "err", err)
			continue
		}
		g.UsedMB.Store(used)
		g.TotalMB.Store(total)

		pct := (used / total) * 100.0
		if pct >= g.cfg.OOMThresholdPct {
			if !g.circuitOpen.Load() {
				slog.Warn("VRAM guard OPEN — rejecting new requests",
					"pct", pct,
					"used_mb", used,
					"total_mb", total,
				)
				g.circuitOpen.Store(true)
			}
		} else {
			if g.circuitOpen.Load() {
				slog.Info("VRAM guard CLOSED — accepting requests",
					"pct", pct,
				)
				g.circuitOpen.Store(false)
			}
		}
	}
}

// queryVRAM runs nvidia-smi and parses used and total VRAM in megabytes.
// It returns an error if nvidia-smi is unavailable or produces unexpected output.
func queryVRAM() (used, total float64, err error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=memory.used,memory.total",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return 0, 0, err
	}
	parts := strings.Split(strings.TrimSpace(string(out)), ", ")
	if len(parts) != 2 {
		return 0, 0, nil
	}
	used, _ = strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	total, _ = strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	return used, total, nil
}
