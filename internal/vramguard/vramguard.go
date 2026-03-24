package vramguard

import (
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type Config struct {
	PollIntervalMs  int
	OOMThresholdPct float64
}

type Guard struct {
	cfg         Config
	circuitOpen atomic.Bool
	UsedMB      atomic.Value
	TotalMB     atomic.Value
}

func New(cfg Config) *Guard {
	g := &Guard{cfg: cfg}
	g.UsedMB.Store(float64(0))
	g.TotalMB.Store(float64(0))
	return g
}

func (g *Guard) IsOpen() bool {
	return g.circuitOpen.Load()
}

func (g *Guard) GetUsage() (float64, float64) {
	return g.UsedMB.Load().(float64), g.TotalMB.Load().(float64)
}

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
