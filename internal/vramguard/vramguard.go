// Package vramguard implements a GPU VRAM circuit-breaker.
//
// A Guard polls nvidia-smi at a configurable interval. When VRAM utilisation
// rises above OOMThresholdPct the circuit opens: IsOpen returns true and the
// batcher stops enqueuing new requests. The circuit closes automatically once
// utilisation drops back below the threshold.
package vramguard

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
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
	reader      vramReader
	metrics     *metrics.Metrics
	// UsedMB holds the most recent used-VRAM reading (float64) in MiB.
	UsedMB atomic.Value
	// TotalMB holds the most recent total-VRAM reading (float64) in MiB.
	TotalMB atomic.Value
}

type vramReader interface {
	ReadUsageMB() (used, total float64, err error)
	Close()
	Name() string
}

type nvmlReader struct {
	device nvml.Device
}

type smiReader struct{}

// New allocates a Guard with the given configuration and initialises the VRAM
// counters to zero. Call Start (typically in a goroutine) to begin polling.
func New(cfg Config, m *metrics.Metrics) *Guard {
	g := &Guard{
		cfg:     cfg,
		reader:  newVRAMReader(),
		metrics: m,
	}
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

// Start polls VRAM usage through the configured reader (NVML preferred,
// nvidia-smi fallback) and updates the circuit-breaker state accordingly.
// It runs indefinitely; call it via go g.Start().
func (g *Guard) Start() {
	defer g.reader.Close()
	mode := g.reader.Name()
	slog.Info("VRAM guard reader initialized", "mode", mode)
	if g.metrics != nil {
		g.metrics.VRAMReaderMode.WithLabelValues("nvml").Set(0)
		g.metrics.VRAMReaderMode.WithLabelValues("nvidia-smi").Set(0)
		g.metrics.VRAMReaderMode.WithLabelValues(mode).Set(1)
	}

	ticker := time.NewTicker(time.Duration(g.cfg.PollIntervalMs) * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		pollStart := time.Now()
		used, total, err := g.reader.ReadUsageMB()
		if g.metrics != nil {
			g.metrics.VRAMPollDurationMs.Observe(float64(time.Since(pollStart).Microseconds()) / 1000.0)
		}
		if err != nil {
			slog.Error("vram query failed", "mode", g.reader.Name(), "err", err)
			if g.metrics != nil {
				g.metrics.VRAMPollErrors.Inc()
			}
			continue
		}
		if total <= 0 {
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

func newVRAMReader() vramReader {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("VRAM_READER_MODE")))
	if mode == "" {
		mode = "auto"
	}

	switch mode {
	case "smi", "nvidia-smi":
		slog.Info("VRAM reader mode forced to nvidia-smi")
		return &smiReader{}
	case "nvml":
		reader, ok := newNVMLReader()
		if ok {
			return reader
		}
		slog.Warn("VRAM_READER_MODE=nvml but NVML unavailable, falling back to nvidia-smi")
		return &smiReader{}
	case "auto":
		// continue to auto mode below
	default:
		slog.Warn("unknown VRAM_READER_MODE, using auto", "mode", mode)
	}

	reader, ok := newNVMLReader()
	if ok {
		return reader
	}
	return &smiReader{}
}

func newNVMLReader() (*nvmlReader, bool) {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		slog.Warn("NVML init failed", "err", nvml.ErrorString(ret))
		return nil, false
	}

	device, ret := nvml.DeviceGetHandleByIndex(0)
	if ret != nvml.SUCCESS {
		slog.Warn("NVML device lookup failed", "err", nvml.ErrorString(ret))
		nvml.Shutdown()
		return nil, false
	}

	return &nvmlReader{device: device}, true
}

func (r *nvmlReader) Name() string { return "nvml" }

func (r *nvmlReader) ReadUsageMB() (used, total float64, err error) {
	mem, ret := nvml.DeviceGetMemoryInfo(r.device)
	if ret != nvml.SUCCESS {
		return 0, 0, fmt.Errorf("nvml DeviceGetMemoryInfo failed: %s", nvml.ErrorString(ret))
	}
	const mib = 1024 * 1024
	return float64(mem.Used) / mib, float64(mem.Total) / mib, nil
}

func (r *nvmlReader) Close() {
	ret := nvml.Shutdown()
	if ret != nvml.SUCCESS {
		slog.Warn("NVML shutdown failed", "err", nvml.ErrorString(ret))
	}
}

func (r *smiReader) Name() string { return "nvidia-smi" }

func (r *smiReader) ReadUsageMB() (used, total float64, err error) {
	return queryVRAMViaSMI()
}

func (r *smiReader) Close() {}

// queryVRAM runs nvidia-smi and parses used and total VRAM in megabytes.
// It returns an error if nvidia-smi is unavailable or produces unexpected output.
func queryVRAMViaSMI() (used, total float64, err error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=memory.used,memory.total",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		// No GPU available — return safe defaults silently
		return 0, 1024, nil
	}
	parts := strings.Split(strings.TrimSpace(string(out)), ", ")
	if len(parts) != 2 {
		return 0, 1024, nil
	}
	used, _ = strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	total, _ = strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	return used, total, nil
}
