// Package vramguard implements a GPU VRAM circuit-breaker.
//
// A Guard polls GPU memory at a configurable interval through a pluggable
// Reader (NVML preferred; nvidia-smi as fallback). When VRAM utilisation rises
// above OOMThresholdPct the circuit opens: IsOpen returns true and the batcher
// rejects new requests. The circuit closes only after utilisation drops to or
// below CloseThresholdPct, providing hysteresis so the breaker does not
// flap near the OOM threshold.
package vramguard

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// Config holds the tuning parameters for the VRAM guard.
type Config struct {
	// PollIntervalMs is how often (in milliseconds) the configured Reader is
	// queried for VRAM usage.
	PollIntervalMs int
	// OOMThresholdPct is the VRAM utilisation percentage at which the
	// circuit-breaker opens. Must be in the range (0, 100].
	OOMThresholdPct float64
	// CloseThresholdPct is the VRAM utilisation percentage at which an open
	// circuit closes. Must be less than OOMThresholdPct to provide hysteresis.
	CloseThresholdPct float64
	// ReaderMode selects the VRAM polling backend: auto, nvml, smi, or nvidia-smi.
	ReaderMode string
}

// Reader reports GPU VRAM usage. It is exported for test injection.
type Reader interface {
	// ReadUsageMB returns used and total VRAM in mebibytes for the most
	// utilised GPU (or the sole device). A non-nil error leaves prior
	// Guard readings unchanged.
	ReadUsageMB() (used, total float64, err error)
	// Close releases any native resources held by the reader.
	Close()
	// Name returns a short identifier for metrics and logs (e.g. "nvml").
	Name() string
}

// Guard monitors GPU VRAM and exposes a hysteretic circuit-breaker that opens
// when usage exceeds OOMThresholdPct and closes below CloseThresholdPct.
// All fields accessed concurrently use atomic operations to avoid data races.
type Guard struct {
	cfg         Config
	circuitOpen atomic.Bool
	reader      Reader
	metrics     *metrics.Metrics
	stopCh      chan struct{}
	stopOnce    sync.Once
	// UsedMB holds the most recent used-VRAM reading (float64) in MiB.
	UsedMB atomic.Value
	// TotalMB holds the most recent total-VRAM reading (float64) in MiB.
	TotalMB atomic.Value
}

// nvmlReader reads VRAM via the NVIDIA Management Library for each discovered
// GPU and reports the device with the highest utilisation.
type nvmlReader struct {
	devices []nvml.Device
}

// smiReader reads VRAM by shelling out to nvidia-smi (CSV memory.used/total).
type smiReader struct{}

// New allocates a Guard with the given configuration, selects a Reader from
// ReaderMode, and initialises the VRAM counters to zero. Call Start (typically
// in a goroutine) to begin polling.
func New(cfg Config, m *metrics.Metrics) *Guard {
	return NewWithReader(cfg, m, newVRAMReader(cfg.ReaderMode))
}

// NewWithReader creates a Guard that polls through the provided reader.
// Prefer this constructor in tests that inject a fake Reader.
func NewWithReader(cfg Config, m *metrics.Metrics, reader Reader) *Guard {
	g := &Guard{
		cfg:     cfg,
		reader:  reader,
		metrics: m,
		stopCh:  make(chan struct{}),
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

	for {
		select {
		case <-g.stopCh:
			slog.Info("VRAM guard stopped")
			return
		case <-ticker.C:
			g.pollOnce()
		}
	}
}

// Stop ends the polling loop started by Start.
func (g *Guard) Stop() {
	g.stopOnce.Do(func() {
		close(g.stopCh)
	})
}

// pollOnce performs a single VRAM sample, updates UsedMB/TotalMB, and applies
// open/close hysteresis to the circuit-breaker. Poll failures leave prior
// readings and circuit state unchanged (fail-steady: a flaky reader must not
// flap the breaker open/closed).
func (g *Guard) pollOnce() {
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
		return
	}
	if total <= 0 {
		return
	}
	g.UsedMB.Store(used)
	g.TotalMB.Store(total)

	pct := (used / total) * 100.0
	closeThreshold := g.cfg.CloseThresholdPct
	if closeThreshold <= 0 {
		// Defensive default if misconfigured: 5pp band below the open line.
		closeThreshold = g.cfg.OOMThresholdPct - 5
	}
	if closeThreshold < 0 {
		closeThreshold = 0
	}

	// Hysteresis: open at OOMThresholdPct, close only at CloseThresholdPct.
	// Between the two, state is sticky so usage oscillating near the OOM line
	// does not rapidly alternate reject/accept.
	if pct >= g.cfg.OOMThresholdPct {
		if !g.circuitOpen.Load() {
			slog.Warn("VRAM guard OPEN — rejecting new requests",
				"pct", pct,
				"used_mb", used,
				"total_mb", total,
			)
			g.circuitOpen.Store(true)
		}
	} else if pct <= closeThreshold && g.circuitOpen.Load() {
		slog.Info("VRAM guard CLOSED — accepting requests",
			"pct", pct,
		)
		g.circuitOpen.Store(false)
	}
}

// newVRAMReader selects a Reader for mode: "smi"/"nvidia-smi" force CLI
// polling; "nvml" prefers NVML with smi fallback; "auto" (default) tries NVML
// first then falls back to nvidia-smi.
func newVRAMReader(mode string) Reader {
	mode = strings.ToLower(strings.TrimSpace(mode))
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

// newNVMLReader initialises NVML and collects device handles. The second
// return value is false when NVML is unavailable or no GPUs are found; the
// caller should fall back to smiReader.
func newNVMLReader() (*nvmlReader, bool) {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		slog.Warn("NVML init failed", "err", nvml.ErrorString(ret))
		return nil, false
	}

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS || count == 0 {
		slog.Warn("NVML device enumeration failed", "count", count, "err", nvml.ErrorString(ret))
		nvml.Shutdown()
		return nil, false
	}

	devices := make([]nvml.Device, 0, count)
	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			slog.Warn("NVML device lookup failed", "index", i, "err", nvml.ErrorString(ret))
			continue
		}
		devices = append(devices, device)
	}
	if len(devices) == 0 {
		nvml.Shutdown()
		return nil, false
	}

	slog.Info("NVML reader initialized", "gpu_count", len(devices))
	return &nvmlReader{devices: devices}, true
}

// Name implements Reader.
func (r *nvmlReader) Name() string { return "nvml" }

// ReadUsageMB reports the reading of the most-utilised GPU so the circuit
// breaker reacts to whichever device is closest to OOM.
func (r *nvmlReader) ReadUsageMB() (used, total float64, err error) {
	const mib = 1024 * 1024
	found := false
	for i, device := range r.devices {
		mem, ret := nvml.DeviceGetMemoryInfo(device)
		if ret != nvml.SUCCESS {
			slog.Warn("nvml DeviceGetMemoryInfo failed", "index", i, "err", nvml.ErrorString(ret))
			continue
		}
		if mem.Total == 0 {
			continue
		}
		u := float64(mem.Used) / mib
		t := float64(mem.Total) / mib
		if !found || u/t > used/total {
			used, total = u, t
			found = true
		}
	}
	if !found {
		return 0, 0, fmt.Errorf("nvml: no GPU produced a valid memory reading")
	}
	return used, total, nil
}

// Close shuts down the NVML library. Safe to call once when the Guard stops.
func (r *nvmlReader) Close() {
	ret := nvml.Shutdown()
	if ret != nvml.SUCCESS {
		slog.Warn("NVML shutdown failed", "err", nvml.ErrorString(ret))
	}
}

// Name implements Reader.
func (r *smiReader) Name() string { return "nvidia-smi" }

// ReadUsageMB implements Reader by invoking nvidia-smi.
func (r *smiReader) ReadUsageMB() (used, total float64, err error) {
	return queryVRAMViaSMI()
}

// Close implements Reader; nvidia-smi has no persistent handle to release.
func (r *smiReader) Close() {}

// smiTimeout bounds each nvidia-smi invocation so a hung driver cannot stall
// the polling loop indefinitely.
const smiTimeout = 5 * time.Second

// queryVRAMViaSMI runs nvidia-smi and parses used and total VRAM in megabytes.
// It returns an error if nvidia-smi is unavailable, times out, or produces
// unexpected output; the caller (pollOnce) keeps the last known readings.
func queryVRAMViaSMI() (used, total float64, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), smiTimeout)
	defer cancel()

	out, err := exec.CommandContext(ctx,
		"nvidia-smi",
		"--query-gpu=memory.used,memory.total",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		if ctx.Err() != nil {
			return 0, 0, fmt.Errorf("nvidia-smi timed out after %s: %w", smiTimeout, err)
		}
		return 0, 0, fmt.Errorf("nvidia-smi failed: %w", err)
	}
	return parseSMIOutput(strings.TrimSpace(string(out)))
}

// parseSMIOutput parses nvidia-smi CSV output. Multi-GPU machines emit one
// line per device; the reading of the most-utilised GPU is returned so the
// circuit breaker reacts to whichever device is closest to OOM.
func parseSMIOutput(output string) (used, total float64, err error) {
	found := false
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		u, t, err := parseSMILine(line)
		if err != nil {
			return 0, 0, err
		}
		if t <= 0 {
			continue
		}
		if !found || u/t > used/total {
			used, total = u, t
			found = true
		}
	}
	if !found {
		return 0, 0, fmt.Errorf("nvidia-smi produced no valid GPU readings: %q", output)
	}
	return used, total, nil
}

// parseSMILine parses one nvidia-smi CSV line of the form "used, total"
// (noheader, nounits). Both values are megabytes.
func parseSMILine(line string) (used, total float64, err error) {
	parts := strings.Split(line, ", ")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("unexpected nvidia-smi output line: %q", line)
	}
	used, err = strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil {
		slog.Warn("failed to parse nvidia-smi memory.used", "value", parts[0], "err", err)
		return 0, 0, fmt.Errorf("parse memory.used: %w", err)
	}
	total, err = strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		slog.Warn("failed to parse nvidia-smi memory.total", "value", parts[1], "err", err)
		return 0, 0, fmt.Errorf("parse memory.total: %w", err)
	}
	return used, total, nil
}
