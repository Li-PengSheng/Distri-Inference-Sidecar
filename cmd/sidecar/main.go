// Package main is the entry point for the Distri-Inference-Sidecar.
//
// It wires together the sidecar subsystems:
//   - tokenizer: trains a Rust BPE vocabulary at startup and admits prompts
//     before they reach the batcher.
//   - vramguard: polls GPU VRAM (NVML preferred, nvidia-smi fallback) and
//     opens a hysteretic circuit-breaker when utilisation exceeds the OOM
//     threshold; the circuit closes only after usage reaches or falls below
//     CloseThresholdPct.
//   - batcher: collects gRPC inference requests into micro-batches and
//     forwards them as a single HTTP call to the Python backend.
//   - grpcserver: serves InferenceService on :50051.
//   - metrics: exposes Prometheus metrics on :9090/metrics.
//
// The process blocks until it receives SIGINT or SIGTERM, then shuts down
// gracefully (gRPC, batcher, VRAM guard, metrics HTTP).
package main

import (
	"context"
	"log/slog"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/batcher"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/grpcserver"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/tokenizer"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/vramguard"
)

const (
	defaultMaxBatchSize     = 8
	defaultMaxWaitMs        = 50
	defaultBackendTimeoutMs = 120000
	defaultPollIntervalMs     = 500
	defaultOOMThresholdPct    = 90.0
	defaultCloseThresholdPct  = 85.0
	shutdownTimeout           = 10 * time.Second
)

// runtimeConfig holds validated environment-derived settings used to wire the
// sidecar subsystems at startup.
type runtimeConfig struct {
	backendURL       string
	maxBatchSize     int
	maxWaitMs        int
	backendTimeoutMs int
	pollIntervalMs   int
	oomThresholdPct  float64
	closeThresholdPct float64
	vramReaderMode   string
}

// main loads configuration, starts tokenizer / metrics / VRAM guard / batcher /
// gRPC, then blocks until SIGINT or SIGTERM and shuts down in reverse order.
func main() {
	cfg := loadAndValidateConfig()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	slog.Info("starting Distri-Inference-Sidecar")

	if err := tokenizer.Init(loadTokenizerCorpus()); err != nil {
		slog.Error("tokenizer initialization failed", "err", err)
		os.Exit(1)
	}
	m := metrics.New()
	metricsSrv := m.StartHTTPServer(":9090")

	vg := vramguard.New(vramguard.Config{
		PollIntervalMs:    cfg.pollIntervalMs,
		OOMThresholdPct:   cfg.oomThresholdPct,
		CloseThresholdPct: cfg.closeThresholdPct,
		ReaderMode:        cfg.vramReaderMode,
	}, m)
	go vg.Start()

	vramGaugeDone := make(chan struct{})
	go func() {
		defer close(vramGaugeDone)
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				used, _ := vg.GetUsage()
				m.VRAMUsedMB.Set(used)
			}
		}
	}()

	b := batcher.New(batcher.Config{
		MaxBatchSize:     cfg.maxBatchSize,
		MaxWaitMs:        cfg.maxWaitMs,
		BackendURL:       cfg.backendURL,
		BackendTimeoutMs: cfg.backendTimeoutMs,
		DebugTokenize:    parseBoolEnv("BATCHER_DEBUG_TOKENIZE"),
		DebugCountTokens: tokenizer.CountTokens,
	}, vg, m)
	go b.Start()

	srv := grpcserver.New(":50051", b, m)
	grpcDone := make(chan error, 1)
	go func() {
		grpcDone <- srv.Serve()
	}()

	slog.Info("sidecar ready", "grpc", ":50051", "metrics", ":9090")

	<-ctx.Done()
	slog.Info("shutting down")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	// Order matters: stop accepting RPCs first, then drain the batcher (which
	// still needs to finish in-flight HTTP flushes), then stop VRAM polling.
	srv.GracefulStop()
	b.Stop()
	vg.Stop()

	if err := metricsSrv.Shutdown(shutdownCtx); err != nil {
		slog.Warn("metrics server shutdown timed out or failed", "err", err)
	}

	select {
	case err := <-grpcDone:
		if err != nil {
			slog.Warn("gRPC server exited with error", "err", err)
		}
	case <-shutdownCtx.Done():
		slog.Warn("gRPC server shutdown timed out")
	}

	select {
	case <-vramGaugeDone:
	case <-time.After(shutdownTimeout):
		slog.Warn("VRAM gauge updater shutdown timed out")
	}

	slog.Info("shutdown complete")
}

// loadTokenizerCorpus returns the BPE training corpus. When
// TOKENIZER_CORPUS_PATH is set the file must be readable and non-empty
// (startup aborts otherwise); without it a small built-in English corpus is
// used, whose token counts poorly approximate real prompts — hence the warning.
func loadTokenizerCorpus() string {
	path := strings.TrimSpace(os.Getenv("TOKENIZER_CORPUS_PATH"))
	if path == "" {
		slog.Warn("TOKENIZER_CORPUS_PATH not set; using built-in toy corpus — " +
			"token counts will not match real model tokenizers")
		return strings.Repeat("hello world foo bar the quick brown fox ", 200)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		slog.Error("failed to read tokenizer corpus", "path", path, "err", err)
		os.Exit(1)
	}
	corpus := string(data)
	if strings.TrimSpace(corpus) == "" {
		slog.Error("tokenizer corpus file is empty", "path", path)
		os.Exit(1)
	}
	slog.Info("tokenizer corpus loaded", "path", path, "bytes", len(data))
	return corpus
}

// parseBoolEnv interprets common truthy environment values ("1", "true",
// "yes", "on", case-insensitive). Any other value, including unset, is false.
func parseBoolEnv(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// loadAndValidateConfig reads required and optional environment variables,
// applies defaults, and aborts the process on invalid values. It validates the
// requested VRAM reader mode but does not guarantee that NVML is available;
// vramguard.New performs its documented runtime fallback to nvidia-smi. See
// docs/configuration.md for the full variable list and operational semantics.
func loadAndValidateConfig() runtimeConfig {
	backendURL := strings.TrimSpace(os.Getenv("BACKEND_URL"))
	if backendURL == "" {
		slog.Error("BACKEND_URL is required")
		os.Exit(1)
	}
	parsedURL, err := url.Parse(backendURL)
	if err != nil || parsedURL.Scheme == "" || parsedURL.Host == "" {
		slog.Error("BACKEND_URL must be a valid URL", "value", backendURL, "err", err)
		os.Exit(1)
	}
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		slog.Error("BACKEND_URL must use http or https scheme", "scheme", parsedURL.Scheme)
		os.Exit(1)
	}

	maxBatchSize := envIntRequired("MAX_BATCH_SIZE", defaultMaxBatchSize)
	if maxBatchSize < 1 {
		slog.Error("MAX_BATCH_SIZE must be >= 1", "value", maxBatchSize)
		os.Exit(1)
	}

	maxWaitMs := envIntRequired("MAX_WAIT_MS", defaultMaxWaitMs)
	if maxWaitMs < 0 {
		slog.Error("MAX_WAIT_MS must be >= 0", "value", maxWaitMs)
		os.Exit(1)
	}

	backendTimeoutMs := envIntRequired("BACKEND_TIMEOUT_MS", defaultBackendTimeoutMs)
	if backendTimeoutMs <= 0 {
		slog.Error("BACKEND_TIMEOUT_MS must be > 0", "value", backendTimeoutMs)
		os.Exit(1)
	}

	pollIntervalMs := envIntRequired("POLL_INTERVAL_MS", defaultPollIntervalMs)
	if pollIntervalMs <= 0 {
		slog.Error("POLL_INTERVAL_MS must be > 0", "value", pollIntervalMs)
		os.Exit(1)
	}

	oomThresholdPct := envFloatRequired("OOM_THRESHOLD_PCT", defaultOOMThresholdPct)
	if oomThresholdPct <= 0 || oomThresholdPct > 100 {
		slog.Error("OOM_THRESHOLD_PCT must be in (0, 100]", "value", oomThresholdPct)
		os.Exit(1)
	}

	closeThresholdPct := envFloatRequired("CLOSE_THRESHOLD_PCT", defaultCloseThresholdPct)
	if closeThresholdPct <= 0 || closeThresholdPct >= oomThresholdPct {
		slog.Error("CLOSE_THRESHOLD_PCT must be in (0, OOM_THRESHOLD_PCT)", "value", closeThresholdPct, "oom_threshold", oomThresholdPct)
		os.Exit(1)
	}

	vramReaderMode := strings.ToLower(strings.TrimSpace(os.Getenv("VRAM_READER_MODE")))
	if vramReaderMode == "" {
		vramReaderMode = "auto"
	}
	switch vramReaderMode {
	case "auto", "nvml", "smi", "nvidia-smi":
	default:
		slog.Error("VRAM_READER_MODE must be auto, nvml, smi, or nvidia-smi", "value", vramReaderMode)
		os.Exit(1)
	}

	return runtimeConfig{
		backendURL:       backendURL,
		maxBatchSize:     maxBatchSize,
		maxWaitMs:        maxWaitMs,
		backendTimeoutMs: backendTimeoutMs,
		pollIntervalMs:    pollIntervalMs,
		oomThresholdPct:   oomThresholdPct,
		closeThresholdPct: closeThresholdPct,
		vramReaderMode:    vramReaderMode,
	}
}

// envIntRequired returns the integer value of key, or fallback when unset.
// Non-integer values cause the process to exit.
func envIntRequired(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		slog.Error("invalid integer environment variable", "key", key, "value", raw, "err", err)
		os.Exit(1)
	}
	return v
}

// envFloatRequired returns the float value of key, or fallback when unset.
// Non-numeric values cause the process to exit.
func envFloatRequired(key string, fallback float64) float64 {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		slog.Error("invalid float environment variable", "key", key, "value", raw, "err", err)
		os.Exit(1)
	}
	return v
}
