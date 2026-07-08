// Package main is the entry point for the Distri-Inference-Sidecar.
//
// It wires together three subsystems:
//   - vramguard: polls nvidia-smi and opens a circuit-breaker when GPU VRAM
//     utilisation exceeds the configured threshold.
//   - batcher: collects gRPC inference requests into micro-batches and
//     forwards them as a single HTTP call to the Python backend.
//   - metrics: exposes Prometheus metrics on :9090/metrics.
//
// The process blocks until it receives SIGINT or SIGTERM, then shuts down
// gracefully.
package main

import (
	"log/slog"
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

func main() {
	slog.Info("starting Distri-Inference-Sidecar")

	tokenizer.Init(strings.Repeat("hello world foo bar the quick brown fox ", 200))
	m := metrics.New()

	vg := vramguard.New(vramguard.Config{
		PollIntervalMs:  500,
		OOMThresholdPct: 90.0,
	}, m)
	go vg.Start()

	// Sync VRAM reading into Prometheus gauge every 5s
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		for range ticker.C {
			used, _ := vg.GetUsage()
			m.VRAMUsedMB.Set(used)
		}
	}()

	b := batcher.New(batcher.Config{
		MaxBatchSize:     envInt("MAX_BATCH_SIZE", 8),
		MaxWaitMs:        envInt("MAX_WAIT_MS", 50),
		BackendURL:       os.Getenv("BACKEND_URL"),
		BackendTimeoutMs: envInt("BACKEND_TIMEOUT_MS", 30000),
	}, vg, m)
	go b.Start()

	// Start gRPC server
	srv := grpcserver.New(":50051", b, m)
	go func() {
		if err := srv.Serve(); err != nil {
			slog.Error("gRPC server failed", "err", err)
			os.Exit(1)
		}
	}()

	slog.Info("sidecar ready", "grpc", ":50051", "metrics", ":9090")

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down")
}

func envInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v < 0 {
		slog.Warn("invalid env int, using fallback", "key", key, "value", raw, "fallback", fallback)
		return fallback
	}
	return v
}
