package main

import (
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/batcher"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/vramguard"
)

func main() {
	slog.Info("starting Distri-Inference-Sidecar")

	m := metrics.New()

	vg := vramguard.New(vramguard.Config{
		PollIntervalMs:  500,
		OOMThresholdPct: 90.0,
	})
	go vg.Start()

	b := batcher.New(batcher.Config{
		MaxBatchSize: 32,
		MaxWaitMs:    20,
		BackendURL:   "http://localhost:8000/infer",
	}, vg, m)
	go b.Start()

	slog.Info("sidecar ready", "grpc", ":50051", "metrics", ":9090")

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down")
}
