// Package metrics registers and exposes Prometheus metrics for the sidecar.
//
// Calling New registers four metrics with the default Prometheus registry and
// starts an HTTP server on :9090 that serves the /metrics scrape endpoint.
package metrics

import (
	"log/slog"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds the Prometheus collectors used across the sidecar.
type Metrics struct {
	// InferLatency tracks the end-to-end backend latency (ms) for each batch flush.
	InferLatency prometheus.Histogram
	// BatchSize tracks how many requests were grouped into each flushed batch.
	BatchSize prometheus.Histogram
	// CircuitBreakerTrips counts requests rejected because the VRAM guard is open.
	CircuitBreakerTrips prometheus.Counter
	// VRAMUsedMB reports the current GPU VRAM consumption in megabytes.
	VRAMUsedMB prometheus.Gauge
}

// New registers all Prometheus metrics and starts the /metrics HTTP server on
// :9090 in a background goroutine. It panics if any metric is already
// registered (double-registration is a programming error).
func New() *Metrics {
	m := &Metrics{
		InferLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "infer_latency_ms",
			Help:    "End-to-end inference latency in milliseconds",
			Buckets: []float64{10, 50, 100, 200, 500, 1000, 2000},
		}),
		BatchSize: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "batch_size",
			Help:    "Number of requests per batch flush",
			Buckets: []float64{1, 4, 8, 16, 32},
		}),
		CircuitBreakerTrips: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "circuit_breaker_trips_total",
			Help: "Requests rejected by VRAM guard",
		}),
		VRAMUsedMB: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "vram_used_mb",
			Help: "Current GPU VRAM usage in MB",
		}),
	}

	prometheus.MustRegister(
		m.InferLatency,
		m.BatchSize,
		m.CircuitBreakerTrips,
		m.VRAMUsedMB,
	)

	// Expose /metrics for Prometheus scraping
	go func() {
		slog.Info("metrics server listening", "addr", ":9090")
		http.Handle("/metrics", promhttp.Handler())
		if err := http.ListenAndServe(":9090", nil); err != nil {
			slog.Error("metrics server failed", "err", err)
		}
	}()

	return m
}
