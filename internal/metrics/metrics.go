package metrics

import (
	"log/slog"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Metrics struct {
	InferLatency        prometheus.Histogram
	BatchSize           prometheus.Histogram
	CircuitBreakerTrips prometheus.Counter
	VRAMUsedMB          prometheus.Gauge
}

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
