// Package metrics registers and exposes Prometheus metrics for the sidecar.
package metrics

import (
	"errors"
	"log/slog"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds the Prometheus collectors used across the sidecar.
type Metrics struct {
	// InferLatency tracks batch flush duration in milliseconds. It starts inside
	// Batcher.flushBatch and therefore excludes time spent waiting in the queue.
	InferLatency prometheus.Histogram
	// BatchSize tracks how many requests were grouped into each flushed batch.
	BatchSize prometheus.Histogram
	// CircuitBreakerTrips counts requests rejected because the VRAM guard is open.
	CircuitBreakerTrips prometheus.Counter
	// VRAMUsedMB reports the latest GPU VRAM consumption in MiB. The field and
	// metric names retain the historical MB suffix for compatibility.
	VRAMUsedMB       prometheus.Gauge
	// RejectedRequests counts prompts rejected by tokenizer admission
	// (token count above MaxInputTokens).
	RejectedRequests prometheus.Counter
	// VRAMPollDurationMs tracks the time spent reading VRAM usage.
	VRAMPollDurationMs prometheus.Histogram
	// VRAMPollErrors counts failures while polling VRAM metrics.
	VRAMPollErrors prometheus.Counter
	// VRAMReaderMode indicates active reader mode (1=active, 0=inactive).
	VRAMReaderMode *prometheus.GaugeVec
	// InferSuccess counts per-request successful inference fan-out results.
	InferSuccess prometheus.Counter
	// InferErrors counts per-request inference failures (backend/batch/fan-out).
	InferErrors prometheus.Counter
	// QueueRejects counts requests rejected because the batch queue is full.
	QueueRejects prometheus.Counter
}

// New registers all Prometheus metrics with the default registry.
func New() *Metrics {
	return registerMetrics(prometheus.DefaultRegisterer)
}

// NewForTest registers metrics with a private registry for unit tests.
func NewForTest() *Metrics {
	return registerMetrics(prometheus.NewRegistry())
}

// registerMetrics constructs collectors and registers them with reg.
// New uses the default registry; NewForTest uses an isolated registry.
func registerMetrics(reg prometheus.Registerer) *Metrics {
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
		RejectedRequests: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "rejected_requests_total",
			Help: "Requests rejected by the tokenizer",
		}),
		VRAMPollDurationMs: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "vram_poll_duration_ms",
			Help:    "Duration of VRAM polling operation in milliseconds",
			Buckets: []float64{0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100},
		}),
		VRAMPollErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vram_poll_errors_total",
			Help: "Total errors encountered during VRAM polling",
		}),
		VRAMReaderMode: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "vram_reader_mode",
			Help: "Active VRAM reader mode (1 active, 0 inactive)",
		}, []string{"mode"}),
		InferSuccess: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "infer_success_total",
			Help: "Total successful inference results returned to callers",
		}),
		InferErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "infer_errors_total",
			Help: "Total inference errors returned to callers",
		}),
		QueueRejects: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "queue_rejects_total",
			Help: "Requests rejected because the batch queue is full",
		}),
	}

	reg.MustRegister(
		m.InferLatency,
		m.BatchSize,
		m.CircuitBreakerTrips,
		m.VRAMUsedMB,
		m.RejectedRequests,
		m.VRAMPollDurationMs,
		m.VRAMPollErrors,
		m.VRAMReaderMode,
		m.InferSuccess,
		m.InferErrors,
		m.QueueRejects,
	)

	return m
}

// StartHTTPServer serves /metrics and a process-liveness /health endpoint on
// addr in a background goroutine, then returns the server for later shutdown.
// /health returns 200 whenever this HTTP server is running; it intentionally
// does not probe the batcher, VRAM reader, Python backend, or Ollama, so it is
// not a readiness or dependency-health check.
func (m *Metrics) StartHTTPServer(addr string) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	srv := &http.Server{Addr: addr, Handler: mux}
	go func() {
		slog.Info("metrics server listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("metrics server failed", "err", err)
		}
	}()
	return srv
}
