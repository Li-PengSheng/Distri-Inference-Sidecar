// Package batcher implements dynamic request batching for inference workloads.
//
// Incoming requests are placed onto an internal queue. A background goroutine
// (Start) drains the queue into batches, each of which is forwarded as a
// single JSON HTTP POST to the configured backend URL. Each caller blocks on
// its own result channel until the batch response is fanned back to it.
//
// The batcher integrates with vramguard: if the GPU circuit-breaker is open,
// Submit returns an error immediately rather than enqueuing the request.
package batcher

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/tokenizer"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/vramguard"
)

// Config holds tuning parameters for the Batcher.
type Config struct {
	// MaxBatchSize is the maximum number of requests collected into one batch
	// before it is flushed immediately, regardless of MaxWaitMs.
	MaxBatchSize int
	// MaxWaitMs is the maximum time in milliseconds the batcher will wait to
	// fill a batch before flushing whatever it has collected so far.
	MaxWaitMs int
	// BackendURL is the HTTP endpoint that receives batched inference requests,
	// e.g. "http://localhost:8000/infer".
	BackendURL string
	// BackendTimeoutMs is the timeout for each backend batch HTTP call.
	BackendTimeoutMs int
}

// Request represents a single inference request submitted by a gRPC caller.
type Request struct {
	// ID is a unique identifier used to correlate the batch response back to
	// this specific request.
	ID string
	// InputData is the raw model input payload (encoding is model-specific).
	InputData []byte
	// ModelName identifies which model the backend should run.
	ModelName string
	// ResultCh receives exactly one Result when the batch containing this
	// request has been processed by the backend.
	ResultCh chan Result
}

// Result carries the outcome of a single inference request.
type Result struct {
	// OutputData is the raw model output returned by the backend.
	OutputData []byte
	// LatencyMs is the end-to-end backend latency for the batch that contained
	// this request, measured in milliseconds.
	LatencyMs int64
	// Err is non-nil when the request or batch failed.
	Err error
}

// Batcher collects inference requests and flushes them in micro-batches to the
// configured HTTP backend.
type Batcher struct {
	cfg        Config
	queue      chan *Request
	vg         *vramguard.Guard
	metrics    *metrics.Metrics
	httpClient *http.Client
	reqCount   atomic.Int64 // requests in last second
	currentQPS atomic.Int64 // updated every second
}

// singleResult holds the backend's response for one request within a batch.
// OutputData is a plain string because the Ollama backend returns text directly.
type singleResult struct {
	ID         string `json:"id"`
	OutputData string `json:"output_data"`
	Error      string `json:"error"`
}

type batchResponse struct {
	Results []singleResult `json:"results"`
}

// New creates and returns a Batcher wired to the given VRAM guard and metrics
// collector. The internal request queue has a capacity of 1 000 entries.
func New(cfg Config, vg *vramguard.Guard, m *metrics.Metrics) *Batcher {
	timeoutMs := cfg.BackendTimeoutMs
	if timeoutMs <= 0 {
		timeoutMs = 30000
	}

	return &Batcher{
		cfg:   cfg,
		queue: make(chan *Request, 1000),
		vg:    vg,
		metrics: m,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutMs) * time.Millisecond,
		},
	}
}

// Submit enqueues a request.
func (b *Batcher) Submit(req *Request) error {
	if b.vg.IsOpen() {
		b.metrics.CircuitBreakerTrips.Inc()
		return fmt.Errorf("vram guard: circuit open, try again later")
	}

	select {
	case b.queue <- req:
		b.reqCount.Add(1)
		return nil
	default:
		b.metrics.QueueRejects.Inc()
		return fmt.Errorf("batch queue full, try again later")
	}
}

// Start runs the batcher's main loop in the calling goroutine. It continuously
// collects batches and dispatches each one to the backend concurrently. This
// method never returns; call it via go b.Start().
func (b *Batcher) Start() {
	slog.Info("batcher started",
		"max_batch_size", b.cfg.MaxBatchSize,
		"max_wait_ms", b.cfg.MaxWaitMs,
	)
	go b.trackQPS()
	for {
		batch := b.collectBatch()
		if len(batch) == 0 {
			continue // no requests collected, try again
		}
		slog.Debug("flushing batch", "size", len(batch))
		go b.flushBatch(batch)
	}
}

// flushBatch serialises the batch into a JSON payload, posts it to the backend,
// and fans each per-request result back through the corresponding ResultCh.
// It is called concurrently (via goroutine) for each collected batch.
func (b *Batcher) flushBatch(batch []*Request) {
	start := time.Now()

	// Build a map so we can fan results back by ID.
	// Slice values preserve all requests even if IDs are duplicated.
	reqMap := make(map[string][]*Request, len(batch))
	for _, r := range batch {
		reqMap[r.ID] = append(reqMap[r.ID], r)
	}

	type singleReq struct {
		ID        string `json:"id"`
		InputData []byte `json:"input_data"`
		ModelName string `json:"model_name"`
	}
	type batchPayload struct {
		Requests []singleReq `json:"requests"`
	}

	payload := batchPayload{}

	for _, req := range batch {
		payload.Requests = append(payload.Requests, singleReq{
			ID:        req.ID,
			InputData: req.InputData,
			ModelName: req.ModelName,
		})
	}

	// Tokenize each request for debug logging before forwarding to the backend.
	for _, req := range batch {
		toks := tokenizer.CountTokens(string(req.InputData))
		slog.Debug("tokenized request", "id", req.ID, "tokens", toks)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		slog.Error("failed to encode backend payload", "err", err)
		b.failBatch(batch, err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), b.httpClient.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.cfg.BackendURL, bytes.NewReader(body))
	if err != nil {
		slog.Error("failed to build backend request", "err", err)
		b.failBatch(batch, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	latencyMs := time.Since(start).Milliseconds()

	b.metrics.InferLatency.Observe(float64(latencyMs))
	b.metrics.BatchSize.Observe(float64(len(batch)))

	if err != nil {
		slog.Error("backend call failed", "err", err, "batch_size", len(batch))
		b.failBatch(batch, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			slog.Error("failed reading non-2xx backend body", "status_code", resp.StatusCode, "err", readErr)
		}
		err = fmt.Errorf("backend returned status %d: %s", resp.StatusCode, string(respBody))
		slog.Error("backend returned non-2xx status", "status_code", resp.StatusCode, "batch_size", len(batch))
		b.failBatch(batch, err)
		return
	}

	var batchResp batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batchResp); err != nil {
		slog.Error("failed to decode backend response", "err", err)
		b.failBatch(batch, err)
		return
	}

	// Fan each result back to the correct waiting caller by ID
	for _, res := range batchResp.Results {
		reqs := reqMap[res.ID]
		if len(reqs) == 0 {
			slog.Warn("got result for unknown request id", "id", res.ID)
			continue
		}
		req := reqs[0]
		reqMap[res.ID] = reqs[1:]

		if res.Error != "" {
			b.metrics.InferErrors.Inc()
			req.ResultCh <- Result{Err: fmt.Errorf("%s", res.Error)}
		} else {
			b.metrics.InferSuccess.Inc()
			req.ResultCh <- Result{
				OutputData: []byte(res.OutputData),
				LatencyMs:  latencyMs,
			}
		}
	}

	// Ensure every queued request gets exactly one terminal result.
	for id, reqs := range reqMap {
		for _, req := range reqs {
			b.metrics.InferErrors.Inc()
			req.ResultCh <- Result{Err: fmt.Errorf("missing backend result for request id %s", id)}
		}
	}

	slog.Debug("batch flushed", "size", len(batch), "latency_ms", latencyMs)
}

func (b *Batcher) failBatch(batch []*Request, err error) {
	for _, req := range batch {
		b.metrics.InferErrors.Inc()
		req.ResultCh <- Result{Err: err}
	}
}

// collectBatch waits up to MaxWaitMs OR until MaxBatchSize is reached.
func (b *Batcher) collectBatch() []*Request {
	var batch []*Request
	deadline := time.After(b.dynamicWaitMs())

	for {
		select {
		case req := <-b.queue:
			batch = append(batch, req)
			if len(batch) >= b.cfg.MaxBatchSize {
				return batch // full batch — flush immediately
			}
		case <-deadline:
			return batch // time up — flush whatever we have
		}
	}
}

// GetGuard exposes the VRAM guard so callers (e.g. the gRPC health-check
// handler) can query circuit-breaker state and VRAM usage directly.
func (b *Batcher) GetGuard() *vramguard.Guard {
	return b.vg
}

// trackQPS counts requests per second and stores the result in currentQPS.
// It is launched as a goroutine by Start and runs for the lifetime of the Batcher.
func (b *Batcher) trackQPS() {
	ticker := time.NewTicker(time.Second)
	for range ticker.C {
		count := b.reqCount.Swap(0)
		b.currentQPS.Store(count)
	}
}

// Dynamic wait: high QPS → wait longer to fill bigger batches
func (b *Batcher) dynamicWaitMs() time.Duration {
	qps := b.currentQPS.Load()
	switch {
	case qps > 100:
		return time.Duration(b.cfg.MaxWaitMs) * time.Millisecond
	case qps > 50:
		return time.Duration(b.cfg.MaxWaitMs/2) * time.Millisecond
	default:
		return time.Duration(b.cfg.MaxWaitMs/4) * time.Millisecond
	}
}
