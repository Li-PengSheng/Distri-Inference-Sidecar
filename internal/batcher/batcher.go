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
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
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
	// DebugTokenize enables per-request token counting in flushBatch logs.
	DebugTokenize bool
	// DebugCountTokens counts tokens when DebugTokenize is enabled.
	DebugCountTokens func(string) int
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
	// Ctx carries the caller context for cancellation propagation. Checked in
	// flushBatch before the backend call; not used to cancel the batch HTTP
	// request itself (one cancelled client must not abort siblings).
	Ctx context.Context
	// ResultCh receives exactly one Result when the batch containing this
	// request has been processed (or failed). Callers should use a buffer of
	// at least 1 so a departed client does not block fan-out.
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
	stopCh     chan struct{}
	stopOnce   sync.Once
	wg         sync.WaitGroup
	// stopMu guards stopped and makes Stop mutually exclusive with in-flight
	// Submit enqueues, so the queue never receives requests after shutdown
	// has begun (the queue channel itself is never closed).
	stopMu  sync.RWMutex
	stopped bool
}

// singleResult holds the backend's response for one request within a batch.
// OutputData is a plain string because the Ollama backend returns text directly.
type singleResult struct {
	ID         string `json:"id"`
	OutputData string `json:"output_data"`
	Error      string `json:"error"`
}

// batchResponse is the JSON envelope returned by the Python backend /infer
// endpoint: one singleResult per submitted request id.
type batchResponse struct {
	Results []singleResult `json:"results"`
}

// New creates and returns a Batcher wired to the given VRAM guard and metrics
// collector. The internal request queue has a capacity of 1 000 entries.
func New(cfg Config, vg *vramguard.Guard, m *metrics.Metrics) *Batcher {
	timeoutMs := cfg.BackendTimeoutMs
	if timeoutMs <= 0 {
		timeoutMs = 120000
	}

	return &Batcher{
		cfg:   cfg,
		queue: make(chan *Request, 1000),
		vg:    vg,
		metrics: m,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutMs) * time.Millisecond,
		},
		stopCh: make(chan struct{}),
	}
}

// Submit enqueues a request for the next micro-batch. It returns immediately
// with one of:
//   - ErrCircuitOpen when the VRAM guard circuit-breaker is open
//   - ErrBatcherStopped when Stop has already been called
//   - ErrQueueFull when the internal queue (capacity 1000) cannot accept more
//
// On success the caller must wait on req.ResultCh for exactly one Result.
func (b *Batcher) Submit(req *Request) error {
	// Fast-path reject under VRAM pressure before taking stopMu or touching
	// the queue — avoids amplifying load when the circuit is already open.
	if b.vg.IsOpen() {
		b.metrics.CircuitBreakerTrips.Inc()
		return ErrCircuitOpen
	}

	// RLock spans the enqueue so Stop's write-lock cannot flip stopped=true
	// between the check and the send (which would leave a post-shutdown
	// request sitting in the queue with no consumer guarantee).
	b.stopMu.RLock()
	defer b.stopMu.RUnlock()
	if b.stopped {
		return ErrBatcherStopped
	}

	select {
	case b.queue <- req:
		b.reqCount.Add(1)
		return nil
	default:
		// Non-blocking: a full queue must fail fast rather than stall the
		// gRPC handler until capacity frees up.
		b.metrics.QueueRejects.Inc()
		return ErrQueueFull
	}
}

// Start runs the batcher's main loop in the calling goroutine. It continuously
// collects batches and dispatches each one to the backend concurrently.
func (b *Batcher) Start() {
	slog.Info("batcher started",
		"max_batch_size", b.cfg.MaxBatchSize,
		"max_wait_ms", b.cfg.MaxWaitMs,
	)
	b.wg.Add(1)
	go func() {
		defer b.wg.Done()
		b.trackQPS()
	}()

	for {
		// Non-blocking peek: collectBatch also watches stopCh, so an empty
		// return after stop still drains below. This early check avoids one
		// extra collect wait when shutdown is already signalled.
		select {
		case <-b.stopCh:
			b.drainQueueWithError(fmt.Errorf("batcher stopped"))
			return
		default:
		}

		batch := b.collectBatch()
		if len(batch) == 0 {
			if b.isStopped() {
				b.drainQueueWithError(fmt.Errorf("batcher stopped"))
				return
			}
			// Idle timeout with nothing queued — loop and wait again.
			continue
		}

		slog.Debug("flushing batch", "size", len(batch))
		// Flush concurrently so the next collectBatch can start while this
		// HTTP round-trip is in flight (pipeline multiple batches).
		b.wg.Add(1)
		go func(batch []*Request) {
			defer b.wg.Done()
			b.flushBatch(batch)
		}(batch)
	}
}

// Stop ends queue processing and waits for in-flight batch flushes to finish.
// Pending requests receive a terminal error on ResultCh.
//
// The queue channel is intentionally never closed: closing it would race with
// concurrent Submit calls (send on closed channel panics). Instead, Stop marks
// the batcher stopped under stopMu — waiting out any in-flight enqueue — and
// then signals the main loop via stopCh to drain whatever remains.
func (b *Batcher) Stop() {
	b.stopOnce.Do(func() {
		b.stopMu.Lock()
		b.stopped = true
		b.stopMu.Unlock()
		close(b.stopCh)
	})
	b.wg.Wait()
}

// isStopped reports whether Stop has signalled the main loop via stopCh.
func (b *Batcher) isStopped() bool {
	select {
	case <-b.stopCh:
		return true
	default:
		return false
	}
}

// drainQueueWithError non-blockingly empties the queue, delivering err to each
// pending request. Used during shutdown so callers are not left waiting.
func (b *Batcher) drainQueueWithError(err error) {
	for {
		select {
		case req := <-b.queue:
			b.failRequest(req, err)
		default:
			return
		}
	}
}

// failRequest increments InferErrors and delivers a terminal Result with Err.
func (b *Batcher) failRequest(req *Request, err error) {
	b.metrics.InferErrors.Inc()
	req.ResultCh <- Result{Err: err}
}

// flushBatch serialises the batch into a JSON payload, posts it to the backend,
// and fans each per-request result back through the corresponding ResultCh.
// It is called concurrently (via goroutine) for each collected batch.
func (b *Batcher) flushBatch(batch []*Request) {
	start := time.Now()

	// Drop callers that already cancelled while waiting in the queue / collect
	// window so we do not spend GPU time on abandoned prompts. Fail them with
	// the context error so statusFromResultErr can map Canceled/DeadlineExceeded.
	active := make([]*Request, 0, len(batch))
	for _, req := range batch {
		if req.Ctx != nil {
			if err := req.Ctx.Err(); err != nil {
				b.failRequest(req, err)
				continue
			}
		}
		active = append(active, req)
	}
	if len(active) == 0 {
		return
	}
	batch = active

	// Key by ID for fan-out. Slice values keep FIFO order when clients reuse
	// the same RequestId within one batch (otherwise a map[string]*Request
	// would silently drop all but one waiter).
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

	if b.cfg.DebugTokenize && b.cfg.DebugCountTokens != nil {
		for _, req := range batch {
			toks := b.cfg.DebugCountTokens(string(req.InputData))
			slog.Debug("tokenized request", "id", req.ID, "tokens", toks)
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		slog.Error("failed to encode backend payload", "err", err)
		b.failBatch(batch, wrapResponseInvalid(err))
		return
	}

	// Detached from any single caller ctx: one cancelled client must not abort
	// the whole batch HTTP call. Timeout is bounded by httpClient.Timeout.
	ctx, cancel := context.WithTimeout(context.Background(), b.httpClient.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.cfg.BackendURL, bytes.NewReader(body))
	if err != nil {
		slog.Error("failed to build backend request", "err", err)
		b.failBatch(batch, wrapResponseInvalid(err))
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	latencyMs := time.Since(start).Milliseconds()

	b.metrics.InferLatency.Observe(float64(latencyMs))
	b.metrics.BatchSize.Observe(float64(len(batch)))

	if err != nil {
		slog.Error("backend call failed", "err", err, "batch_size", len(batch))
		b.failBatch(batch, wrapUnavailable(err))
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
		// Transport-level failure of the whole batch — same sentinel for every
		// waiter so gRPC maps it to Unavailable.
		b.failBatch(batch, wrapUnavailable(err))
		return
	}

	var batchResp batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batchResp); err != nil {
		slog.Error("failed to decode backend response", "err", err)
		b.failBatch(batch, wrapResponseInvalid(err))
		return
	}

	// Match results by ID in arrival order; pop from the front of each slice
	// so duplicate IDs each receive one response.
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
			// Per-item backend error string (not a sentinel) — gRPC will put
			// this in InferResponse.Error rather than an RPC status.
			req.ResultCh <- Result{Err: fmt.Errorf("%s", res.Error)}
		} else {
			b.metrics.InferSuccess.Inc()
			req.ResultCh <- Result{
				OutputData: []byte(res.OutputData),
				LatencyMs:  latencyMs,
			}
		}
	}

	// Backend omitted some IDs — still unblock every waiter exactly once.
	for id, reqs := range reqMap {
		for _, req := range reqs {
			b.metrics.InferErrors.Inc()
			req.ResultCh <- Result{Err: fmt.Errorf("missing backend result for request id %s", id)}
		}
	}

	slog.Debug("batch flushed", "size", len(batch), "latency_ms", latencyMs)
}

// failBatch delivers the same terminal error to every request in batch.
func (b *Batcher) failBatch(batch []*Request, err error) {
	for _, req := range batch {
		b.failRequest(req, err)
	}
}

// collectBatch waits up to dynamicWaitMs() or until MaxBatchSize is reached,
// whichever comes first. An empty slice may be returned on stop or timeout
// with nothing collected.
func (b *Batcher) collectBatch() []*Request {
	var batch []*Request
	// Timer starts when collection begins; under high QPS dynamicWaitMs grows
	// so more requests can join before flush.
	deadline := time.After(b.dynamicWaitMs())

	for {
		select {
		case req := <-b.queue:
			batch = append(batch, req)
			if len(batch) >= b.cfg.MaxBatchSize {
				return batch // full batch — flush immediately
			}
		case <-deadline:
			return batch // time up — flush whatever we have (may be empty)
		case <-b.stopCh:
			// Return partial batch to the Start loop so in-flight items still
			// get flushed (or drained) rather than being dropped silently.
			return batch
		}
	}
}

// GetGuard exposes the VRAM guard so callers (e.g. the gRPC health-check
// handler) can query circuit-breaker state and VRAM usage directly.
func (b *Batcher) GetGuard() *vramguard.Guard {
	return b.vg
}

// trackQPS counts requests per second and stores the result in currentQPS.
func (b *Batcher) trackQPS() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-b.stopCh:
			return
		case <-ticker.C:
			count := b.reqCount.Swap(0)
			b.currentQPS.Store(count)
		}
	}
}

// minCollectWait bounds the batch collection window from below. Without it,
// MaxWaitMs values below 4 produce a zero wait (integer division), turning the
// collect loop into a CPU-burning busy loop on an idle queue.
const minCollectWait = time.Millisecond

// dynamicWaitMs returns the collection window for the next batch. Higher
// observed QPS lengthens the wait (up to MaxWaitMs) so batches fill larger;
// lower QPS shortens it for lower latency. Thresholds (50 / 100 QPS) are
// coarse heuristics tied to currentQPS from trackQPS. The result is never
// below minCollectWait, which prevents a busy-loop when MaxWaitMs is very
// small (integer division would otherwise yield 0).
func (b *Batcher) dynamicWaitMs() time.Duration {
	qps := b.currentQPS.Load()
	var wait time.Duration
	switch {
	case qps > 100:
		wait = time.Duration(b.cfg.MaxWaitMs) * time.Millisecond
	case qps > 50:
		wait = time.Duration(b.cfg.MaxWaitMs/2) * time.Millisecond
	default:
		wait = time.Duration(b.cfg.MaxWaitMs/4) * time.Millisecond
	}
	if wait < minCollectWait {
		return minCollectWait
	}
	return wait
}

// wrapUnavailable wraps err with ErrBackendUnavailable unless it already is.
func wrapUnavailable(err error) error {
	if errors.Is(err, ErrBackendUnavailable) {
		return err
	}
	return fmt.Errorf("%w: %v", ErrBackendUnavailable, err)
}

// wrapResponseInvalid wraps err with ErrBackendResponseInvalid unless it
// already is (payload encode/decode failures).
func wrapResponseInvalid(err error) error {
	if errors.Is(err, ErrBackendResponseInvalid) {
		return err
	}
	return fmt.Errorf("%w: %v", ErrBackendResponseInvalid, err)
}
