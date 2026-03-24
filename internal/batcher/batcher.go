package batcher

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/vramguard"
)

type Config struct {
	MaxBatchSize int
	MaxWaitMs    int
	BackendURL   string
}

type Request struct {
	ID        string
	InputData []byte
	ModelName string
	ResultCh  chan Result
}

type Result struct {
	OutputData []byte
	LatencyMs  int64
	Err        error
}

type Batcher struct {
	cfg     Config
	queue   chan *Request
	vg      *vramguard.Guard
	metrics *metrics.Metrics
}

type singleResult struct {
	ID         string `json:"id"`
	OutputData []byte `json:"output_data"`
	Error      string `json:"error"`
}

type batchResponse struct {
	Results []singleResult `json:"results"`
}

func New(cfg Config, vg *vramguard.Guard, m *metrics.Metrics) *Batcher {
	return &Batcher{
		cfg:     cfg,
		queue:   make(chan *Request, 1000),
		vg:      vg,
		metrics: m,
	}
}

// Submit enqueues a request.
func (b *Batcher) Submit(req *Request) error {
	if b.vg.IsOpen() {
		b.metrics.CircuitBreakerTrips.Inc()
		return fmt.Errorf("vram guard: circuit open, try again later")
	}
	b.queue <- req
	return nil
}

func (b *Batcher) Start() {
	slog.Info("batcher started",
		"max_batch_size", b.cfg.MaxBatchSize,
		"max_wait_ms", b.cfg.MaxWaitMs,
	)
	for {
		batch := b.collectBatch()
		if len(batch) == 0 {
			continue // no requests collected, try again
		}
		slog.Debug("flushing batch", "size", len(batch))
		go b.flushBatch(batch)
	}
}

func (b *Batcher) flushBatch(batch []*Request) {
	start := time.Now()

	// Build a map so we can fan results back by ID
	reqMap := make(map[string]*Request, len(batch))
	for _, r := range batch {
		reqMap[r.ID] = r
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

	body, _ := json.Marshal(payload)
	resp, err := http.Post(b.cfg.BackendURL, "application/json", bytes.NewReader(body))
	latencyMs := time.Since(start).Milliseconds()

	b.metrics.InferLatency.Observe(float64(latencyMs))
	b.metrics.BatchSize.Observe(float64(len(batch)))

	if err != nil {
		slog.Error("backend call failed", "err", err, "batch_size", len(batch))
		for _, req := range batch {
			req.ResultCh <- Result{Err: err}
		}
		return
	}
	defer resp.Body.Close()

	var batchResp batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batchResp); err != nil {
		slog.Error("failed to decode backend response", "err", err)
		for _, req := range batch {
			req.ResultCh <- Result{Err: err}
		}
		return
	}

	// Fan each result back to the correct waiting caller by ID
	for _, res := range batchResp.Results {
		req, ok := reqMap[res.ID]
		if !ok {
			slog.Warn("got result for unknown request id", "id", res.ID)
			continue
		}
		if res.Error != "" {
			req.ResultCh <- Result{Err: fmt.Errorf(res.Error)}
		} else {
			req.ResultCh <- Result{
				OutputData: res.OutputData,
				LatencyMs:  latencyMs,
			}
		}
	}

	slog.Debug("batch flushed", "size", len(batch), "latency_ms", latencyMs)
}

// collectBatch waits up to MaxWaitMs OR until MaxBatchSize is reached.
func (b *Batcher) collectBatch() []*Request {
	var batch []*Request
	deadline := time.After(time.Duration(b.cfg.MaxWaitMs) * time.Millisecond)

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

// GetGuard exposes the VRAM guard for health checks.
func (b *Batcher) GetGuard() *vramguard.Guard {
	return b.vg
}
