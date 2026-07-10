package batcher

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/vramguard"
)

type testVRAMReader struct {
	used  float64
	total float64
}

func (r *testVRAMReader) ReadUsageMB() (float64, float64, error) {
	return r.used, r.total, nil
}

func (r *testVRAMReader) Close() {}

func (r *testVRAMReader) Name() string { return "test" }

func startTestBatcher(t *testing.T, cfg Config, handler http.HandlerFunc) *Batcher {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	cfg.BackendURL = srv.URL
	if cfg.BackendTimeoutMs == 0 {
		cfg.BackendTimeoutMs = 5000
	}

	m := metrics.NewForTest()
	vg := vramguard.NewWithReader(
		vramguard.Config{PollIntervalMs: 1000, OOMThresholdPct: 90, CloseThresholdPct: 85},
		m,
		&testVRAMReader{used: 100, total: 10000},
	)
	b := New(cfg, vg, m)
	go b.Start()
	t.Cleanup(b.Stop)
	return b
}

func submitRequest(t *testing.T, b *Batcher, id string) chan Result {
	return submitRequestWithCtx(t, b, id, context.Background())
}

func submitRequestWithCtx(t *testing.T, b *Batcher, id string, ctx context.Context) chan Result {
	t.Helper()
	ch := make(chan Result, 1)
	err := b.Submit(&Request{
		ID:        id,
		InputData: []byte("hello"),
		ModelName: "test-model",
		Ctx:       ctx,
		ResultCh:  ch,
	})
	if err != nil {
		t.Fatalf("submit %s: %v", id, err)
	}
	return ch
}

func waitResult(t *testing.T, ch chan Result) Result {
	t.Helper()
	select {
	case res := <-ch:
		return res
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for batcher result")
		return Result{}
	}
}

func TestBatcher_FullBatchFlush(t *testing.T) {
	var batchCount atomic.Int32

	b := startTestBatcher(t, Config{MaxBatchSize: 3, MaxWaitMs: 5000}, func(w http.ResponseWriter, r *http.Request) {
		batchCount.Add(1)

		var payload struct {
			Requests []struct {
				ID string `json:"id"`
			} `json:"requests"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if len(payload.Requests) != 3 {
			t.Errorf("expected batch size 3, got %d", len(payload.Requests))
		}

		results := make([]map[string]string, len(payload.Requests))
		for i, req := range payload.Requests {
			results[i] = map[string]string{
				"id":          req.ID,
				"output_data": "ok-" + req.ID,
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
	})

	ch1 := submitRequest(t, b, "r1")
	ch2 := submitRequest(t, b, "r2")
	ch3 := submitRequest(t, b, "r3")

	for _, ch := range []chan Result{ch1, ch2, ch3} {
		res := waitResult(t, ch)
		if res.Err != nil {
			t.Fatalf("unexpected error: %v", res.Err)
		}
	}

	if got := batchCount.Load(); got != 1 {
		t.Fatalf("expected 1 backend flush, got %d", got)
	}
}

func TestBatcher_TimeoutFlush(t *testing.T) {
	var batchCount atomic.Int32

	b := startTestBatcher(t, Config{MaxBatchSize: 8, MaxWaitMs: 30}, func(w http.ResponseWriter, r *http.Request) {
		batchCount.Add(1)
		var payload struct {
			Requests []struct {
				ID string `json:"id"`
			} `json:"requests"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]string{{
				"id":          payload.Requests[0].ID,
				"output_data": "timeout-ok",
			}},
		})
	})

	ch := submitRequest(t, b, "solo")
	res := waitResult(t, ch)
	if res.Err != nil {
		t.Fatalf("unexpected error: %v", res.Err)
	}
	if got := batchCount.Load(); got != 1 {
		t.Fatalf("expected 1 backend flush after wait timeout, got %d", got)
	}
}

func TestBatcher_QueueFull(t *testing.T) {
	m := metrics.NewForTest()
	vg := vramguard.NewWithReader(
		vramguard.Config{PollIntervalMs: 1000, OOMThresholdPct: 90, CloseThresholdPct: 85},
		m,
		&testVRAMReader{used: 100, total: 10000},
	)
	b := New(Config{MaxBatchSize: 8, MaxWaitMs: 50, BackendURL: "http://example.invalid"}, vg, m)

	for i := 0; i < 1000; i++ {
		ch := make(chan Result, 1)
		err := b.Submit(&Request{
			ID:        "q-" + strconv.Itoa(i),
			InputData: []byte("x"),
			ModelName: "test",
			ResultCh:  ch,
		})
		if err != nil {
			t.Fatalf("submit %d should succeed, got %v", i, err)
		}
	}

	err := b.Submit(&Request{
		ID:        "overflow",
		InputData: []byte("x"),
		ModelName: "test",
		ResultCh:  make(chan Result, 1),
	})
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
}

func TestBatcher_BackendError(t *testing.T) {
	b := startTestBatcher(t, Config{MaxBatchSize: 1, MaxWaitMs: 0}, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "backend failure", http.StatusInternalServerError)
	})

	res := waitResult(t, submitRequest(t, b, "err1"))
	if res.Err == nil {
		t.Fatal("expected backend error result")
	}
	if !errors.Is(res.Err, ErrBackendUnavailable) {
		t.Fatalf("expected ErrBackendUnavailable, got %v", res.Err)
	}
}

func TestBatcher_BackendInvalidJSONReturnsResponseInvalid(t *testing.T) {
	b := startTestBatcher(t, Config{MaxBatchSize: 1, MaxWaitMs: 0}, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not-json"))
	})

	res := waitResult(t, submitRequest(t, b, "bad-json"))
	if res.Err == nil {
		t.Fatal("expected backend error result")
	}
	if !errors.Is(res.Err, ErrBackendResponseInvalid) {
		t.Fatalf("expected ErrBackendResponseInvalid, got %v", res.Err)
	}
}

func TestBatcher_MismatchedResponseCount(t *testing.T) {
	b := startTestBatcher(t, Config{MaxBatchSize: 2, MaxWaitMs: 0}, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]string{{
				"id":          "a",
				"output_data": "only-one",
			}},
		})
	})

	chA := submitRequest(t, b, "a")
	chB := submitRequest(t, b, "b")

	resA := waitResult(t, chA)
	if resA.Err != nil {
		t.Fatalf("request a unexpected error: %v", resA.Err)
	}

	resB := waitResult(t, chB)
	if resB.Err == nil {
		t.Fatal("expected missing-result error for request b")
	}
}

func TestBatcher_SkipsCancelledRequests(t *testing.T) {
	var batchCount atomic.Int32

	b := startTestBatcher(t, Config{MaxBatchSize: 4, MaxWaitMs: 200}, func(w http.ResponseWriter, r *http.Request) {
		batchCount.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"results": []map[string]string{}})
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res := waitResult(t, submitRequestWithCtx(t, b, "cancelled", ctx))
	if !errors.Is(res.Err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", res.Err)
	}
	if got := batchCount.Load(); got != 0 {
		t.Fatalf("expected no backend flush for cancelled request, got %d", got)
	}
}

func TestBatcher_SkipsEntireBatchWhenAllCancelled(t *testing.T) {
	var batchCount atomic.Int32

	b := startTestBatcher(t, Config{MaxBatchSize: 4, MaxWaitMs: 200}, func(w http.ResponseWriter, r *http.Request) {
		batchCount.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"results": []map[string]string{}})
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ch1 := submitRequestWithCtx(t, b, "c1", ctx)
	ch2 := submitRequestWithCtx(t, b, "c2", ctx)

	for _, ch := range []chan Result{ch1, ch2} {
		res := waitResult(t, ch)
		if !errors.Is(res.Err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", res.Err)
		}
	}
	if got := batchCount.Load(); got != 0 {
		t.Fatalf("expected no backend flush when entire batch cancelled, got %d", got)
	}
}
