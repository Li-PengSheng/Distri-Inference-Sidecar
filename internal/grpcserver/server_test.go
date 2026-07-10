package grpcserver

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/Li-PengSheng/Distri-Inference-Sidecar/gen"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/batcher"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/tokenizer"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/vramguard"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

var initTokenizerOnce sync.Once

func initTestTokenizer() {
	initTokenizerOnce.Do(func() {
		tokenizer.Init(strings.Repeat("hello world foo bar ", 50))
	})
}

type testVRAMReader struct {
	used  float64
	total float64
}

func (r *testVRAMReader) ReadUsageMB() (float64, float64, error) { return r.used, r.total, nil }
func (r *testVRAMReader) Close()                                 {}
func (r *testVRAMReader) Name() string                           { return "test" }

func startTestGRPCServer(t *testing.T, handler http.HandlerFunc) pb.InferenceServiceClient {
	t.Helper()
	initTestTokenizer()

	backend := httptest.NewServer(handler)
	t.Cleanup(backend.Close)

	m := metrics.NewForTest()
	vg := vramguard.NewWithReader(
		vramguard.Config{PollIntervalMs: 1000, OOMThresholdPct: 90, CloseThresholdPct: 85},
		m,
		&testVRAMReader{used: 100, total: 10000},
	)
	b := batcher.New(batcher.Config{
		MaxBatchSize:     1,
		MaxWaitMs:        0,
		BackendURL:       backend.URL,
		BackendTimeoutMs: 5000,
	}, vg, m)
	go b.Start()
	t.Cleanup(b.Stop)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := lis.Addr().String()
	_ = lis.Close()

	srv := New(addr, b, m)
	serveDone := make(chan error, 1)
	go func() {
		serveDone <- srv.Serve()
	}()
	t.Cleanup(func() {
		srv.GracefulStop()
		<-serveDone
	})

	conn, err := grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	return pb.NewInferenceServiceClient(conn)
}

func TestInfer_BackendHTTP500ReturnsUnavailable(t *testing.T) {
	client := startTestGRPCServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "backend failure", http.StatusInternalServerError)
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := client.Infer(ctx, &pb.InferRequest{
		RequestId: "grpc-500-test",
		InputData: []byte("hello"),
		ModelName: "test-model",
	})
	if err == nil {
		t.Fatal("expected gRPC error for backend HTTP 500")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %T: %v", err, err)
	}
	if st.Code() != codes.Unavailable {
		t.Fatalf("expected codes.Unavailable, got %v (%s)", st.Code(), st.Message())
	}
}

func TestInfer_BackendExecutionErrorUsesResponseField(t *testing.T) {
	client := startTestGRPCServer(t, func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Requests []struct {
				ID string `json:"id"`
			} `json:"requests"`
		}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		id := "unknown"
		if len(payload.Requests) > 0 {
			id = payload.Requests[0].ID
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]string{{
				"id":          id,
				"output_data": "",
				"error":       "ollama timeout",
			}},
		})
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.Infer(ctx, &pb.InferRequest{
		RequestId: "biz-err",
		InputData: []byte("hello"),
		ModelName: "test-model",
	})
	if err != nil {
		t.Fatalf("expected successful gRPC response with error field, got %v", err)
	}
	if resp.Error != "ollama timeout" {
		t.Fatalf("expected backend execution error in response field, got %q", resp.Error)
	}
}

func TestInfer_BackendInvalidJSONReturnsInternal(t *testing.T) {
	client := startTestGRPCServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not-json"))
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := client.Infer(ctx, &pb.InferRequest{
		RequestId: "bad-json",
		InputData: []byte("hello"),
		ModelName: "test-model",
	})
	if err == nil {
		t.Fatal("expected gRPC error for invalid backend JSON")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %T: %v", err, err)
	}
	if st.Code() != codes.Internal {
		t.Fatalf("expected codes.Internal, got %v (%s)", st.Code(), st.Message())
	}
}
