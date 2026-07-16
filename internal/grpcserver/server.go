// Package grpcserver implements the gRPC InferenceService defined in
// proto/inference.proto.
//
// The server exposes two RPCs:
//   - Infer: admits the prompt via the BPE tokenizer, submits it to the
//     Batcher, and blocks until the result is available or the client
//     context is done.
//   - HealthCheck: returns current VRAM usage and circuit-breaker state.
package grpcserver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"

	pb "github.com/Li-PengSheng/Distri-Inference-Sidecar/gen"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/batcher"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/tokenizer"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server is the gRPC service implementation. It embeds the generated
// UnimplementedInferenceServiceServer so that it satisfies the interface even
// if new RPCs are added to the proto in the future.
type Server struct {
	pb.UnimplementedInferenceServiceServer
	addr    string
	batcher *batcher.Batcher
	metrics *metrics.Metrics
	grpcSrv *grpc.Server
}

// New creates a Server that will listen on addr and delegate inference work to
// the provided Batcher, recording outcomes with the given Metrics.
func New(addr string, b *batcher.Batcher, m *metrics.Metrics) *Server {
	return &Server{
		addr:    addr,
		batcher: b,
		metrics: m,
		grpcSrv: grpc.NewServer(grpc.ChainUnaryInterceptor(recoveryInterceptor)),
	}
}

// recoveryInterceptor converts handler panics into codes.Internal instead of
// letting them crash the whole sidecar process (grpc-go does not recover
// panics itself).
func recoveryInterceptor(
	ctx context.Context,
	req any,
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (resp any, err error) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("panic in gRPC handler", "method", info.FullMethod, "panic", r)
			resp = nil
			err = status.Error(codes.Internal, "internal server error")
		}
	}()
	return handler(ctx, req)
}

// Serve starts the gRPC server and blocks until it terminates. It registers the
// InferenceService and begins accepting connections on the configured address.
func (s *Server) Serve() error {
	lis, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.addr, err)
	}

	pb.RegisterInferenceServiceServer(s.grpcSrv, s)

	slog.Info("gRPC server listening", "addr", s.addr)
	return s.grpcSrv.Serve(lis)
}

// GracefulStop stops accepting new RPCs and waits for in-flight handlers to finish.
func (s *Server) GracefulStop() {
	if s.grpcSrv != nil {
		s.grpcSrv.GracefulStop()
	}
}

// Infer admits the prompt via the BPE tokenizer, submits it to the batcher,
// and blocks until a Result arrives or the client context is cancelled.
// Over-limit prompts return InvalidArgument; tokenizer FFI failures return
// Internal. Batcher rejection and backend errors are mapped by
// statusFromSubmitErr / statusFromResultErr.
func (s *Server) Infer(ctx context.Context, req *pb.InferRequest) (*pb.InferResponse, error) {
	// Admit before enqueue so over-long prompts never consume batcher capacity
	// or hit the Python/Ollama path. Tokenizer failures are Internal (sidecar
	// fault); length violations are InvalidArgument (client fault).
	input := string(req.InputData)
	if err := tokenizer.Validate(input); err != nil {
		if errors.Is(err, tokenizer.ErrTokenizerFailure) {
			slog.Error("tokenizer failure during admission", "id", req.RequestId, "err", err)
			return nil, status.Error(codes.Internal, err.Error())
		}
		s.metrics.RejectedRequests.Inc()
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	// Buffer of 1 so flushBatch never blocks if the client already left via
	// ctx.Done(); the Result is still delivered once and then dropped.
	resultCh := make(chan batcher.Result, 1)

	bReq := &batcher.Request{
		ID:        req.RequestId,
		InputData: req.InputData,
		ModelName: req.ModelName,
		Ctx:       ctx,
		ResultCh:  resultCh,
	}

	if err := s.batcher.Submit(bReq); err != nil {
		slog.Warn("request rejected", "id", req.RequestId, "err", err)
		return nil, statusFromSubmitErr(err)
	}

	select {
	case result := <-resultCh:
		if result.Err != nil {
			// Infrastructure failures become RPC status codes so clients can
			// retry/back off. Opaque model/backend error strings stay in the
			// response body (RPC succeeds) to preserve per-request detail.
			if st, ok := statusFromResultErr(result.Err); ok {
				return nil, st
			}
			return &pb.InferResponse{
				RequestId: req.RequestId,
				Error:     result.Err.Error(),
			}, nil
		}
		return &pb.InferResponse{
			RequestId:  req.RequestId,
			OutputData: result.OutputData,
			LatencyMs:  result.LatencyMs,
		}, nil

	case <-ctx.Done():
		// Request may still be in the queue or an in-flight batch; flushBatch
		// drops cancelled items before the HTTP call. We return immediately
		// and do not wait for ResultCh.
		slog.Warn("request context cancelled", "id", req.RequestId)
		return nil, statusFromContextErr(ctx.Err())
	}
}

// statusFromSubmitErr maps batcher.Submit failures to gRPC status codes:
// ResourceExhausted for circuit-open / queue-full (client should back off),
// Unavailable when stopped (process is draining; retry after reconnect).
func statusFromSubmitErr(err error) error {
	switch {
	case errors.Is(err, batcher.ErrCircuitOpen), errors.Is(err, batcher.ErrQueueFull):
		return status.Error(codes.ResourceExhausted, err.Error())
	case errors.Is(err, batcher.ErrBatcherStopped):
		return status.Error(codes.Unavailable, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}

// statusFromResultErr maps known batch Result errors to gRPC statuses.
// The bool is false for opaque backend/per-request errors that should be
// returned in InferResponse.Error instead of as an RPC status — those are
// usually model-side failures that are not retryable as transport errors.
func statusFromResultErr(err error) (error, bool) {
	switch {
	case errors.Is(err, batcher.ErrBackendUnavailable):
		return status.Error(codes.Unavailable, err.Error()), true
	case errors.Is(err, batcher.ErrBackendResponseInvalid):
		return status.Error(codes.Internal, err.Error()), true
	case errors.Is(err, context.Canceled):
		return status.Error(codes.Canceled, err.Error()), true
	case errors.Is(err, context.DeadlineExceeded):
		return status.Error(codes.DeadlineExceeded, err.Error()), true
	default:
		return nil, false
	}
}

// statusFromContextErr maps a cancelled or deadline-exceeded client context
// to the corresponding gRPC status while Infer is waiting on ResultCh.
func statusFromContextErr(err error) error {
	if errors.Is(err, context.Canceled) {
		return status.Error(codes.Canceled, "request cancelled by client")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return status.Error(codes.DeadlineExceeded, "request deadline exceeded")
	}
	return status.Error(codes.Canceled, err.Error())
}

// HealthCheck returns the latest VRAM reading and the VRAM circuit-breaker
// state. Healthy is true exactly when the breaker is closed, meaning this
// check sees no VRAM-pressure rejection. It does not verify that the batcher,
// Python backend, Ollama, or VRAM reader itself is otherwise ready.
func (s *Server) HealthCheck(ctx context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	usedMB, totalMB := s.batcher.GetGuard().GetUsage()
	return &pb.HealthResponse{
		Healthy:     !s.batcher.GetGuard().IsOpen(),
		VramUsedMb:  float32(usedMB),
		VramTotalMb: float32(totalMB),
	}, nil
}
