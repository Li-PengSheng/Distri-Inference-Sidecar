// Package grpcserver implements the gRPC InferenceService defined in
// proto/inference.proto.
//
// The server exposes two RPCs:
//   - Infer: accepts a single inference request, submits it to the Batcher,
//     and blocks until the result is available or the client context is done.
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

// Infer is called by gRPC clients.
// It submits the request to the batcher and blocks until result comes back.
func (s *Server) Infer(ctx context.Context, req *pb.InferRequest) (*pb.InferResponse, error) {
	input := string(req.InputData)
	if err := tokenizer.Validate(input); err != nil {
		if errors.Is(err, tokenizer.ErrTokenizerFailure) {
			slog.Error("tokenizer failure during admission", "id", req.RequestId, "err", err)
			return nil, status.Error(codes.Internal, err.Error())
		}
		s.metrics.RejectedRequests.Inc()
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

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
		slog.Warn("request context cancelled", "id", req.RequestId)
		return nil, statusFromContextErr(ctx.Err())
	}
}

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

func statusFromContextErr(err error) error {
	if errors.Is(err, context.Canceled) {
		return status.Error(codes.Canceled, "request cancelled by client")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return status.Error(codes.DeadlineExceeded, "request deadline exceeded")
	}
	return status.Error(codes.Canceled, err.Error())
}

// HealthCheck returns the current VRAM utilisation and whether the
// circuit-breaker is closed (healthy = true) or open (healthy = false).
func (s *Server) HealthCheck(ctx context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	usedMB, totalMB := s.batcher.GetGuard().GetUsage()
	return &pb.HealthResponse{
		Healthy:     !s.batcher.GetGuard().IsOpen(),
		VramUsedMb:  float32(usedMB),
		VramTotalMb: float32(totalMB),
	}, nil
}
