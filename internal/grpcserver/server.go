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
	"fmt"
	"log/slog"
	"net"

	pb "github.com/Li-PengSheng/Distri-Inference-Sidecar/gen"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/batcher"
	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
	"google.golang.org/grpc"
)

// Server is the gRPC service implementation. It embeds the generated
// UnimplementedInferenceServiceServer so that it satisfies the interface even
// if new RPCs are added to the proto in the future.
type Server struct {
	pb.UnimplementedInferenceServiceServer
	addr    string
	batcher *batcher.Batcher
	metrics *metrics.Metrics
}

// New creates a Server that will listen on addr and delegate inference work to
// the provided Batcher, recording outcomes with the given Metrics.
func New(addr string, b *batcher.Batcher, m *metrics.Metrics) *Server {
	return &Server{
		addr:    addr,
		batcher: b,
		metrics: m,
	}
}

// Serve starts the gRPC server and blocks until it terminates. It registers the
// InferenceService and begins accepting connections on the configured address.
func (s *Server) Serve() error {
	lis, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.addr, err)
	}

	grpcSrv := grpc.NewServer()
	pb.RegisterInferenceServiceServer(grpcSrv, s)

	slog.Info("gRPC server listening", "addr", s.addr)
	return grpcSrv.Serve(lis)
}

// Infer is called by gRPC clients.
// It submits the request to the batcher and blocks until result comes back.
func (s *Server) Infer(ctx context.Context, req *pb.InferRequest) (*pb.InferResponse, error) {
	resultCh := make(chan batcher.Result, 1)

	bReq := &batcher.Request{
		ID:        req.RequestId,
		InputData: req.InputData,
		ModelName: req.ModelName,
		ResultCh:  resultCh,
	}

	// Submit to batcher — may fail if VRAM circuit is open
	if err := s.batcher.Submit(bReq); err != nil {
		slog.Warn("request rejected", "id", req.RequestId, "err", err)
		return &pb.InferResponse{
			RequestId: req.RequestId,
			Error:     err.Error(),
		}, nil
	}

	// Block until batcher fans the result back
	select {
	case result := <-resultCh:
		if result.Err != nil {
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
		// Client cancelled or timed out
		slog.Warn("request context cancelled", "id", req.RequestId)
		return &pb.InferResponse{
			RequestId: req.RequestId,
			Error:     "request cancelled by client",
		}, nil
	}
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
