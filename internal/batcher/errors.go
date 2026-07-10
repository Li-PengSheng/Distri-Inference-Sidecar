package batcher

import "errors"

var (
	// ErrCircuitOpen is returned when the VRAM guard circuit breaker is open.
	ErrCircuitOpen = errors.New("vram guard: circuit open, try again later")
	// ErrQueueFull is returned when the internal request queue is full.
	ErrQueueFull = errors.New("batch queue full, try again later")
	// ErrBatcherStopped is returned when Submit is called after Stop.
	ErrBatcherStopped = errors.New("batcher stopped")
	// ErrBackendUnavailable indicates the HTTP backend could not be reached or
	// returned a non-success HTTP status code.
	ErrBackendUnavailable = errors.New("backend unavailable")
	// ErrBackendResponseInvalid indicates the backend response could not be
	// encoded or decoded by the sidecar.
	ErrBackendResponseInvalid = errors.New("backend response invalid")
)
