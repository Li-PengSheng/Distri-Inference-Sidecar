package vramguard

import (
	"testing"

	"github.com/Li-PengSheng/Distri-Inference-Sidecar/internal/metrics"
)

// fakeReader returns predetermined used/total (or err) without touching GPUs.
type fakeReader struct {
	used  float64
	total float64
	err   error
}

func (f *fakeReader) ReadUsageMB() (float64, float64, error) {
	return f.used, f.total, f.err
}

func (f *fakeReader) Close() {}

func (f *fakeReader) Name() string { return "fake" }

// testGuardConfig returns a small-interval hysteretic config for unit tests.
func testGuardConfig() Config {
	return Config{
		PollIntervalMs:    100,
		OOMThresholdPct:   90,
		CloseThresholdPct: 85,
	}
}

func TestGuard_CircuitOpensAboveThreshold(t *testing.T) {
	m := metrics.NewForTest()
	reader := &fakeReader{used: 950, total: 1000}
	g := NewWithReader(testGuardConfig(), m, reader)

	g.pollOnce()
	if !g.IsOpen() {
		t.Fatal("expected circuit to open when usage exceeds threshold")
	}
}

func TestGuard_CircuitClosesBelowCloseThreshold(t *testing.T) {
	m := metrics.NewForTest()
	reader := &fakeReader{used: 950, total: 1000}
	g := NewWithReader(testGuardConfig(), m, reader)
	g.pollOnce()
	if !g.IsOpen() {
		t.Fatal("expected circuit open after high usage poll")
	}

	reader.used = 800
	g.pollOnce()
	if g.IsOpen() {
		t.Fatal("expected circuit to close when usage drops below close threshold")
	}
}

func TestGuard_HysteresisKeepsCircuitOpenInDeadband(t *testing.T) {
	m := metrics.NewForTest()
	reader := &fakeReader{used: 950, total: 1000}
	g := NewWithReader(testGuardConfig(), m, reader)
	g.pollOnce()
	if !g.IsOpen() {
		t.Fatal("expected circuit open after high usage poll")
	}

	reader.used = 880
	g.pollOnce()
	if !g.IsOpen() {
		t.Fatal("expected circuit to remain open inside hysteresis deadband")
	}

	reader.used = 840
	g.pollOnce()
	if g.IsOpen() {
		t.Fatal("expected circuit to close after dropping below close threshold")
	}
}

func TestGuard_StaysClosedBelowThreshold(t *testing.T) {
	m := metrics.NewForTest()
	reader := &fakeReader{used: 500, total: 1000}
	g := NewWithReader(testGuardConfig(), m, reader)

	g.pollOnce()
	if g.IsOpen() {
		t.Fatal("expected circuit to remain closed below threshold")
	}
}

func TestGuard_PollErrorDoesNotChangeCircuit(t *testing.T) {
	m := metrics.NewForTest()
	reader := &fakeReader{err: errTestVRAM}
	g := NewWithReader(testGuardConfig(), m, reader)

	g.pollOnce()
	if g.IsOpen() {
		t.Fatal("expected circuit to remain closed on poll error")
	}
}

var errTestVRAM = &testVRAMError{}

type testVRAMError struct{}

func (e *testVRAMError) Error() string { return "test vram read failure" }

func TestQueryVRAMViaSMI_ParseErrors(t *testing.T) {
	t.Run("invalid used value", func(t *testing.T) {
		used, total, err := parseSMIOutput("not-a-number, 1024")
		if err == nil {
			t.Fatalf("expected parse error, got used=%v total=%v", used, total)
		}
	})

	t.Run("invalid total value", func(t *testing.T) {
		used, total, err := parseSMIOutput("512, not-a-number")
		if err == nil {
			t.Fatalf("expected parse error, got used=%v total=%v", used, total)
		}
	})

	t.Run("malformed output", func(t *testing.T) {
		_, _, err := parseSMIOutput("only-one-field")
		if err == nil {
			t.Fatal("expected error for malformed output")
		}
	})
}
