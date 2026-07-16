// Package tokenizer provides a CGo bridge to the Rust BPE tokenizer library
// (rust_ops). It exposes token counting and input-length validation used by the
// gRPC server to reject prompts that exceed the configured token limit.
package tokenizer

/*
#cgo LDFLAGS: -L${SRCDIR}/../../rust_ops/target/release -lrust_ops
#include <stdlib.h>
extern int bpe_count_tokens(const char* input);
extern int bpe_encode_len(const char* input);
extern void bpe_train(const char* text, size_t vocab_size);
*/
import "C"
import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// MaxInputTokens is the maximum number of BPE tokens accepted per inference
// request. Requests exceeding this limit are rejected before batching.
const MaxInputTokens = 512

// ErrTokenizerFailure indicates the Rust tokenizer could not produce a token
// count (uninitialised vocabulary or a panic caught at the FFI boundary).
// Callers should treat it as an internal error, not an input-validation error.
var ErrTokenizerFailure = errors.New("tokenizer failure")

// Init trains the BPE tokenizer on the provided corpus and must be called once
// at process startup before any call to CountTokens or Validate. It returns an
// error if the corpus is empty or the Rust tokenizer failed to initialise.
func Init(trainCorpus string) error {
	if strings.TrimSpace(trainCorpus) == "" {
		return fmt.Errorf("tokenizer: training corpus is empty")
	}

	cs := C.CString(trainCorpus)
	defer C.free(unsafe.Pointer(cs))
	C.bpe_train(cs, 500)

	// bpe_train reports nothing over the FFI boundary, so probe the trained
	// tokenizer: bpe_encode_len returns -1 when no vocabulary is loaded.
	probe := C.CString("tokenizer init self-check")
	defer C.free(unsafe.Pointer(probe))
	if int(C.bpe_encode_len(probe)) < 0 {
		return fmt.Errorf("tokenizer: BPE training did not initialise a vocabulary")
	}
	return nil
}

// CountTokens returns the number of BPE tokens in input as determined by the
// Rust tokenizer (bpe_count_tokens). Init must have succeeded first; a
// negative return value signals a tokenizer failure (uninitialised vocabulary
// or an FFI-side error) and should be treated as ErrTokenizerFailure by
// callers such as Validate.
func CountTokens(input string) int {
	cs := C.CString(input)
	defer C.free(unsafe.Pointer(cs))
	return int(C.bpe_count_tokens(cs))
}

// Validate returns an error when the token count of input exceeds
// MaxInputTokens, or an ErrTokenizerFailure when no count could be produced.
func Validate(input string) error {
	n := CountTokens(input)
	if n < 0 {
		return fmt.Errorf("%w: token count unavailable", ErrTokenizerFailure)
	}
	if n > MaxInputTokens {
		return fmt.Errorf("input too long: %d tokens (max %d)", n, MaxInputTokens)
	}
	return nil
}
