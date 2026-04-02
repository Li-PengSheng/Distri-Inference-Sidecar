// Package tokenizer provides a CGo bridge to the Rust BPE tokenizer library
// (rust_ops). It exposes token counting and input-length validation used by the
// gRPC server to reject prompts that exceed the configured token limit.
package tokenizer

/*
#cgo LDFLAGS: -L${SRCDIR}/../../rust_ops/target/release -lrust_ops
#include <stdlib.h>
extern int bpe_count_tokens(const char* input);
extern void bpe_train(const char* text, size_t vocab_size);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// MaxInputTokens is the maximum number of BPE tokens accepted per inference
// request. Requests exceeding this limit are rejected before batching.
const MaxInputTokens = 512

// Init trains the BPE tokenizer on the provided corpus and must be called once
// at process startup before any call to CountTokens or Validate.
func Init(trainCorpus string) {
	cs := C.CString(trainCorpus)
	defer C.free(unsafe.Pointer(cs))
	C.bpe_train(cs, 500)
}

// CountTokens returns the number of BPE tokens in input as determined by the
// Rust tokenizer. If the tokenizer has not been initialized it falls back to
// whitespace splitting.
func CountTokens(input string) int {
	cs := C.CString(input)
	defer C.free(unsafe.Pointer(cs))
	return int(C.bpe_count_tokens(cs))
}

// Validate returns an error when the token count of input exceeds MaxInputTokens.
func Validate(input string) error {
	n := CountTokens(input)
	if n > MaxInputTokens {
		return fmt.Errorf("input too long: %d tokens (max %d)", n, MaxInputTokens)
	}
	return nil
}
