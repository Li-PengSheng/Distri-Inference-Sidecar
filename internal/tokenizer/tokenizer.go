// Package tokenizer provides a lightweight token-counting utility backed by a
// Rust static library via CGo.
//
// TokenizeLen calls the C-ABI function tokenize_len exported from rust_ops,
// which splits the input on whitespace and returns the word count as a fast
// approximation of the number of tokens in a prompt.
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

const MaxInputTokens = 512

// Init 在服务启动时调用一次
func Init(trainCorpus string) {
	cs := C.CString(trainCorpus)
	defer C.free(unsafe.Pointer(cs))
	C.bpe_train(cs, 500)
}

// CountTokens 返回 token 数量
func CountTokens(input string) int {
	cs := C.CString(input)
	defer C.free(unsafe.Pointer(cs))
	return int(C.bpe_count_tokens(cs))
}

// Validate 超过限制返回 error
func Validate(input string) error {
	n := CountTokens(input)
	if n > MaxInputTokens {
		return fmt.Errorf("input too long: %d tokens (max %d)", n, MaxInputTokens)
	}
	return nil
}
