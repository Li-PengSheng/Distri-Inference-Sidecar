// Package tokenizer provides a lightweight token-counting utility backed by a
// Rust static library via CGo.
//
// TokenizeLen calls the C-ABI function tokenize_len exported from rust_ops,
// which splits the input on whitespace and returns the word count as a fast
// approximation of the number of tokens in a prompt.
package tokenizer

/*
#cgo LDFLAGS: -L${SRCDIR}/../../rust_ops/target/release -lrust_ops
#cgo CFLAGS: -I${SRCDIR}/../../rust_ops
#include <stdlib.h>
#include "tokenizer.h"
*/
import "C"
import "unsafe"

func TokenizeLen(text string) int {
	cs := C.CString(text)
	defer C.free(unsafe.Pointer(cs))
	return int(C.tokenize_len(cs))
}
