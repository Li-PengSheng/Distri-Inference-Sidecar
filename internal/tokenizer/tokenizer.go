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
