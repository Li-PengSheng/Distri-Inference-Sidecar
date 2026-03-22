package main

/*
#cgo LDFLAGS: -L../../rust_ops/target/release -lrust_ops
#include <stdint.h>
int32_t add_numbers(int32_t a, int32_t b);
*/
import "C"
import (
	"fmt"
)

func main() {
	// 调用 Rust 函数
	res := C.add_numbers(10, 20)
	fmt.Printf("Hello from Go! Rust says 10 + 20 = %d\n", res)
}
