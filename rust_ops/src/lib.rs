// rust_ops/src/lib.rs
//
// Performance-critical helper functions exposed via a C ABI so they can be
// called from Go (via CGo) or any other C-compatible runtime.
//
// Build as a static library with `cargo build --release`; the resulting
// `librust_ops.a` can be linked into external projects.

/// Adds two 32-bit signed integers and returns their sum.
///
/// # Safety
///
/// This function is marked `#[unsafe(no_mangle)]` and uses `extern "C"` so
/// that its symbol is stable and callable from C/Go. The function itself
/// performs no unsafe operations.
#[unsafe(no_mangle)]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
