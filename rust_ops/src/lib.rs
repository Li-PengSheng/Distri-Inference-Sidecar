// rust_ops/src/lib.rs

#[unsafe(no_mangle)] // 注意这里加了 unsafe(...)
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
