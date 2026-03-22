// rust_ops/src/lib.rs
#[no_mangle]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b // 先写最简单的，跑通 cgo 调用流程最重要
}
