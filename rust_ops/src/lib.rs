use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// 全局 tokenizer 实例（用 Once 保证线程安全初始化）
use std::sync::OnceLock;
mod bpe_token;
use bpe_token::BPETokenizer;
static TOKENIZER: OnceLock<BPETokenizer> = OnceLock::new();

// #[unsafe(no_mangle)]
// pub extern "C" fn tokenize_len(input: *const c_char) -> i32 {
//     if input.is_null() {
//         return 0;
//     }
//     let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
//     s.split_whitespace().count() as i32
// }

/// free_string releases a C string that was previously allocated by Rust with
/// `CString::into_raw`. Do not call this on strings allocated outside Rust.
///
/// # Safety
/// `s` must have been produced by `CString::into_raw` from within this library.
#[unsafe(no_mangle)]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

// 训练接口
#[unsafe(no_mangle)]
pub extern "C" fn bpe_train(text: *const c_char, vocab_size: usize) {
    let s = unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("");
    let mut tok = BPETokenizer::new(vocab_size);
    tok.train(s, vocab_size);
    TOKENIZER.set(tok).ok();
}

// 编码接口：返回 token 数量
#[unsafe(no_mangle)]
pub extern "C" fn bpe_encode_len(input: *const c_char) -> i32 {
    let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
    match TOKENIZER.get() {
        Some(tok) => tok.encode(s).len() as i32,
        None => -1,
    }
}

// 保留原来的 whitespace tokenizer 作对比
#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len(input: *const c_char) -> i32 {
    let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
    s.split_whitespace().count() as i32
}

// 返回 token 数量，供 Go 层判断
#[unsafe(no_mangle)]
pub extern "C" fn bpe_count_tokens(input: *const c_char) -> i32 {
    let s = unsafe { CStr::from_ptr(input) }
        .to_str()
        .unwrap_or("");
    match TOKENIZER.get() {
        Some(tok) => tok.encode(s).len() as i32,
        None => tokenize_len(input), // 降级到 whitespace
    }
}