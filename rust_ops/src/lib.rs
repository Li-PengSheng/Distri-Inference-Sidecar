use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// Global BPE tokenizer instance, initialized exactly once via bpe_train.
// OnceLock ensures thread-safe lazy initialization without a Mutex.
use std::sync::OnceLock;
mod bpe_token;
use bpe_token::BPETokenizer;
static TOKENIZER: OnceLock<BPETokenizer> = OnceLock::new();

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

/// bpe_train trains the global BPE tokenizer on `text` with the given
/// `vocab_size`. It must be called once before any call to bpe_count_tokens or
/// bpe_encode_len. Subsequent calls are silently ignored.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_train(text: *const c_char, vocab_size: usize) {
    let s = unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("");
    let mut tok = BPETokenizer::new(vocab_size);
    tok.train(s, vocab_size);
    TOKENIZER.set(tok).ok();
}

/// bpe_encode_len returns the number of BPE token IDs produced by encoding
/// `input`. Returns -1 if the tokenizer has not been initialized.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_encode_len(input: *const c_char) -> i32 {
    let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
    match TOKENIZER.get() {
        Some(tok) => tok.encode(s).len() as i32,
        None => -1,
    }
}

/// tokenize_len counts tokens by splitting on whitespace. Kept as a reference
/// implementation and fallback for comparison against the BPE tokenizer.
#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len(input: *const c_char) -> i32 {
    let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
    s.split_whitespace().count() as i32
}

/// bpe_count_tokens returns the BPE token count for `input`. Falls back to
/// whitespace splitting if the tokenizer has not been initialised via bpe_train.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_count_tokens(input: *const c_char) -> i32 {
    let s = unsafe { CStr::from_ptr(input) }
        .to_str()
        .unwrap_or("");
    match TOKENIZER.get() {
        Some(tok) => tok.encode(s).len() as i32,
        None => tokenize_len(input), // fall back to whitespace splitting
    }
}