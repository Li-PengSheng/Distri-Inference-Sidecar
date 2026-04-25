#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;

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

/// tokenize_len_batch counts whitespace tokens for a batch of C strings and
/// returns the total token count across all inputs. This avoids per-input FFI
/// overhead for Python benchmarks.
///
/// # Safety
/// `inputs` must point to an array of `len` valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len_batch(inputs: *const *const c_char, len: usize) -> i64 {
    if inputs.is_null() {
        return -1;
    }

    let arr = unsafe { slice::from_raw_parts(inputs, len) };
    let mut total = 0_i64;
    for &ptr in arr {
        if ptr.is_null() {
            continue;
        }
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("");
        total += s.split_whitespace().count() as i64;
    }
    total
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

/// bpe_encode_len_batch returns the total number of BPE token IDs across a
/// batch of inputs. Returns -1 if the tokenizer is not initialized.
///
/// # Safety
/// `inputs` must point to an array of `len` valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_encode_len_batch(inputs: *const *const c_char, len: usize) -> i64 {
    let tok = match TOKENIZER.get() {
        Some(t) => t,
        None => return -1,
    };
    if inputs.is_null() {
        return -1;
    }

    let arr = unsafe { slice::from_raw_parts(inputs, len) };
    let mut total = 0_i64;
    for &ptr in arr {
        if ptr.is_null() {
            continue;
        }
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("");
        total += tok.encode(s).len() as i64;
    }
    total
}
