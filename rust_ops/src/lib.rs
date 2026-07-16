#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};
use std::slice;

// Global BPE tokenizer instance, published exactly once via bpe_train.
// OnceLock permits concurrent reads without a Mutex, which matters because
// multiple gRPC handlers can count tokens after startup. A later bpe_train call
// still performs local training but cannot replace the published tokenizer.
use std::sync::OnceLock;
mod bpe_token;
use bpe_token::BPETokenizer;
static TOKENIZER: OnceLock<BPETokenizer> = OnceLock::new();

/// catch_ffi_panic runs `f` and converts any panic into `fallback` instead of
/// letting it unwind across the FFI boundary. A panic escaping an `extern "C"`
/// function aborts the whole process (Rust >= 1.81), which would take down the
/// Go sidecar; with this guard the caller just sees the fallback value.
fn catch_ffi_panic<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("rust_ops: panic caught at FFI boundary");
            fallback
        }
    }
}

/// free_string releases a C string previously allocated by this library with
/// `CString::into_raw`. It is a reserved C-ABI ownership utility; the current
/// tokenizer exports do not return Rust-owned strings. Do not call it on a
/// pointer allocated outside this library.
///
/// # Safety
/// `s` must have been produced by `CString::into_raw` from within this library.
#[unsafe(no_mangle)]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// bpe_train trains a tokenizer on the null-terminated `text` using
/// `vocab_size`, then attempts to publish it as the process-global tokenizer.
///
/// The first successful call wins because the tokenizer is shared by all Go
/// callers and changing it at runtime would make admission limits inconsistent
/// between concurrent requests. Later calls still perform training but their
/// result is discarded by `OnceLock`; null input and caught panics have no
/// observable return value because this C ABI function returns `void`.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_train(text: *const c_char, vocab_size: usize) {
    if text.is_null() {
        return;
    }
    catch_ffi_panic((), || {
        let s = unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("");
        let mut tok = BPETokenizer::new(vocab_size);
        tok.train(s, vocab_size);
        TOKENIZER.set(tok).ok();
    });
}

/// bpe_encode_len returns the number of BPE token IDs produced by encoding the
/// null-terminated `input`. It returns -1 when the tokenizer is uninitialised,
/// input is null, or a panic is caught at the FFI boundary. Invalid UTF-8 input
/// is treated as an empty string by the C-string conversion.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_encode_len(input: *const c_char) -> i32 {
    if input.is_null() {
        return -1;
    }
    catch_ffi_panic(-1, || {
        let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
        match TOKENIZER.get() {
            Some(tok) => tok.encode(s).len() as i32,
            None => -1,
        }
    })
}

/// tokenize_len counts whitespace-delimited tokens in null-terminated `input`.
/// It is retained as a simple reference implementation and bpe_count_tokens
/// fallback. It returns -1 for null input or a caught FFI panic; invalid UTF-8
/// is treated as an empty string.
#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len(input: *const c_char) -> i32 {
    if input.is_null() {
        return -1;
    }
    catch_ffi_panic(-1, || {
        let s = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
        s.split_whitespace().count() as i32
    })
}

/// tokenize_len_batch counts whitespace tokens for a batch of C strings and
/// returns the total token count across all non-null inputs. This avoids
/// per-input FFI overhead for Python benchmarks. It returns -1 when `inputs`
/// is null or a panic is caught; null elements are skipped and invalid UTF-8
/// elements contribute zero tokens.
///
/// # Safety
/// `inputs` must point to an array of `len` valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len_batch(inputs: *const *const c_char, len: usize) -> i64 {
    if inputs.is_null() {
        return -1;
    }

    catch_ffi_panic(-1, || {
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
    })
}

/// bpe_count_tokens returns the BPE token count for `input` after
/// `bpe_train` has initialised the global tokenizer. Before initialisation it
/// deliberately falls back to whitespace splitting so callers can still obtain
/// a count; sidecar admission must call `bpe_train` first to ensure its limit
/// is based on BPE tokens. It returns -1 for null input or a panic caught at
/// the FFI boundary; invalid UTF-8 is treated as an empty string.
#[unsafe(no_mangle)]
pub extern "C" fn bpe_count_tokens(input: *const c_char) -> i32 {
    if input.is_null() {
        return -1;
    }
    catch_ffi_panic(-1, || {
        let s = unsafe { CStr::from_ptr(input) }
            .to_str()
            .unwrap_or("");
        match TOKENIZER.get() {
            Some(tok) => tok.encode(s).len() as i32,
            None => tokenize_len(input), // fall back to whitespace splitting
        }
    })
}

/// bpe_encode_len_batch returns the total number of BPE token IDs across a
/// batch of non-null inputs. Returns -1 if the tokenizer is uninitialised,
/// `inputs` is null, or a panic is caught at the FFI boundary. Null elements
/// are skipped and invalid UTF-8 elements contribute zero tokens.
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

    catch_ffi_panic(-1, || {
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
    })
}
