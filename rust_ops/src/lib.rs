use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// tokenize_len counts the whitespace-separated words in `input` as a fast
/// approximation of the token count. Returns 0 if `input` is null or invalid
/// UTF-8.
///
/// # Safety
/// `input` must be a valid, null-terminated C string for the duration of the
/// call. The caller retains ownership; this function does not free `input`.
#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len(input: *const c_char) -> i32 {
    if input.is_null() { return 0; }
    let s = unsafe { CStr::from_ptr(input) }
        .to_str()
        .unwrap_or("");
    s.split_whitespace().count() as i32
}

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