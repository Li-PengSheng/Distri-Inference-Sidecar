use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[unsafe(no_mangle)]
pub extern "C" fn tokenize_len(input: *const c_char) -> i32 {
    if input.is_null() { return 0; }
    let s = unsafe { CStr::from_ptr(input) }
        .to_str()
        .unwrap_or("");
    s.split_whitespace().count() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}