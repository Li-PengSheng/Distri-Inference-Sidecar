import time
import ctypes
from pathlib import Path

LIB_PATH = (
    Path(__file__).resolve().parents[2]
    / "rust_ops"
    / "target"
    / "release"
    / "librust_ops.so"
)
lib = ctypes.CDLL(str(LIB_PATH))

print(f"Loaded Rust library: {LIB_PATH}")

# 原有接口
lib.tokenize_len.restype = ctypes.c_int
lib.tokenize_len.argtypes = [ctypes.c_char_p]

# 新 BPE 接口
lib.bpe_train.restype = None
lib.bpe_train.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
lib.bpe_encode_len.restype = ctypes.c_int
lib.bpe_encode_len.argtypes = [ctypes.c_char_p]

tokenize_len_batch = getattr(lib, "tokenize_len_batch", None)
if tokenize_len_batch is not None:
    tokenize_len_batch.restype = ctypes.c_longlong
    tokenize_len_batch.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]

bpe_encode_len_batch = getattr(lib, "bpe_encode_len_batch", None)
if bpe_encode_len_batch is not None:
    bpe_encode_len_batch.restype = ctypes.c_longlong
    bpe_encode_len_batch.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]

print(
    "CTypes batch symbols:",
    f"tokenize_len_batch={'yes' if tokenize_len_batch is not None else 'no'}",
    f"bpe_encode_len_batch={'yes' if bpe_encode_len_batch is not None else 'no'}",
)

TRAIN_TEXT = "hello world foo bar " * 500  # 缩小语料
lib.bpe_train(TRAIN_TEXT.encode(), 50)     # vocab_size 改成 50

# BPE encode 也用小一点的文本
texts_bpe = ["hello world foo bar " * 30] * 10000  # 缩短文本
encoded_texts = [t.encode("utf-8") for t in texts_bpe]
encoded_array = (ctypes.c_char_p * len(encoded_texts))(*encoded_texts)

# Python baseline
def python_tokenize(text):
    return len(text.split())

start = time.perf_counter()
for t in texts_bpe:
    python_tokenize(t)
python_ws_time = time.perf_counter() - start

# Rust whitespace
start = time.perf_counter()
for t in encoded_texts:
    lib.tokenize_len(t)
rust_ws_time = time.perf_counter() - start

# Rust whitespace batch (single FFI call)
if tokenize_len_batch is not None:
    start = time.perf_counter()
    tokenize_len_batch(encoded_array, len(encoded_texts))
    rust_ws_batch_time = time.perf_counter() - start
    rust_ws_batch_mode = "ffi batch"
else:
    # Fallback keeps benchmark comparable even if batch symbol is unavailable.
    start = time.perf_counter()
    for t in encoded_texts:
        lib.tokenize_len(t)
    rust_ws_batch_time = time.perf_counter() - start
    rust_ws_batch_mode = "fallback loop"

# Rust BPE
start = time.perf_counter()
for t in encoded_texts:
    lib.bpe_encode_len(t)
rust_bpe_time = time.perf_counter() - start

# Rust BPE batch (single FFI call)
if bpe_encode_len_batch is not None:
    start = time.perf_counter()
    bpe_encode_len_batch(encoded_array, len(encoded_texts))
    rust_bpe_batch_time = time.perf_counter() - start
    rust_bpe_batch_mode = "ffi batch"
else:
    # Fallback keeps benchmark comparable even if batch symbol is unavailable.
    start = time.perf_counter()
    for t in encoded_texts:
        lib.bpe_encode_len(t)
    rust_bpe_batch_time = time.perf_counter() - start
    rust_bpe_batch_mode = "fallback loop"

print(f"Python whitespace:  {python_ws_time:.3f}s")
print(f"Rust whitespace:    {rust_ws_time:.3f}s  ({python_ws_time/rust_ws_time:.1f}x vs Python whitespace)")
print(f"Rust ws batch:      {rust_ws_batch_time:.3f}s  ({python_ws_time/rust_ws_batch_time:.1f}x vs Python whitespace)  [{rust_ws_batch_mode}]")
print()
print(f"Rust BPE encode:    {rust_bpe_time:.3f}s")
print(f"Rust BPE batch:     {rust_bpe_batch_time:.3f}s  ({rust_bpe_time/rust_bpe_batch_time:.2f}x vs Rust BPE single)  [{rust_bpe_batch_mode}]")
print()
print("Benchmark note:")
print("- Python baseline here is whitespace splitting only.")
print("- Rust BPE numbers are not directly comparable to Python whitespace timing.")
print("- For apples-to-apples BPE comparison, use an equivalent Python BPE tokenizer implementation.")