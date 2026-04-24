import time
import ctypes

lib = ctypes.CDLL("../../rust_ops/target/release/librust_ops.so")
try:
    import rust_ops as rust_ops_py
except ImportError:
    rust_ops_py = None

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

TRAIN_TEXT = "hello world foo bar " * 500  # 缩小语料
lib.bpe_train(TRAIN_TEXT.encode(), 50)     # vocab_size 改成 50
if rust_ops_py is not None:
    rust_ops_py.py_bpe_train(TRAIN_TEXT, 50)

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
python_time = time.perf_counter() - start

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
else:
    rust_ws_batch_time = None

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
else:
    rust_bpe_batch_time = None

pyo3_ws_time = None
pyo3_ws_batch_time = None
pyo3_bpe_time = None
pyo3_bpe_batch_time = None

if rust_ops_py is not None:
    start = time.perf_counter()
    for t in texts_bpe:
        rust_ops_py.py_tokenize_len(t)
    pyo3_ws_time = time.perf_counter() - start

    start = time.perf_counter()
    rust_ops_py.py_tokenize_len_batch(texts_bpe)
    pyo3_ws_batch_time = time.perf_counter() - start

    start = time.perf_counter()
    for t in texts_bpe:
        rust_ops_py.py_bpe_encode_len(t)
    pyo3_bpe_time = time.perf_counter() - start

    start = time.perf_counter()
    rust_ops_py.py_bpe_encode_len_batch(texts_bpe)
    pyo3_bpe_batch_time = time.perf_counter() - start

print(f"Python whitespace:  {python_time:.3f}s")
print(f"Rust whitespace:    {rust_ws_time:.3f}s  ({python_time/rust_ws_time:.1f}x)")
print(f"Rust BPE encode:    {rust_bpe_time:.3f}s  ({python_time/rust_bpe_time:.1f}x)")
if rust_ws_batch_time is not None:
    print(f"Rust ws batch:      {rust_ws_batch_time:.3f}s  ({python_time/rust_ws_batch_time:.1f}x)")
else:
    print("Rust ws batch:      N/A (symbol not exported)")

if rust_bpe_batch_time is not None:
    print(f"Rust BPE batch:     {rust_bpe_batch_time:.3f}s  ({python_time/rust_bpe_batch_time:.1f}x)")
else:
    print("Rust BPE batch:     N/A (symbol not exported)")

if pyo3_ws_time is not None:
    print(f"PyO3 whitespace:    {pyo3_ws_time:.3f}s  ({python_time/pyo3_ws_time:.1f}x)")
    print(f"PyO3 ws batch:      {pyo3_ws_batch_time:.3f}s  ({python_time/pyo3_ws_batch_time:.1f}x)")
    print(f"PyO3 BPE encode:    {pyo3_bpe_time:.3f}s  ({python_time/pyo3_bpe_time:.1f}x)")
    print(f"PyO3 BPE batch:     {pyo3_bpe_batch_time:.3f}s  ({python_time/pyo3_bpe_batch_time:.1f}x)")
else:
    print("PyO3:               N/A (module not installed)")
    print("Install hint:       cd ../../rust_ops && maturin develop --release --features python")