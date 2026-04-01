""" # benchmark/tokenizer_bench.py
import time
import ctypes

# Python 原生实现
def python_tokenize(text):
    return len(text.split())

# 调用 Rust FFI 版本
lib = ctypes.CDLL("../../rust_ops/target/release/librust_ops.so")
lib.tokenize_len.restype = ctypes.c_int
lib.tokenize_len.argtypes = [ctypes.c_char_p]

texts = ["hello world " * 1000] * 10000

# 对比耗时
start = time.perf_counter()
for t in texts:
    python_tokenize(t)
python_time = time.perf_counter() - start

start = time.perf_counter()
for t in texts:
    lib.tokenize_len(t.encode())
rust_time = time.perf_counter() - start

print(f"Python: {python_time:.3f}s")
print(f"Rust:   {rust_time:.3f}s")
print(f"Speedup: {python_time/rust_time:.1f}x") """



# benchmark/tokenizer_bench.py

import time
import ctypes

lib = ctypes.CDLL("../../rust_ops/target/release/librust_ops.so")

# 原有接口
lib.tokenize_len.restype = ctypes.c_int
lib.tokenize_len.argtypes = [ctypes.c_char_p]

# 新 BPE 接口
lib.bpe_train.restype = None
lib.bpe_train.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
lib.bpe_encode_len.restype = ctypes.c_int
lib.bpe_encode_len.argtypes = [ctypes.c_char_p]

TRAIN_TEXT = "hello world foo bar " * 500  # 缩小语料
lib.bpe_train(TRAIN_TEXT.encode(), 50)     # vocab_size 改成 50

# BPE encode 也用小一点的文本
texts_bpe = ["hello world foo bar " * 10] * 10000  # 缩短文本

# Python baseline
def python_tokenize(text):
    return len(text.split())

start = time.perf_counter()
for t in texts_bpe:
    python_tokenize(t)
python_time = time.perf_counter() - start

# Rust whitespace
start = time.perf_counter()
for t in texts_bpe:
    lib.tokenize_len(t.encode())
rust_ws_time = time.perf_counter() - start

# Rust BPE
start = time.perf_counter()
for t in texts_bpe:
    lib.bpe_encode_len(t.encode())
rust_bpe_time = time.perf_counter() - start

print(f"Python whitespace:  {python_time:.3f}s")
print(f"Rust whitespace:    {rust_ws_time:.3f}s  ({python_time/rust_ws_time:.1f}x)")
print(f"Rust BPE encode:    {rust_bpe_time:.3f}s  ({python_time/rust_bpe_time:.1f}x)")