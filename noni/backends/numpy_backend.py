"""
noni/backends/numpy_backend.py
====================================
CPU backend using NumPy.  Always available.  Reference implementation.

This is the backend Noni was originally written with.  All other backends
are essentially GPU-accelerated versions of what's here.  When something
behaves unexpectedly on a GPU backend, comparing against NumPy output is the
first debugging step.

Performance characteristics
----------------------------
NumPy uses BLAS (OpenBLAS, MKL, or ATLAS) for matmul.  On a modern CPU with
AVX-512 and a good BLAS:
  - matmul(1024, 1024, 1024):  ~2 ms
  - matmul(4096, 4096, 4096):  ~2-5 s

Compare to CUDA GPU:
  - matmul(1024, 1024, 1024):  ~0.1 ms   (20× faster)
  - matmul(4096, 4096, 4096):  ~20 ms    (100× faster)

For small matrices (batch inference, single sequences), CPU is often
competitive because GPU kernel launch overhead (~10 µs) dominates.

Thread safety
-------------
NumPy releases the GIL for most array operations, so this backend is safe
for concurrent use from multiple Python threads.  The threading.local() flag
in no_grad() also ensures correctness across threads.
"""

from typing import Optional

import numpy as np

from .base import Backend


class NumpyBackend(Backend):
	@property
	def name(self) -> str:
		return "cpu"

	def is_available(self) -> bool:
		return True

	def to_device(self, arr: np.ndarray, dtype=np.float32) -> np.ndarray:
		return arr.astype(dtype) if arr.dtype != dtype else arr

	def from_device(self, dev_arr: object) -> np.ndarray:
		return np.ascontiguousarray(dev_arr, dtype=np.float32)

	def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
		return np.matmul(a, b)

	def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
		return np.add(a, b)

	def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
		return np.multiply(a, b)

	def sum(self, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
		return np.sum(a, axis=axis, keepdims=keepdims)

	def exp(self, a: np.ndarray) -> np.ndarray:
		return np.exp(a)

	def log(self, a: np.ndarray) -> np.ndarray:
		return np.log(a)

	def relu(self, a: np.ndarray) -> np.ndarray:
		return np.maximum(0.0, a)

	def softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
		shifted = a - a.max(axis=axis, keepdims=True)
		e: np.ndarray = np.exp(shifted)
		return e/e.sum(axis=axis, keepdims=True)
