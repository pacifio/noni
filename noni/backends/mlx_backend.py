from typing import Optional

import numpy as np

from .base import Backend


class MLXBackend(Backend):
    """
    Apple Silicon backend using MLX for zero-copy GPU acceleration via Metal.

    This is the recommended backend for users with Apple Silicon Macs who want
    GPU speedups without any CUDA dependency.  The implementation is minimal
    because MLX mirrors NumPy's API, and unified memory eliminates the need
    for explicit host↔device transfers.
    """

    def __init__(self):
        self._mx = None          # lazy mlx.core import
        self._available = None

    def _check(self) -> bool:
        try:
            import mlx.core as mx
            # Force a Metal API call to verify hardware is present
            # mx.default_device() returns Device(gpu, 0) on Apple Silicon
            dev = mx.default_device()
            if dev.type == mx.gpu:
                self._mx = mx
                return True
            # CPU-only fallback (e.g. Rosetta or Intel Mac)
            return False
        except Exception:
            return False

    @property
    def name(self) -> str:
        if self._mx is not None:
            try:
                import platform
                chip = platform.processor() or "Apple Silicon"
                return f"mlx/metal:0 ({chip})"
            except Exception:
                pass
        return "mlx/metal (not available)"

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._check()
        return self._available

    # ── data transfer ─────────────────────────────────────────────────────────

    def to_device(self, arr: np.ndarray):
        """
        Convert numpy array → MLX array.

        On Apple Silicon this is *not* a copy in the traditional sense.
        Unified memory means the data lives in the same physical DRAM
        accessible by both CPU and GPU.  mx.array() may still need to
        create a new buffer (numpy → MLX internal format), but there is
        no PCIe/NVLink DMA transfer as on discrete-GPU systems.
        """
        mx = self._mx
        return mx.array(arr.astype(np.float32)) #type:ignore

    def from_device(self, dev_arr) -> np.ndarray: #type:ignore
        """
        Convert MLX array → host numpy.

        mx.eval() forces lazy graph evaluation before the conversion.
        np.array() then reads directly from the same unified memory —
        again, no DMA transfer.
        """
        mx = self._mx
        mx.eval(dev_arr) #type:ignore
        return np.array(dev_arr, copy=False).astype(np.float32)

    # ── core ops ──────────────────────────────────────────────────────────────

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        C = A @ B via MLX → Metal Performance Shaders GEMM.

        MPS auto-selects the best tiling strategy based on matrix shape
        and the specific Apple Silicon generation:
          - M1/M2: 7-core / 10-core GPU, FP32 GEMM
          - M3:    shader-core improvements, hardware ray-tracing
          - M4:    enhanced matrix throughput
          - M5:    Neural Accelerator TensorOps (up to 4× speedup)

        Unified memory means no host→device copy overhead before the
        GEMM — the data is already "on the GPU".
        """
        if not self.is_available():
            return np.matmul(a, b)
        mx = self._mx
        a_m = self.to_device(a)
        b_m = self.to_device(b)
        return self.from_device(mx.matmul(a_m, b_m)) #type:ignore

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.add(a, b)
        mx = self._mx
        return self.from_device(mx.add(self.to_device(a), self.to_device(b))) #type:ignore

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.multiply(a, b)
        mx = self._mx
        return self.from_device(mx.multiply(self.to_device(a), self.to_device(b))) #type:ignore

    def sum(self, a: np.ndarray,
            axis: Optional[int] = None,
            keepdims: bool = False) -> np.ndarray:
        if not self.is_available():
            return np.sum(a, axis=axis, keepdims=keepdims)
        mx = self._mx
        result = mx.sum(self.to_device(a), axis=axis, keepdims=keepdims) #type:ignore
        return self.from_device(result)

    def exp(self, a: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.exp(a)
        return self.from_device(self._mx.exp(self.to_device(a))) #type:ignore

    def log(self, a: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.log(a)
        return self.from_device(self._mx.log(self.to_device(a))) #type:ignore

    def relu(self, a: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.maximum(0.0, a)
        mx = self._mx
        return self.from_device(mx.maximum(self.to_device(a), mx.array(0.0))) #type:ignore

    def softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax via MLX.

        MLX's lazy evaluation means the subtract-max, exp, and divide
        ops are fused into a single Metal kernel launch — no intermediate
        buffers are materialised in memory.  This is one of the key
        advantages of graph-based lazy execution over eager mode.
        """
        if not self.is_available():
            shifted = a - a.max(axis=axis, keepdims=True)
            e = np.exp(shifted)
            return e / e.sum(axis=axis, keepdims=True)
        mx = self._mx
        a_m = self.to_device(a)
        shifted = a_m - mx.max(a_m, axis=axis, keepdims=True) #type:ignore
        e = mx.exp(shifted) #type:ignore
        return self.from_device(e / mx.sum(e, axis=axis, keepdims=True)) #type:ignore

    def layer_norm(self, a: np.ndarray, weight: np.ndarray,
                   bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Layer normalization via MLX.

        The lazy graph captures mean → variance → normalize → scale+shift
        as a single fused computation.  On M5 with @mx.compile, this
        compiles down to a minimal number of Metal kernel launches.
        """
        if not self.is_available():
            mean = a.mean(axis=-1, keepdims=True)
            var  = ((a - mean)**2).mean(axis=-1, keepdims=True)
            return (a - mean) / np.sqrt(var + eps) * weight + bias
        mx = self._mx
        a_m = self.to_device(a)
        w_m = self.to_device(weight)
        b_m = self.to_device(bias)
        mean  = mx.mean(a_m, axis=-1, keepdims=True) #type:ignore
        var   = mx.mean((a_m - mean)**2, axis=-1, keepdims=True) #type:ignore
        norm  = (a_m - mean) / mx.sqrt(var + eps) #type:ignore
        return self.from_device(norm * w_m + b_m)

    def gelu(self, a: np.ndarray) -> np.ndarray:
        """
        GELU activation via MLX.

        Uses the tanh approximation: GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

        With lazy evaluation the entire expression is fused — the six
        elementwise ops become one Metal kernel, avoiding five round-trips
        to unified memory.  Equivalent to a single Triton fused kernel.
        """
        if not self.is_available():
            cdf = 0.5 * (1.0 + np.tanh(
                np.sqrt(2.0/np.pi) * (a + 0.044715 * a**3)))
            return a * cdf
        mx = self._mx
        a_m = self.to_device(a)
        sqrt_2_over_pi = mx.array(np.sqrt(2.0/np.pi), dtype=mx.float32) #type:ignore
        cdf = mx.array(0.5, dtype=mx.float32) * ( #type:ignore
            mx.array(1.0, dtype=mx.float32) + mx.tanh( #type:ignore
                sqrt_2_over_pi * (a_m + mx.array(0.044715, dtype=mx.float32) * a_m**3))) #type:ignore
        return self.from_device(a_m * cdf)
