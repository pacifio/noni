from typing import Optional

import numpy as np

from .base import Backend

_MATMUL_KERNEL_SRC = """
#define TS 16

__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    const int tile_row = get_group_id(0);
    const int tile_col = get_group_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = tile_row * TS + local_row;
    const int global_col = tile_col * TS + local_col;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;

    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        int a_col = t * TS + local_col;
        Asub[local_row][local_col] = (global_row < M && a_col < K)
            ? A[global_row * K + a_col] : 0.0f;

        int b_row = t * TS + local_row;
        Bsub[local_row][local_col] = (b_row < K && global_col < N)
            ? B[b_row * N + global_col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++)
            acc += Asub[local_row][k] * Bsub[k][local_col];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N)
        C[global_row * N + global_col] = acc;
}
"""

_ELEMENTWISE_KERNEL_SRC = """
__kernel void add(__global const float* a,
                  __global const float* b,
                  __global float* out,
                  const int N) {
    int gid = get_global_id(0);
    if (gid < N) out[gid] = a[gid] + b[gid];
}

__kernel void multiply(__global const float* a,
                       __global const float* b,
                       __global float* out,
                       const int N) {
    int gid = get_global_id(0);
    if (gid < N) out[gid] = a[gid] * b[gid];
}

__kernel void relu(__global const float* a,
                   __global float* out,
                   const int N) {
    int gid = get_global_id(0);
    if (gid < N) out[gid] = max(a[gid], 0.0f);
}

__kernel void exp_kernel(__global const float* a,
                         __global float* out,
                         const int N) {
    int gid = get_global_id(0);
    if (gid < N) out[gid] = exp(a[gid]);
}

__kernel void log_kernel(__global const float* a,
                         __global float* out,
                         const int N) {
    int gid = get_global_id(0);
    if (gid < N) out[gid] = log(a[gid]);
}

__kernel void softmax(__global const float* input,
                      __global float* output,
                      const int n_rows,
                      const int n_cols) {
    int row = get_global_id(0);
    if (row >= n_rows) return;

    const __global float* row_in = input + row * n_cols;
    __global float* row_out = output + row * n_cols;

    float row_max = row_in[0];
    for (int j = 1; j < n_cols; j++)
        row_max = max(row_max, row_in[j]);

    float sum = 0.0f;
    for (int j = 0; j < n_cols; j++) {
        row_out[j] = exp(row_in[j] - row_max);
        sum += row_out[j];
    }

    for (int j = 0; j < n_cols; j++)
        row_out[j] /= sum;
}
"""


class OpenCLBuffer:
    """Wrapper combining a pyopencl.Buffer with shape and dtype metadata."""

    __slots__ = ("buf", "shape", "dtype")

    def __init__(self, buf, shape: tuple, dtype):
        self.buf = buf
        self.shape = shape
        self.dtype = dtype


class OpenCLBackend(Backend):
    """
    OpenCL backend for any OpenCL-capable device.

    Lazily initialises the context and compiles kernels on first use.
    Kernel objects are cached to avoid repeated retrieval overhead.
    """

    def __init__(self):
        self._cl = None
        self._ctx = None
        self._queue = None
        self._device = None
        self._available: Optional[bool] = None
        self._kernels: dict = {}

    # ── lazy init ─────────────────────────────────────────────────────────────

    def _init(self) -> bool:
        try:
            import pyopencl as cl
            self._cl = cl

            # Select device: prefer GPU, fall back to any
            self._device = None
            for platform in cl.get_platforms():
                gpus = platform.get_devices(cl.device_type.GPU)
                if gpus:
                    self._device = gpus[0]
                    break
            if self._device is None:
                for platform in cl.get_platforms():
                    devs = platform.get_devices()
                    if devs:
                        self._device = devs[0]
                        break
            if self._device is None:
                return False

            self._ctx = cl.Context([self._device])
            self._queue = cl.CommandQueue(
                self._ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE,
            )

            # Compile programs and cache kernel objects
            matmul_prog = cl.Program(self._ctx, _MATMUL_KERNEL_SRC).build()
            elem_prog = cl.Program(self._ctx, _ELEMENTWISE_KERNEL_SRC).build()

            self._kernels = {
                "matmul": cl.Kernel(matmul_prog, "matmul"),
                "add": cl.Kernel(elem_prog, "add"),
                "multiply": cl.Kernel(elem_prog, "multiply"),
                "relu": cl.Kernel(elem_prog, "relu"),
                "exp_kernel": cl.Kernel(elem_prog, "exp_kernel"),
                "log_kernel": cl.Kernel(elem_prog, "log_kernel"),
                "softmax": cl.Kernel(elem_prog, "softmax"),
            }
            return True

        except Exception as e:
            print(f"[noni] OpenCL init failed: {e}")
            return False

    def _ensure_init(self) -> None:
        if self._available is None:
            self._available = self._init()

    @property
    def name(self) -> str:
        self._ensure_init()
        if self._device is not None:
            return f"opencl/{self._device.name.strip()}"
        return "opencl (not available — install pyopencl + OpenCL drivers)"

    def is_available(self) -> bool:
        self._ensure_init()
        return self._available  # type: ignore[return-value]

    # ── data transfer ─────────────────────────────────────────────────────────

    def to_device(self, arr: np.ndarray) -> OpenCLBuffer:
        cl = self._cl
        arr32 = np.ascontiguousarray(arr, dtype=np.float32)
        buf = cl.Buffer(  # type: ignore[union-attr]
            self._ctx, #type:ignore
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,  # type: ignore[union-attr]
            hostbuf=arr32,
        )
        return OpenCLBuffer(buf, arr32.shape, arr32.dtype)

    def from_device(self, dev_buf: OpenCLBuffer) -> np.ndarray:  # type: ignore[override]
        result = np.empty(dev_buf.shape, dtype=np.float32)
        self._cl.enqueue_copy(self._queue, result, dev_buf.buf)  # type: ignore[union-attr]
        self._queue.finish()  # type: ignore[union-attr]
        return result

    def _alloc(self, shape: tuple) -> OpenCLBuffer:
        cl = self._cl
        n_bytes = int(np.prod(shape)) * 4
        buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, size=n_bytes)  # type: ignore[union-attr]
        return OpenCLBuffer(buf, shape, np.float32)

    def _enqueue(self, kernel_name: str, global_size: tuple,
                 local_size: Optional[tuple], *args) -> None:
        """Set args, enqueue kernel, and wait for completion."""
        kernel = self._kernels[kernel_name]
        kernel.set_args(*args)
        event = self._cl.enqueue_nd_range_kernel(  # type: ignore[union-attr]
            self._queue, kernel, global_size, local_size, #type:ignore
        )
        event.wait()

    # ── ops ───────────────────────────────────────────────────────────────────

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.matmul(a, b)

        if a.ndim == 3:
            return np.stack([
                self.matmul(a[i], b[i] if b.ndim == 3 else b)
                for i in range(a.shape[0])
            ])

        M, K = a.shape
        _, N = b.shape

        a_buf = self.to_device(a)
        b_buf = self.to_device(b)
        c_buf = self._alloc((M, N))

        TS = 16
        global_size = (
            ((M + TS - 1) // TS) * TS,
            ((N + TS - 1) // TS) * TS,
        )

        self._enqueue(
            "matmul", global_size, (TS, TS),
            a_buf.buf, b_buf.buf, c_buf.buf,
            np.int32(M), np.int32(N), np.int32(K),
        )
        return self.from_device(c_buf)

    def _elementwise(self, kernel_name: str, *arrays: np.ndarray,
                     out_shape: Optional[tuple] = None) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("OpenCL not available")

        flat = [a.ravel() for a in arrays]
        N = flat[0].size
        shape = out_shape or arrays[0].shape

        dev = [self.to_device(a) for a in flat]
        out_buf = self._alloc((N,))

        self._enqueue(
            kernel_name, (N,), None,
            *[d.buf for d in dev], out_buf.buf, np.int32(N),
        )
        return self.from_device(out_buf).reshape(shape)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.add(a, b)
        a_bc, b_bc = np.broadcast_arrays(a, b)
        return self._elementwise("add", a_bc, b_bc, out_shape=a_bc.shape)

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.multiply(a, b)
        a_bc, b_bc = np.broadcast_arrays(a, b)
        return self._elementwise("multiply", a_bc, b_bc, out_shape=a_bc.shape)

    def sum(self, a: np.ndarray,
            axis: Optional[int] = None,
            keepdims: bool = False) -> np.ndarray:
        # Reductions fall back to numpy — not the training bottleneck.
        return np.sum(a, axis=axis, keepdims=keepdims)

    def exp(self, a: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.exp(a)
        return self._elementwise("exp_kernel", a)

    def log(self, a: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.log(a)
        return self._elementwise("log_kernel", a)

    def relu(self, a: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.maximum(0.0, a)
        return self._elementwise("relu", a)

    def softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        if not self.is_available() or axis != -1:
            shifted = a - a.max(axis=axis, keepdims=True)
            e = np.exp(shifted)
            return e / e.sum(axis=axis, keepdims=True)

        orig_shape = a.shape
        a2d = a.reshape(-1, a.shape[-1])
        n_rows, n_cols = a2d.shape

        in_buf = self.to_device(a2d)
        out_buf = self._alloc(a2d.shape)

        self._enqueue(
            "softmax", (n_rows,), None,
            in_buf.buf, out_buf.buf,
            np.int32(n_rows), np.int32(n_cols),
        )
        return self.from_device(out_buf).reshape(orig_shape)
