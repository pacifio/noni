from typing import Optional, Tuple

import numpy as np

from .base import Backend

# OpenCL C source for the matmul kernel with local memory tiling.
# Stored as a string because it's a different language (OpenCL C) compiled
# at runtime by pyopencl.clBuildProgram().

_MATMUL_KERNEL_SRC = """
/*
 * Tiled matrix multiply: C = A @ B
 *
 * Grid:       (ceil(M/TS), ceil(N/TS))  work-groups
 * Work-group: (TS, TS) work-items  (TS = TILE_SIZE)
 *
 * Each work-group cooperatively loads:
 *   A-tile: TS × TS elements into local memory Asub
 *   B-tile: TS × TS elements into local memory Bsub
 * Then computes the partial dot product for its output tile.
 *
 * Local memory used: 2 × TS × TS × 4 bytes
 *   TS=16: 2×16×16×4 = 2 KB  (far below the 32-64 KB limit)
 *   TS=32: 2×32×32×4 = 8 KB
 *
 * Memory access pattern:
 *   Each work-item reads A[row, k] once per K-loop iteration.
 *   With local memory, each element is read once from global memory
 *   and TS times from (fast) local memory.
 *   Speedup: TS-fold reduction in global memory reads.
 */
#define TS 16

__kernel void matmul(
    __global const float* A,   /* [M, K] row-major */
    __global const float* B,   /* [K, N] row-major */
    __global float* C,         /* [M, N] row-major, output */
    const int M,
    const int N,
    const int K
) {
    /* Work-group tile coordinates */
    const int tile_row = get_group_id(0);
    const int tile_col = get_group_id(1);

    /* Local work-item coordinates within this tile */
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    /* Global output element coordinates */
    const int global_row = tile_row * TS + local_row;
    const int global_col = tile_col * TS + local_col;

    /* Shared tiles — declared in local address space (SRAM) */
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;

    /* Iterate over tiles in the K dimension */
    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        /* Collaborative load: each work-item loads one element of each tile */
        int a_row = global_row;
        int a_col = t * TS + local_col;
        Asub[local_row][local_col] = (a_row < M && a_col < K)
            ? A[a_row * K + a_col] : 0.0f;

        int b_row = t * TS + local_row;
        int b_col = global_col;
        Bsub[local_row][local_col] = (b_row < K && b_col < N)
            ? B[b_row * N + b_col] : 0.0f;

        /* Synchronise: ensure all work-items have finished loading */
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Compute partial dot product for this tile */
        for (int k = 0; k < TS; k++) {
            acc += Asub[local_row][k] * Bsub[k][local_col];
        }

        /* Synchronise: ensure all work-items done before loading next tile */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Write result, guarded by boundary check */
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}
"""

_ELEMENTWISE_KERNEL_SRC = """
/*
 * Element-wise operations.  Simple embarrassingly parallel kernels.
 * Each work-item handles one element.
 * For small arrays, the kernel launch overhead (~5-10 µs on OpenCL)
 * dominates. Consider batching small ops together.
 */

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

/* Row-wise softmax (numerically stable).
 * Each work-item handles one row.
 * Requires n_cols ≤ max work-group size (usually 256-1024).
 * For larger n_cols use a reduction-based approach. */
__kernel void softmax(__global const float* input,
                      __global float* output,
                      const int n_rows,
                      const int n_cols) {
    int row = get_global_id(0);
    if (row >= n_rows) return;

    const __global float* row_in = input + row * n_cols;
    __global float* row_out = output + row * n_cols;

    /* Find row max for numerical stability */
    float row_max = row_in[0];
    for (int j = 1; j < n_cols; j++)
        row_max = max(row_max, row_in[j]);

    /* Compute exp(x - max) and sum */
    float sum = 0.0f;
    for (int j = 0; j < n_cols; j++) {
        row_out[j] = exp(row_in[j] - row_max);
        sum += row_out[j];
    }

    /* Normalise */
    for (int j = 0; j < n_cols; j++)
        row_out[j] /= sum;
}
"""


class OpenCLBuffer:
    """
    Wrapper combining a pyopencl.Buffer with its shape and dtype metadata.

    pyopencl.Buffer is just a raw byte buffer — it carries no shape info.
    We wrap it with the metadata needed to reconstruct a numpy array on
    device-to-host transfer.
    """
    def __init__(self, buf, shape: tuple, dtype):
        self.buf = buf          # the actual cl.Buffer
        self.shape = shape      # numpy shape tuple
        self.dtype = dtype      # numpy dtype


class OpenCLBackend(Backend):
    """
    OpenCL backend for any OpenCL-capable device.

    Lazily initialises the OpenCL context and compiles the kernel programs
    on first use.  Subsequent calls reuse the compiled kernels (no JIT cost).

    Device selection
    ----------------
    By default picks the first GPU found, then falls back to CPU.
    To explicitly choose a device:

        import pyopencl as cl
        platforms = cl.get_platforms()
        devices   = platforms[0].get_devices(cl.device_type.GPU)
        # Pass device to OpenCLBackend constructor
    """

    def __init__(self):
        self._cl = None
        self._ctx = None
        self._queue = None
        self._matmul_prog = None
        self._elem_prog   = None
        self._available   = None

    def _init(self):
        """
        Initialise OpenCL context, device, and compile kernel programs.
        Called on first use (lazy initialisation pattern).
        """
        try:
            import pyopencl as cl
            self._cl = cl

            # Select device: prefer GPU, fall back to any device
            platforms = cl.get_platforms()
            if not platforms:
                return False

            self._device = None
            for platform in platforms:
                # Try GPU first
                gpus = platform.get_devices(cl.device_type.GPU)
                if gpus:
                    self._device = gpus[0]
                    break
            if self._device is None:
                for platform in platforms:
                    devices = platform.get_devices()
                    if devices:
                        self._device = devices[0]
                        break
            if self._device is None:
                return False

            # Create context and command queue
            self._ctx   = cl.Context([self._device])
            # PROFILING_ENABLE lets us measure kernel execution time
            self._queue = cl.CommandQueue(
                self._ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE
            )

            # Compile kernel programs (happens once, result is cached on device)
            # This is analogous to nvcc compilation in CUDA or PSO in Metal.
            self._matmul_prog = cl.Program(self._ctx, _MATMUL_KERNEL_SRC).build()
            self._elem_prog   = cl.Program(self._ctx, _ELEMENTWISE_KERNEL_SRC).build()
            return True

        except Exception as e:
            print(f"[noni] OpenCL init failed: {e}")
            return False

    @property
    def name(self) -> str:
        dev = getattr(self, '_device', None)
        if dev is not None:
            return f"opencl/{dev.name.strip()}"
        return "opencl (not available — install pyopencl + OpenCL drivers)"

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._init()
        return self._available

    # ── data transfer ─────────────────────────────────────────────────────────

    def to_device(self, arr: np.ndarray) -> OpenCLBuffer:
        """
        Transfer numpy array → OpenCL device buffer.

        Uses CL_MEM_COPY_HOST_PTR which initiates a DMA transfer to device
        DRAM.  For repeated small transfers, consider using pinned (page-locked)
        memory (CL_MEM_ALLOC_HOST_PTR) to avoid the extra copy.
        """
        cl = self._cl
        arr32 = np.ascontiguousarray(arr, dtype=np.float32)
        buf = cl.Buffer(self._ctx, #type:ignore
                        cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, #type:ignore
                        hostbuf=arr32)
        return OpenCLBuffer(buf, arr32.shape, arr32.dtype)

    def from_device(self, dev_buf: OpenCLBuffer) -> np.ndarray: #type:ignore
        """
        Transfer OpenCL device buffer → host numpy array.

        cl.enqueue_copy with wait=True blocks until the DMA completes.
        For async transfers, use an event and cl.wait_for_events().
        """
        cl = self._cl
        result = np.empty(dev_buf.shape, dtype=np.float32)
        cl.enqueue_copy(self._queue, result, dev_buf.buf) #type:ignore
        self._queue.finish() #type:ignore
        return result

    def _alloc(self, shape: tuple) -> OpenCLBuffer:
        """Allocate an uninitialised device buffer."""
        cl = self._cl
        n_bytes = int(np.prod(shape)) * 4  # float32 = 4 bytes
        buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, size=n_bytes) #type:ignore
        return OpenCLBuffer(buf, shape, np.float32)

    # ── core ops ──────────────────────────────────────────────────────────────

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Tiled GEMM using the OpenCL kernel compiled from _MATMUL_KERNEL_SRC.

        Work-group size: (TS, TS) = (16, 16)
        Global size:     (ceil(M/TS)*TS, ceil(N/TS)*TS) — rounded up to tile multiple
        """
        if not self.is_available():
            return np.matmul(a, b)

        if a.ndim == 3:
            return np.stack([
                self.matmul(a[i], b[i] if b.ndim == 3 else b)
                for i in range(a.shape[0])
            ])

        M, K = a.shape
        K2, N = b.shape
        assert K == K2

        a_buf = self.to_device(a)
        b_buf = self.to_device(b)
        c_buf = self._alloc((M, N))

        TS = 16
        # Global work size must be a multiple of local work size
        global_size = (
            ((M + TS - 1) // TS) * TS,
            ((N + TS - 1) // TS) * TS,
        )
        local_size = (TS, TS)

        kernel = self._matmul_prog.matmul #type:ignore
        kernel.set_args(
            a_buf.buf, b_buf.buf, c_buf.buf,
            np.int32(M), np.int32(N), np.int32(K)
        )
        event = self._cl.enqueue_nd_range_kernel( #type:ignore
            self._queue, kernel, global_size, local_size #type:ignore
        )
        event.wait()
        return self.from_device(c_buf)

    def _elementwise(self, kernel_name: str, *arrays: np.ndarray,
                     out_shape: Optional[tuple] = None) -> np.ndarray:
        """Helper: launch an elementwise kernel over flattened arrays."""
        if not self.is_available():
            raise RuntimeError("OpenCL not available")

        flat_arrays = [a.ravel() for a in arrays]
        N = flat_arrays[0].size
        shape = out_shape or arrays[0].shape

        dev_arrays = [self.to_device(a) for a in flat_arrays]
        out_buf    = self._alloc((N,))

        kernel = getattr(self._elem_prog, kernel_name)
        kernel.set_args(*[d.buf for d in dev_arrays], out_buf.buf, np.int32(N))
        event = self._cl.enqueue_nd_range_kernel( #type:ignore
            self._queue, kernel, (N,), None #type:ignore
        )
        event.wait()
        return self.from_device(out_buf).reshape(shape)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.add(a, b)
        # Handle broadcasting by expanding to common shape first
        a_bc, b_bc = np.broadcast_arrays(a, b)
        out_shape   = a_bc.shape
        return self._elementwise("add", a_bc, b_bc, out_shape=out_shape)

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.is_available():
            return np.multiply(a, b)
        a_bc, b_bc = np.broadcast_arrays(a, b)
        return self._elementwise("multiply", a_bc, b_bc, out_shape=a_bc.shape)

    def sum(self, a: np.ndarray,
            axis: Optional[int] = None,
            keepdims: bool = False) -> np.ndarray:
        # Reductions are complex to implement in OpenCL (tree reduction pattern).
        # For educational clarity, we use numpy here.
        # A production implementation would use a parallel prefix-sum kernel.
        if not self.is_available():
            return np.sum(a, axis=axis, keepdims=keepdims)
        # Transfer to device then back (for the matmul-heavy training case,
        # reductions are not the bottleneck)
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
        """
        Row-wise softmax using the OpenCL kernel.

        Currently only handles axis=-1 (last dimension).
        For other axes, we fall back to numpy.
        """
        if not self.is_available() or axis != -1:
            shifted = a - a.max(axis=axis, keepdims=True)
            e = np.exp(shifted)
            return e / e.sum(axis=axis, keepdims=True)

        # Reshape to 2D: (n_rows, n_cols)
        orig_shape = a.shape
        a2d = a.reshape(-1, a.shape[-1])
        n_rows, n_cols = a2d.shape

        in_buf  = self.to_device(a2d)
        out_buf = self._alloc(a2d.shape)

        kernel = self._elem_prog.softmax #type:ignore
        kernel.set_args(
            in_buf.buf, out_buf.buf,
            np.int32(n_rows), np.int32(n_cols)
        )
        event = self._cl.enqueue_nd_range_kernel( #type:ignore
            self._queue, kernel, (n_rows,), None #type:ignore
        )
        event.wait()
        return self.from_device(out_buf).reshape(orig_shape)
