import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from noni import Tensor
from noni.backends.base import (
    BackendNotAvailableError,
    get_backend,
    list_available_backends,
)
from noni.backends.numpy_backend import NumpyBackend

DEVICES = ("cpu", "opencl", "mlx")


def allclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    return np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=rtol, atol=atol)


def opencl_available() -> bool:
    try:
        return get_backend("opencl").is_available()
    except BackendNotAvailableError:
        return False


def skip_if_no_opencl(fn):
    def wrapper():
        if not opencl_available():
            print(f"\n[{fn.__name__}]  SKIPPED (OpenCL not available)")
            return
        fn()
    wrapper.__name__ = fn.__name__
    return wrapper


def mlx_available() -> bool:
    try:
        return get_backend("mlx").is_available()
    except BackendNotAvailableError:
        return False


def skip_if_no_mlx(fn):
    def wrapper():
        if not mlx_available():
            print(f"\n[{fn.__name__}]  SKIPPED (MLX not available)")
            return
        fn()
    wrapper.__name__ = fn.__name__
    return wrapper


def test_registry():
    print("\n[test_registry]")
    available = list_available_backends()

    for device in DEVICES:
        assert device in available, f"Device '{device}' missing from registry"
        name, is_avail = available[device]
        status = "✓ available" if is_avail else "○ not available"
        print(f"  {device:10s} → {name:45s}  {status}")

    assert available["cpu"][1] is True, "CPU backend must always be available"
    print("  ✓ CPU backend is always available")

    try:
        get_backend("nonexistent_device_xyz")
        assert False, "Should have raised BackendNotAvailableError"
    except BackendNotAvailableError as e:
        assert "nonexistent_device_xyz" in str(e)
    print("  ✓ Unknown device raises BackendNotAvailableError")


def test_numpy_backend():
    print("\n[test_numpy_backend]")
    b = NumpyBackend()

    # Matmul
    A = np.random.randn(4, 6).astype(np.float32)
    B = np.random.randn(6, 5).astype(np.float32)
    assert allclose(b.matmul(A, B), A @ B)
    print("  ✓ matmul")

    # Batched matmul
    A3 = np.random.randn(3, 4, 6).astype(np.float32)
    B3 = np.random.randn(3, 6, 5).astype(np.float32)
    assert allclose(b.matmul(A3, B3), np.matmul(A3, B3))
    print("  ✓ batched matmul")

    # Elementwise
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    assert allclose(b.add(x, y), x + y)
    assert allclose(b.multiply(x, y), x * y)
    print("  ✓ add, multiply")

    # Reductions
    assert allclose(b.sum(x), np.sum(x)) #type:ignore
    assert allclose(b.sum(x, axis=0), np.sum(x, axis=0))
    assert allclose(b.sum(x, axis=1, keepdims=True), np.sum(x, axis=1, keepdims=True))
    print("  ✓ sum")

    # Activations
    xp = np.abs(x) + 0.01
    assert allclose(b.exp(x), np.exp(x))
    assert allclose(b.log(xp), np.log(xp))
    assert allclose(b.relu(x), np.maximum(0, x))
    print("  ✓ exp, log, relu")

    # Softmax
    x_large = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    s = b.softmax(x_large, axis=-1)
    assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(s))
    print("  ✓ softmax")

    # LayerNorm
    x_ln = np.random.randn(2, 8).astype(np.float32)
    out_ln = b.layer_norm(x_ln, np.ones(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
    assert np.allclose(out_ln.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out_ln.std(axis=-1), 1.0, atol=1e-3)
    print("  ✓ layer_norm")

    # GELU
    x_gelu = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    g = b.gelu(x_gelu)
    assert np.isclose(g[2], 0.0, atol=1e-6)
    assert g[4] > 0 and g[0] < 0
    print("  ✓ gelu")


def test_tensor_to():
    print("\n[test_tensor_to]")

    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, device="cpu")
    assert x.device == "cpu"

    x_oc = x.to("opencl")
    assert x_oc.device == "opencl"
    assert allclose(x_oc.data, x.data)
    assert x_oc.requires_grad == x.requires_grad
    print("  ✓ .to('opencl') updates device tag and preserves data")

    x.grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    x_oc2 = x.to("opencl")
    assert x_oc2.grad is not None and allclose(x_oc2.grad, x.grad)
    print("  ✓ .to() preserves grad")

    x_back = x_oc.cpu()
    assert x_back.device == "cpu"
    print("  ✓ .cpu() shorthand works")

    # Device propagates through matmul
    a = Tensor(np.random.randn(4, 6).astype(np.float32), device="cpu")
    b = Tensor(np.random.randn(6, 5).astype(np.float32))
    c = a @ b
    assert c.device == "cpu"
    print("  ✓ Device propagates through matmul")


def test_module_to():
    print("\n[test_module_to]")
    from noni.nn import Linear, TransformerBlock

    lin = Linear(8, 4)
    assert lin.weight.device == "cpu" and lin.bias.device == "cpu" #type:ignore

    lin.to("opencl")
    assert lin.weight.device == "opencl" and lin.bias.device == "opencl" #type:ignore
    print("  ✓ Linear.to('opencl') updates weight and bias device")

    lin.to("cpu")
    assert lin.weight.device == "cpu"
    print("  ✓ Module.to('cpu') moves back")

    block = TransformerBlock(d_model=16, n_heads=2, d_ff=32, dropout=0.0)
    block.to("opencl")
    params = block.parameters()
    for p in params:
        assert p.device == "opencl", f"Parameter on '{p.device}', expected 'opencl'"
    block.to("cpu")
    print(f"  ✓ TransformerBlock.to('opencl') updated {len(params)} parameters")


@skip_if_no_opencl
def test_opencl_correctness():
    print("\n[test_opencl_correctness]")
    b_gpu = get_backend("opencl")
    b_cpu = NumpyBackend()
    rng = np.random.default_rng(42)

    A = rng.standard_normal((64, 128)).astype(np.float32)
    B = rng.standard_normal((128, 64)).astype(np.float32)
    assert allclose(b_gpu.matmul(A, B), b_cpu.matmul(A, B), rtol=1e-3, atol=1e-4)
    print("  ✓ matmul matches CPU")

    x = rng.standard_normal((32, 64)).astype(np.float32)
    assert allclose(b_gpu.softmax(x, axis=-1), b_cpu.softmax(x, axis=-1))
    print("  ✓ softmax matches CPU")

    assert allclose(b_gpu.relu(x), b_cpu.relu(x))
    print("  ✓ relu matches CPU")


@skip_if_no_mlx
def test_mlx_correctness():
    print("\n[test_mlx_correctness]")
    b_mlx = get_backend("mlx")
    b_cpu = NumpyBackend()
    rng = np.random.default_rng(42)

    # matmul
    A = rng.standard_normal((64, 128)).astype(np.float32)
    B = rng.standard_normal((128, 64)).astype(np.float32)
    assert allclose(b_mlx.matmul(A, B), b_cpu.matmul(A, B), rtol=1e-3, atol=1e-4)
    print("  ✓ matmul matches CPU")

    x = rng.standard_normal((32, 64)).astype(np.float32)
    y = rng.standard_normal((32, 64)).astype(np.float32)

    # elementwise
    assert allclose(b_mlx.add(x, y), b_cpu.add(x, y))
    print("  ✓ add matches CPU")

    assert allclose(b_mlx.multiply(x, y), b_cpu.multiply(x, y))
    print("  ✓ multiply matches CPU")

    # reductions
    assert allclose(b_mlx.sum(x), b_cpu.sum(x)) #type:ignore
    assert allclose(b_mlx.sum(x, axis=0), b_cpu.sum(x, axis=0))
    assert allclose(b_mlx.sum(x, axis=1, keepdims=True), b_cpu.sum(x, axis=1, keepdims=True))
    print("  ✓ sum matches CPU")

    # activations
    xp = np.abs(x) + 0.01
    assert allclose(b_mlx.exp(x), b_cpu.exp(x))
    assert allclose(b_mlx.log(xp), b_cpu.log(xp))
    assert allclose(b_mlx.relu(x), b_cpu.relu(x))
    print("  ✓ exp, log, relu match CPU")

    # softmax — check sums to 1 and is finite
    assert allclose(b_mlx.softmax(x, axis=-1), b_cpu.softmax(x, axis=-1), rtol=1e-4, atol=1e-5)
    s = b_mlx.softmax(x, axis=-1)
    assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(s))
    print("  ✓ softmax matches CPU")

    # softmax numerical stability with large values
    x_large = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    s_large = b_mlx.softmax(x_large, axis=-1)
    assert np.allclose(s_large.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(s_large))
    print("  ✓ softmax numerically stable for large inputs")

    # layer_norm
    x_ln = rng.standard_normal((4, 16)).astype(np.float32)
    w = np.ones(16, dtype=np.float32)
    b_ln = np.zeros(16, dtype=np.float32)
    out_mlx = b_mlx.layer_norm(x_ln, w, b_ln)
    out_cpu = b_cpu.layer_norm(x_ln, w, b_ln)
    assert allclose(out_mlx, out_cpu, rtol=1e-3, atol=1e-4)
    assert np.allclose(out_mlx.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(out_mlx.std(axis=-1), 1.0, atol=1e-3)
    print("  ✓ layer_norm matches CPU")

    # gelu
    x_gelu = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    g_mlx = b_mlx.gelu(x_gelu)
    g_cpu = b_cpu.gelu(x_gelu)
    assert allclose(g_mlx, g_cpu, rtol=1e-4, atol=1e-5)
    assert np.isclose(g_mlx[2], 0.0, atol=1e-5)   # gelu(0) == 0
    assert g_mlx[4] > 0 and g_mlx[0] < 0
    print("  ✓ gelu matches CPU")


def test_device_forward_pass():
    print("\n[test_device_forward_pass]")
    from noni.nn import Linear

    x = Tensor(np.random.randn(4, 8).astype(np.float32), device="cpu")
    lin = Linear(8, 4)
    lin.to("cpu")

    y = lin(x)
    loss = y.sum()
    loss.backward()

    assert lin.weight.grad is not None and lin.weight.grad.shape == lin.weight.data.shape
    assert lin.bias.grad is not None #type:ignore
    print("  ✓ Forward + backward through device='cpu' Linear works")

    # Verify gradients match a reference path
    lin_ref = Linear(8, 4)
    lin_ref.weight.data = lin.weight.data.copy()
    lin_ref.bias.data = lin.bias.data.copy() #type:ignore

    x_ref = Tensor(x.data.copy(), device="cpu")
    loss_ref = lin_ref(x_ref).sum()
    loss_ref.backward()

    assert allclose(lin.weight.grad, lin_ref.weight.grad) #type:ignore
    assert allclose(lin.bias.grad, lin_ref.bias.grad) #type:ignore
    print("  ✓ Gradients identical between device-dispatched and reference paths")


def benchmark_matmul():
    print("\n[benchmark_matmul]  (1024 × 1024 × 1024 = 2.1 GFLOP)")

    SIZE = 1024
    REPS = 5
    FLOPS = 2 * SIZE**3

    A = np.random.randn(SIZE, SIZE).astype(np.float32)
    B = np.random.randn(SIZE, SIZE).astype(np.float32)

    available = list_available_backends()
    for device in DEVICES:
        if device not in available or not available[device][1]:
            print(f"  {device:10s} skipped (not available)")
            continue

        b = get_backend(device)
        _ = b.matmul(A, B)  # warmup

        start = time.perf_counter()
        for _ in range(REPS):
            _ = b.matmul(A, B)
        elapsed = (time.perf_counter() - start) / REPS

        gflops = FLOPS / elapsed / 1e9
        name = available[device][0]
        print(f"  {device:10s}  {elapsed*1000:6.1f} ms   {gflops:5.1f} GFLOPS   ({name})")


if __name__ == "__main__":
    tests = [
        test_registry,
        test_numpy_backend,
        test_tensor_to,
        test_module_to,
        test_device_forward_pass,
        test_opencl_correctness,
        test_mlx_correctness,
        benchmark_matmul,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {t.__name__} → {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR:  {t.__name__} → {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Backend tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
