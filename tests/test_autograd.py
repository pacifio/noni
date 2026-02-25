import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import noni as ng
from noni.tensor import Tensor


def grad_check(fn, *inputs, eps=1e-3, atol=2e-2):
    """
    Check that analytical gradients match numerical gradients for all inputs.

    We use float64 arithmetic for the numerical pass to reduce cancellation
    error, then compare against the float32 analytical gradients.

    inputs: Tensors with requires_grad=True
    fn:     function that takes those Tensors and returns a scalar Tensor
    """
    # Save original data as float64
    orig_data = [x.data.copy().astype(np.float64) for x in inputs]

    # --- Analytical grads (float32 engine) ---
    outputs = fn(*inputs)
    outputs.backward()
    analytical = [x.grad.copy().astype(np.float64) if x.grad is not None else None
                  for x in inputs]

    # Reset
    for x in inputs:
        x.grad = None

    # --- Numerical grads (float64 perturbations) ---
    numerical = []
    for xi, x in enumerate(inputs):
        x.grad = None
        num_g = np.zeros_like(orig_data[xi])
        it = np.nditer(orig_data[xi], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index

            # +eps
            x.data = orig_data[xi].astype(np.float32)
            x.data[idx] = float(orig_data[xi][idx] + eps)
            for xx in inputs: xx.grad = None
            f_plus = float(fn(*inputs).data.sum())

            # -eps
            x.data = orig_data[xi].astype(np.float32)
            x.data[idx] = float(orig_data[xi][idx] - eps)
            for xx in inputs: xx.grad = None
            f_minus = float(fn(*inputs).data.sum())

            num_g[idx] = (f_plus - f_minus) / (2.0 * eps)
            it.iternext()

        # Restore original data
        x.data = orig_data[xi].astype(np.float32)
        x.grad = None
        numerical.append(num_g)

    # Restore all data
    for x, od in zip(inputs, orig_data):
        x.data = od.astype(np.float32)
        x.grad = None

    for i, (a, n) in enumerate(zip(analytical, numerical)):
        if a is None:
            continue
        abs_err = np.abs(a - n)
        rel_err = abs_err / (np.abs(n) + 1e-6)
        # Use absolute error for near-zero grads, relative elsewhere
        err = np.where(np.abs(n) < 1e-4, abs_err, rel_err)
        max_err = err.max()
        if max_err > atol:
            print(f"  ✗ Input {i}: max error = {max_err:.2e}")
            print(f"    analytical:\n{a}")
            print(f"    numerical:\n{n}")
            return False
        else:
            print(f"  ✓ Input {i}: max error = {max_err:.2e}")
    return True



def test_add():
    print("\n[test_add]")
    a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
    b = Tensor([[0.5, -1.], [2., 0.]], requires_grad=True)
    assert grad_check(lambda x, y: (x + y).sum(), a, b)

def test_sub():
    print("\n[test_sub]")
    a = Tensor([1., 2., 3.], requires_grad=True)
    b = Tensor([3., 1., 2.], requires_grad=True)
    assert grad_check(lambda x, y: (x - y).sum(), a, b)

def test_mul():
    print("\n[test_mul]")
    a = Tensor([[2., -1.], [0.5, 3.]], requires_grad=True)
    b = Tensor([[1.,  2.], [3., -1.]], requires_grad=True)
    assert grad_check(lambda x, y: (x * y).sum(), a, b)

def test_div():
    print("\n[test_div]")
    a = Tensor([2., 4., 6.], requires_grad=True)
    b = Tensor([1., 2., 3.], requires_grad=True)
    assert grad_check(lambda x, y: (x / y).sum(), a, b)

def test_pow():
    print("\n[test_pow]")
    a = Tensor([1., 2., 3.], requires_grad=True)
    assert grad_check(lambda x: (x ** 3).sum(), a)

def test_matmul():
    print("\n[test_matmul]")
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x, y: (x @ y).sum(), a, b)

def test_sum():
    print("\n[test_sum]")
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.sum(axis=1).sum(), a)

def test_mean():
    print("\n[test_mean]")
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.mean(), a)

def test_exp():
    print("\n[test_exp]")
    a = Tensor(np.random.randn(4).astype(np.float32) * 0.5, requires_grad=True)
    assert grad_check(lambda x: x.exp().sum(), a)

def test_log():
    print("\n[test_log]")
    a = Tensor(np.abs(np.random.randn(4).astype(np.float32)) + 0.1, requires_grad=True)
    assert grad_check(lambda x: x.log().sum(), a)

def test_relu():
    print("\n[test_relu]")
    a = Tensor(np.array([-1., 0.5, -0.3, 2.0]), requires_grad=True)
    assert grad_check(lambda x: x.relu().sum(), a, eps=1e-3)

def test_gelu():
    print("\n[test_gelu]")
    a = Tensor(np.random.randn(8).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.gelu().sum(), a)

def test_sigmoid():
    print("\n[test_sigmoid]")
    a = Tensor(np.random.randn(5).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.sigmoid().sum(), a)

def test_tanh():
    print("\n[test_tanh]")
    a = Tensor(np.random.randn(5).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.tanh().sum(), a)

def test_softmax():
    print("\n[test_softmax]")
    # softmax(x).sum() has ZERO gradient because softmax sums to 1.
    # Use a proper loss: multiply by a weight vector.
    np.random.seed(1)
    a = Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)
    w = Tensor(np.random.randn(3, 5).astype(np.float32))  # no grad needed
    # L = sum(softmax(a) * w)
    assert grad_check(lambda x: (x.softmax(axis=-1) * w).sum(), a)

def test_log_softmax():
    print("\n[test_log_softmax]")
    np.random.seed(2)
    a = Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)
    w = Tensor(np.random.randn(3, 5).astype(np.float32))
    assert grad_check(lambda x: (x.log_softmax(axis=-1) * w).sum(), a, atol=3.5e-2)

def test_broadcast_add():
    print("\n[test_broadcast_add]  (broadcasting)")
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x, y: (x + y).sum(), a, b)

def test_reshape():
    print("\n[test_reshape]")
    a = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.reshape(6).sum(), a)

def test_transpose():
    print("\n[test_transpose]")
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    assert grad_check(lambda x: x.transpose().sum(), a)

def test_indexing():
    print("\n[test_indexing]")
    a = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
    idx = np.array([0, 2, 4])
    assert grad_check(lambda x: x[idx].sum(), a)

def test_chain_rule():
    print("\n[test_chain_rule]  (deep chain: relu → matmul → sum)")
    W = Tensor(np.random.randn(4, 4).astype(np.float32) * 0.5, requires_grad=True)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    assert grad_check(lambda W, x: (x @ W).relu().sum(), W, x)

def test_layernorm():
    print("\n[test_layernorm]")
    from noni.nn import LayerNorm
    np.random.seed(42)
    ln = LayerNorm(8)
    ln.weight.data[:] = 1.0
    ln.bias.data[:]   = 0.0

    x = Tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    orig = x.data.copy()

    # Analytical
    out = ln(x).sum()
    out.backward()
    analytical = x.grad.copy() #type:ignore

    # Numerical
    x.grad = None
    ln.zero_grad()
    num_g = np.zeros_like(x.data, dtype=np.float64)
    eps = 1e-2
    for idx in np.ndindex(x.shape):
        xd = orig.copy()
        xd[idx] = orig[idx] + eps
        x.data = xd
        ln.zero_grad(); x.grad = None
        fp = float(ln(x).sum().data)

        xd[idx] = orig[idx] - eps
        x.data = xd
        ln.zero_grad(); x.grad = None
        fm = float(ln(x).sum().data)

        num_g[idx] = (fp - fm) / (2 * eps)

    x.data = orig

    abs_err = np.abs(analytical.astype(np.float64) - num_g)
    rel_err = abs_err / (np.abs(num_g) + 1e-5)
    err = np.where(np.abs(num_g) < 1e-3, abs_err, rel_err)
    max_err = err.max()
    print(f"  ✓ Input 0: max error = {max_err:.2e}" if max_err < 0.1
          else f"  ✗ Input 0: max error = {max_err:.2e}")
    assert max_err < 0.1, f"LayerNorm grad check failed: {max_err}"

def test_cross_entropy():
    print("\n[test_cross_entropy]  (logits)")
    from noni.nn import CrossEntropyLoss
    np.random.seed(42)
    loss_fn = CrossEntropyLoss()
    targets = np.array([1, 3, 7, 0])

    logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
    orig = logits.data.copy()

    # Analytical
    loss = loss_fn(logits, targets)
    loss.backward()
    analytical = logits.grad.copy() #type:ignore

    # Numerical
    num_g = np.zeros_like(logits.data, dtype=np.float64)
    eps = 1e-2
    for idx in np.ndindex(logits.shape):
        ld = orig.copy()
        ld[idx] = orig[idx] + eps
        logits.data = ld; logits.grad = None
        fp = float(loss_fn(logits, targets).data)

        ld[idx] = orig[idx] - eps
        logits.data = ld; logits.grad = None
        fm = float(loss_fn(logits, targets).data)

        num_g[idx] = (fp - fm) / (2 * eps)

    logits.data = orig

    abs_err = np.abs(analytical.astype(np.float64) - num_g)
    rel_err = abs_err / (np.abs(num_g) + 1e-5)
    err = np.where(np.abs(num_g) < 1e-4, abs_err, rel_err)
    max_err = err.max()
    print(f"  ✓ Input 0: max error = {max_err:.2e}" if max_err < 0.05
          else f"  ✗ Input 0: max error = {max_err:.2e}")
    assert max_err < 0.05


# Phase 1 tests: retain_graph & no_grad ─────────────────────────────────────

def test_retain_graph():
    """
    backward(retain_graph=True) leaves the graph intact for a second call.

    Key subtlety — intermediate gradients accumulate across calls:
      Round 1:  pow_node.grad = [1, 1]  →  x.grad  = [4,  6]
      Round 2:  pow_node.grad = [2, 2]  →  x.grad += [8, 12]  = [12, 18]
      Round 3:  pow_node.grad = [3, 3]  →  x.grad += [12,18]  = [24, 36]

    This matches PyTorch's behaviour.  To get "exactly one pass worth of grad"
    on subsequent calls you'd zero non-leaf grads manually — but for the
    typical use case (adversarial training, multi-task loss) accumulation is
    exactly what you want.
    """
    print("\n[test_retain_graph]")

    # Core: three calls, each accumulates with the growing non-leaf grad ─
    # L = x0² + x1²
    # Single-pass analytical grad: dL/dx = [2*2, 2*3] = [4, 6]
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()

    # Pass 1 — non-leaf (pow_node) grad starts at 0, so x gets [4, 6]
    y.backward(retain_graph=True)
    assert np.allclose(x.grad, [4., 6.]), f"Pass 1 wrong: {x.grad}" #type:ignore
    print("  ✓ Pass 1 correct: [4, 6]")

    # Pass 2 — pow_node.grad was [1,1] from pass 1, now += [1,1] = [2,2]
    #           x.grad contribution: 2 * [2,2] * [2,3] = [8,12]
    #           total x.grad: [4+8, 6+12] = [12, 18]
    y.backward(retain_graph=True)
    assert np.allclose(x.grad, [12., 18.]), f"Pass 2 wrong: {x.grad}" #type:ignore
    print("  ✓ Pass 2 correct (intermediate grad accumulation): [12, 18]")

    # Pass 3 (retain_graph=False) — frees closures after this pass
    # pow_node.grad was [2,2] → += [1,1] = [3,3]; x.grad += [3,3]*2*[2,3] = [12,18]
    y.backward(retain_graph=False)
    assert np.allclose(x.grad, [24., 36.]), f"Pass 3 wrong: {x.grad}" #type:ignore
    print("  ✓ Pass 3 correct (final, no retain): [24, 36]")

    # Closures are freed after retain_graph=False ────────────────────────
    x.grad = None
    y._backward()   # should be no-op lambda now
    assert x.grad is None, "Closure should be no-op after retain_graph=False"
    print("  ✓ Closures freed (no-op confirmed)")

    # Practical pattern: multi-loss with zeroed intermediate grads ───────
    # To get exactly "one pass" on each backward, zero leaf grads between
    # passes.  Non-leaf grads don't need manual zeroing in the usual training
    # loop because each forward creates fresh intermediate tensors.
    a = Tensor(np.array([1., 2., 3.]), requires_grad=True)
    loss_a = (a * 2).sum()   # dL/da = [2, 2, 2]
    loss_b = (a * 3).sum()   # dL/da = [3, 3, 3]

    # loss_a and loss_b are independent roots sharing leaf `a`.
    # retain_graph on loss_a keeps a's part of the graph alive.
    loss_a.backward(retain_graph=True)    # a.grad = [2, 2, 2]
    loss_b.backward()                     # a.grad += [3, 3, 3] → [5, 5, 5]
    assert np.allclose(a.grad, [5., 5., 5.]), f"Multi-loss wrong: {a.grad}" #type:ignore
    print("  ✓ Multi-loss retain_graph correct: [5, 5, 5]")


def test_no_grad():
    """
    Inside no_grad():
      - requires_grad is overridden to False, even for explicit requires_grad=True
      - No _prev links are stored (graph is not built)
      - .backward() raises RuntimeError (tensor has no grad)
      - Operations that would normally track grad produce non-tracking tensors
      - Exiting the block restores normal grad tracking
    """
    print("\n[test_no_grad]")

    import noni

    # 1. requires_grad forced False inside block ─────────────────────────
    with noni.no_grad():
        t = Tensor([1., 2.], requires_grad=True)   # request grad
    assert not t.requires_grad, \
        "requires_grad should be False inside no_grad()"
    print("  ✓ requires_grad forced False inside no_grad()")

    # 2. No parent links (_prev is empty) ───────────────────────────────
    x = Tensor([3., 4.], requires_grad=True)   # created OUTSIDE → has grad
    with noni.no_grad():
        y = x * 2   # op inside no_grad — output should have no _prev
    assert y._prev == (), \
        f"_prev should be empty inside no_grad, got {y._prev}"
    assert not y.requires_grad, \
        "Output of op inside no_grad should have requires_grad=False"
    print("  ✓ No graph nodes created inside no_grad()")

    # 3. .backward() raises because no graph was built ──────────────────
    try:
        y.backward()
        assert False, "backward() should raise RuntimeError"
    except RuntimeError:
        pass   # expected
    print("  ✓ backward() correctly raises on no-grad tensor")

    # 4. Grad tracking restored after block exits ───────────────────────
    z = x * 3   # outside the block — should build graph normally
    assert z.requires_grad, \
        "requires_grad should be True outside no_grad() for a grad-enabled input"
    assert len(z._prev) > 0, \
        "Graph should be built outside no_grad()"
    print("  ✓ Grad tracking fully restored after exiting no_grad()")

    # 5. Nesting: outer no_grad, inner no_grad — stays disabled ─────────
    with noni.no_grad():
        with noni.no_grad():
            inner = Tensor([1.], requires_grad=True)
        # exiting inner no_grad — should still be disabled (outer still active)
        still_disabled = Tensor([2.], requires_grad=True)
    assert not inner.requires_grad
    assert not still_disabled.requires_grad
    print("  ✓ Nested no_grad blocks stay disabled until outermost exits")

    # 6. Exception safety: no_grad restores state even on exception ─────
    original_state = noni._grad_enabled()
    try:
        with noni.no_grad():
            raise ValueError("deliberate error")
    except ValueError:
        pass
    assert noni._grad_enabled() == original_state, \
        "no_grad() must restore grad state even when an exception occurs"
    print("  ✓ no_grad() restores state on exception (exception-safe)")

    # 7. Inference memory: no graph means no retained activations ───────
    # A rough check: run a forward pass inside and outside no_grad, confirm
    # the no_grad path creates no _prev links anywhere in the chain.
    a = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    with noni.no_grad():
        h  = (a @ Tensor(np.eye(8))).relu()
        out = h.sum()
    assert out._prev == () and h._prev == (), \
        "Intermediate tensors should have no _prev inside no_grad()"
    print("  ✓ No activation graph built during inference (memory-safe)")



if __name__ == "__main__":
    tests = [
        test_add, test_sub, test_mul, test_div, test_pow,
        test_matmul, test_sum, test_mean, test_exp, test_log,
        test_relu, test_gelu, test_sigmoid, test_tanh,
        test_softmax, test_log_softmax,
        test_broadcast_add, test_reshape, test_transpose, test_indexing,
        test_chain_rule, test_layernorm, test_cross_entropy,
        # Phase 1 — Core Engine Hardening
        test_retain_graph,
        test_no_grad,
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
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
