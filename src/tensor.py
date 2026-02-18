import threading
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np

float16 = np.float16
float32 = np.float32
float64 = np.float64


_thread_local = threading.local()

def _grad_enabled() -> bool:
	return getattr(_thread_local, "grad_enabled", True)

@contextmanager
def no_grad():
	"""
    Context manager: disable gradient computation for the enclosed block.

    While active, Tensor.__init__ ignores requires_grad=True and sets it
    to False instead.  No _backward closures are registered and no parent
    references (_prev) are stored.  This halves memory use at inference
    time because intermediate activations no longer need to be retained for
    the backward pass.

    Usage
    -----
    >>> with noni.no_grad():
    ...     logits = model(x)   # pure forward pass — no graph built

    Can be used as a decorator too:
    >>> @noni.no_grad()   # note: must call it as no_grad()
    ... def predict(x):
    ...     return model(x)

    Nesting is safe:
    >>> with noni.no_grad():
    ...     with noni.no_grad():   # fine — stays disabled
    ...         y = x + 1

    Thread safety:
    The flag lives on threading.local(), so each thread has its own
    independent copy.  Worker threads in a DataLoader are unaffected.
    """

	prev = _grad_enabled()
	_thread_local.grad_enabled = False

	try:
		yield
	finally:
		_thread_local.grad_enabled = prev


def _ensure_array(x, dtype=float32) -> np.ndarray:
	"""Convert python scalars / lists / existing ndarrays to dtype, float32 is default."""
	if isinstance(x, np.ndarray):
		return x.astype(dtype) if x.dtype != dtype else x
	return np.array(x, dtype=dtype)


def _unbroadcast(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
	"""
 	NumPy broadcasting means a (3,) tensor added to a (4,3) tensor produces
    a (4,3) grad.  We must sum back over the broadcast dims to match the
    original leaf shape.
	"""
	# pad original_shape to same ndim as grad
	ndim_diff = grad.ndim - len(original_shape)
	padded = (1, ) * ndim_diff + tuple(original_shape)

	# sum over dims that were broadcast
	axes = tuple(i for i, (g, o) in enumerate(zip(grad.shape, padded)) if o == 1)
	if axes:
		grad = grad.sum(axis=axes, keepdims=True)
	return grad.reshape(original_shape)

def _get_backend(device:str):
	"""
	Lazy backend lookup.
	"""

	from .backends import get_backend
	return get_backend(device)

class Tensor:
	"""
	A multidimensional array that supports automatic differentiation.

    Parameters
    ----------
    data          : array-like — the underlying numbers
    requires_grad : track gradients through this tensor
                    Silently overridden to False when inside a no_grad() block.
    device        : which compute device to use for forward operations.
                    Default 'cpu'.  Other values: 'opencl', 'cuda', 'cupy',
                    'vulkan'.  See noni/backends/ for details.
    _children     : parent tensors that produced this one (internal use)
    _op           : string label of the operation (debugging / pretty-printing)

    no_grad interaction
    -------------------
    When _grad_enabled() is False (i.e. inside a `with noni.no_grad():` block):
      - requires_grad is forced to False regardless of what the caller passed.
      - _children is set to () so no parent references are stored.
      - _backward remains the no-op lambda.
    This means the tensor is a "dead end" in the graph — it accumulates no
    memory for the backward pass, which is exactly what inference needs.

    Device & backend
    ----------------
    The `device` attribute selects which backend handles compute-heavy ops
    (matmul, softmax, etc.).  The autograd graph and gradient accumulation
    always run in NumPy on the CPU — only forward ops are dispatched to GPU.

    This is the "NumPy-on-CPU autograd, GPU-for-GEMM" architecture.
    It gives most of the training speedup (matmul dominates) with minimal
    complexity (no need to port every backward closure to GPU).

    For a fully on-device autograd (like PyTorch), see FUTURE.md Phase 7.

    Usage example:
    >>> x = Tensor(data, device='opencl', requires_grad=True)
    >>> y = x @ w      # matmul dispatched to OpenCL GPU
    >>> y.backward()   # backward runs in NumPy on CPU
    >>> # Move to different device:
    >>> x_cuda = x.to('cuda')
	"""

	def __init__(self, data,
		requires_grad:bool=False,
		dtype=float32,
		_children: Tuple["Tensor", ...] = (),
		_op: str = "",
		device: str="cpu"
	) -> None:
		self.data: np.ndarray = _ensure_array(data, dtype=dtype)
		self.device = device

		if not _grad_enabled():
			requires_grad = False
			_children = () # no graph nodes created

		self.requires_grad: bool = requires_grad
		self.grad: Optional[np.ndarray] = None

		# autograd
		self._backward = lambda: None # filled up by each operation
		self._prev: Tuple["Tensor", ...] = _children
		self._op: str = _op

	@property
	def shape(self) -> tuple:
		return self.data.shape

	@property
	def ndim(self) -> int:
		return self.data.ndim

	@property
	def dtype(self):
		return self.data.dtype

	@property
	def T(self) -> "Tensor":
		return self.transpose()

	def transpose(self) -> "Tensor":
		return self

	def __repr__(self) -> str:
		grad_str = f", required_grad={self.requires_grad}" if self.requires_grad else ""
		op_str = f", op='{self._op}'" if self._op else ""
		dtype_str = f", dtype={self.dtype}" if self.dtype else "noni.float32"
		return f"Tensor({self.data}{grad_str}{op_str}{dtype_str})"

	def backward(self, grad: Optional[np.ndarray] = None, retain_graph: bool = False):
		"""
		Reverse-mode autodiff via topological sort + DFS.

        Call on a *scalar* loss.  Pass grad=None to seed with ones (the
        standard case).  Pass a numpy array for non-scalar outputs or custom
        gradient seeding.

        Parameters
        ----------
        grad          : seed gradient — shape must match self.shape.
                        Defaults to np.ones_like(self.data) for scalar tensors.
        retain_graph  : if False (default), free the backward closures after
                        the pass to release memory.  Set True to keep the graph
                        alive for a second backward call.

        retain_graph=True use cases
        ---------------------------
        1. Adversarial training (FGSM / PGD):
               loss.backward(retain_graph=True)   # get input grads for perturbation
               optimizer.step()                   # update model weights
               # graph still alive → can call backward again on same loss

        2. Multiple losses sharing a graph:
               loss_a.backward(retain_graph=True)
               loss_b.backward()                  # graph freed here

        3. Higher-order gradients (grad of grad):
               # First backward builds a new graph if create_graph were
               # implemented.  retain_graph keeps the original alive.

        Memory note
        -----------
        With retain_graph=False (default): after backward(), the _backward
        closures on every node are replaced with a no-op lambda.  The _prev
        parent references are not cleared (that would break future graph
        traversals if you do retain_graph on a parent node), but the
        closures — which hold references to the intermediate activation arrays
        captured during the forward pass — are freed.  Python's reference
        counter then collects those arrays, reclaiming their memory.

        With retain_graph=True: closures are kept intact.  The intermediate
        activations remain in memory until the Tensor objects themselves go
        out of scope.  This is why repeated backward() without retain_graph
        on a large model will OOM — the activations from every step
        accumulate.
		"""

		if not self.requires_grad:
			raise RuntimeError(
				"backward() called on a tensor with requires_grad=False"
			)

		if grad is None:
			"""
			NOTE:
			We want to pass only scalars to backward()
			   x1 ----\
			            \
			             →  L  (scalar)
			            /
			   x2 ----/
						∂x1/∂L,∂x2/∂L

			We are not supposed to pass down vectors because we
			don't want to compute a complete jacobian matrix
			which would look like this

			   x1 ----\
			            \
			             →  y = [y1, y2, y3]
			            /
			   x2 ----/
			"""
			if self.data.size != 1:
				raise RuntimeError(
					"backward() without a gradient argument is only supported "
					f"for scalar tensor. Got shape {self.shape}. "
					"Pass grad explicitly for non scalar outputs."
				)
			grad = np.ones_like(self.data, dtype=self.dtype if self.dtype else float32)
		self.grad = grad.astype(dtype=self.dtype if self.dtype else float32)

		topo: List["Tensor"] = []
		visited = set()

		# Lazy loaded
		# Only built if backward() is invoked
		def _build_topo(node: "Tensor"):
			if id(node) not in visited:
				visited.add(id(node))
				for child in node._prev:
					_build_topo(child)
				topo.append(node)

		_build_topo(self)

		for node in reversed(topo):
			node._backward()

		if not retain_graph:
			for node in topo:
				node._backward = lambda: None

	def zero_grad(self):
		self.grad = None

	def to(self, device: str, safe: bool = False) -> "Tensor":
		"""
		Return a new Tensor on the specified device.

        The data (self.data numpy array) is always kept on CPU in Noni's
        current architecture — 'device' selects which backend handles the
        forward compute ops, not where the raw bytes live.

        This mirrors PyTorch's .to(device) semantics for the ops that matter:
        model(x.to('opencl')) will run matmul, softmax etc. on the OpenCL GPU
        even though x.data is still a numpy array.

        Parameters
        ----------
        device : str
            One of: 'cpu', 'numpy', 'opencl', 'cuda', 'triton', 'cupy', 'vulkan'

        Important note on devices:
        	Only numpy and opencl backend is being worked on
        	every other backend are planned and specially metal will get first party priority

        Returns
        -------
        A new Tensor with the same data and requires_grad, but device=device.

        Example
        -------
        >>> x = Tensor(np.random.randn(512, 512), device='cpu')
        >>> x_gpu = x.to('opencl')    # will use OpenCL for matmul
        >>> y = x_gpu @ w             # matmul runs on OpenCL GPU
        >>> y.backward()              # backward still runs on CPU

        Moving a model to GPU
        ---------------------
        >>> for p in model.parameters():
        ...     p.data = p.data  # data stays numpy
        ...     p.device = 'opencl'  # just change the device tag

        Or use Module.to(device):
        >>> model.to('opencl')
		"""

		if safe:
			from backends import list_backends, only_available_backends
			if device not in only_available_backends():
				raise RuntimeError(
					f"Device {device} is not supported, list of available backends -> ",
					f"{list_backends()}"
				)

		new_t = Tensor(self.data, requires_grad=self.requires_grad, device=device)
		if self.grad is not None:
			new_t.grad = self.grad.copy()
		return new_t

	def cuda(self) -> "Tensor":
		if self.device == "cuda":
			return self
		return self.to("cuda")

	def cpu(self) -> "Tensor":
		if self.device == "cpu":
			return self
		return self.to("cpu")

	def _backend(self):
		"""Get the backend for this tensor's device.  Cached on first call."""
		return _get_backend(self.device)

	@classmethod
	def zeros(cls, *shape, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

	@classmethod
	def ones(cls, *shape, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

	@classmethod
	def randn(cls, *shape, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

	@classmethod
	def rand(cls, *shape, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

	@classmethod
	def arange(cls, *args, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(np.arange(*args, dtype=dtype), requires_grad=requires_grad)

	@classmethod
	def eye(cls, n, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(np.eye(n, dtype=dtype), requires_grad=requires_grad)

	@classmethod
	def from_numpy(cls, arr: np.ndarray, requires_grad=False, dtype=float32) -> "Tensor":
		return cls(arr.astype(dtype), requires_grad=requires_grad)

	def __add__(self, other: Union["Tensor", float]) -> "Tensor":
		other = other if isinstance(other, Tensor) else Tensor(other)
		needs_grad = self.requires_grad or other.requires_grad
		out = Tensor(self.data + other.data, requires_grad=needs_grad, _children=(self, other), _op="+")

		def _backward():
			if out.grad is None:
				return

			if self.requires_grad:
				g = _unbroadcast(out.grad, self.shape)
				self.grad = g if self.grad is None else self.grad + g
			if other.requires_grad:
				g = _unbroadcast(out.grad, other.shape)
				other.grad = g if other.grad is None else other.grad + g

		out._backward = _backward
		return out

"""
For internal testing only
"""
if __name__ == "__main__":
	def test_addition():
		x = Tensor(3.0, requires_grad=True)
		y = Tensor(4.0, requires_grad=True)
		z = x + y
		z.backward()
		print(x.grad)

	test_addition()
