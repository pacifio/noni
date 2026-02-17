import threading
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np

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


def _ensure_array(x, dtype=np.float32) -> np.ndarray:
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
		dtype=np.float32,
		_children: Tuple["Tensor", ...] = (),
		_op: str = "",
		device: str="cpu"
	) -> None:
		self.data: np.ndarray = _ensure_array(data, dtype=dtype)
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
