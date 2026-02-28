import threading
from contextlib import contextmanager
from types import FunctionType
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

	def __empty_fill(self) -> "Tensor":
		return self

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
			if node.grad is not None:
				node._backward()

		if not retain_graph:
			for node in topo:
				node._backward = lambda: None
				node._prev = ()

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
			if self.requires_grad:
				g = _unbroadcast(out.grad, self.shape)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g
			if other.requires_grad:
				g = _unbroadcast(out.grad, other.shape)  # type: ignore
				other.grad = g if other.grad is None else other.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def __radd__(self, other): return self.__add__(other)

	def __sub__(self, other: Union["Tensor", float]) -> "Tensor":
		return self + (-other)

	def __neg__(self) -> "Tensor":
		"""
		Because self-other
		becomes self + (-other) -> see __sub__
		which is self.__add__(other.__neg__())
		which essentially just flips the data and automatically reuses the backward pass from __add__
		"""

		out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="- (neg)")

		def _backward():
			if self.requires_grad:
				g = -out.grad  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
		other = other if isinstance(other, Tensor) else Tensor(other)
		needs_grad = self.requires_grad or other.requires_grad
		out = Tensor(self.data*other.data, requires_grad=needs_grad, _children=(self, other), _op="*")

		def _backward():
			if self.requires_grad:
				g = _unbroadcast(out.grad * other.data, self.shape)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

			if other.requires_grad:
				g = _unbroadcast(out.grad * self.data, other.shape)  # type: ignore
				other.grad = g if other.grad is None else other.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def __rmul__(self, other): return self.__mul__(other)

	def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
		return self * (other ** - 1)

	def __pow__(self, exponent: float) -> "Tensor":
		assert isinstance(exponent, (int, float)), "Exponent must be a scalar"
		out = Tensor(self.data**exponent, requires_grad=self.requires_grad, _children=(self,), _op=f"**{exponent}")

		def _backward():
			if self.requires_grad:
				g = out.grad * (exponent*self.data**(exponent-1))  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def matmul(self, other: "Tensor") -> "Tensor":
		"""
		General matmul:  (..., m, k) @ (..., k, n)  →  (..., m, n)

        This is the primary operation that benefits from GPU acceleration.
        The forward pass dispatches to `self._backend().matmul()`, which
        routes to the appropriate compute backend (NumPy CPU, Triton CUDA,
        OpenCL, CuPy, etc.) based on `self.device`.

        Backward always runs in NumPy on CPU (the "hybrid" architecture):
          dL/dA = dL/dC @ B^T
          dL/dB = A^T @ dL/dC

        Why dispatch only matmul?
        -------------------------
        In a transformer training step the compute breakdown is roughly:
          matmul (QKV, FFN, projection): ~85% of FLOPs
          activations (GELU, softmax):   ~10% of FLOPs
          layer norm, misc:              ~5%  of FLOPs

        Dispatching matmul to GPU and keeping everything else in NumPy
        captures the bulk of the speedup with minimal complexity.
        The softmax operation is also dispatch-ready — see the softmax()
        method below.

        Device propagation
        ------------------
        The output tensor inherits `device` from self.  This means the
        device "sticks" through the computation graph: if x is on 'opencl',
        all downstream tensors are also on 'opencl'.
		"""
		needs_grad = self.requires_grad or other.requires_grad
		device = self.device

		# forward pass needs to be dispatched to selected backend
		result_data = self._backend().matmul(self.data, other.data)
		out = Tensor(result_data, requires_grad=needs_grad, _children=(self, other), _op="@", device=device)

		# backward always happens on CPU
		def _backward():
			if self.requires_grad:
				g = out.grad @ other.data.swapaxes(-1, -2)  # type: ignore
				while g.ndim > self.ndim:
					g = g.sum(axis=0)
				self.grad = g if self.grad is None else self.grad + g
			if other.requires_grad:
				g = self.data.swapaxes(-1, -2) @ out.grad  # type: ignore
				while g.ndim > other.ndim:
					g = g.sum(axis=0)
				other.grad = g if other.grad is None else other.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def __matmul__(self, other): return self.matmul(other)

	def sum(self, axis=None, keepdims=False) -> "Tensor":
		out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op="sum")

		def _backward():
			if self.requires_grad:
				grad = out.grad  # type: ignore
				if axis is not None and not keepdims:
					grad = np.expand_dims(grad, axis=axis) #type:ignore
				g = np.broadcast_to(grad, self.shape).copy() #type:ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def mean(self, axis=None, keepdims=False) -> "Tensor":
		out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
			requires_grad=self.requires_grad, _children=(self,), _op="mean")

		def _backward():
			if self.requires_grad:
				if axis is None:
					n = self.data.size
				else:
					n = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a]for a in axis])
				grad = out.grad  # type: ignore
				if axis is not None and not keepdims:
					grad = np.expand_dims(grad, axis=axis) #type:ignore
				g = np.broadcast_to(grad/n, self.shape).copy() #type:ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def max(self, axis=None, keepdims=False) -> "Tensor":
		val = self.data.max(axis=axis, keepdims=keepdims)
		out = Tensor(val, requires_grad=self.requires_grad, _children=(self,), _op="max")

		def _backward():
			if self.requires_grad:
				v = val if keepdims else (np.expand_dims(val, axis=axis) if axis is not None else val)
				mask: np.ndarray = (self.data == np.broadcast_to(v, self.shape)).astype(float32)
				mask /= mask.sum(axis=axis, keepdims=True)
				grad = out.grad  # type: ignore
				if axis is not None and not keepdims:
					grad = np.expand_dims(grad, axis=axis) #type:ignore
				g = mask * np.broadcast_to(grad, self.shape) #type:ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def var(self, axis=None, keepdims=False) -> "Tensor":
		m = self.data.mean(axis=axis, keepdims=keepdims)
		n = self.data.shape[axis] if axis is not None else self.data.size
		val: np.ndarray = ((self.data-m)**2).sum(axis=axis, keepdims=keepdims) / (n-1)
		out = Tensor(val, requires_grad=self.requires_grad, _children=(self,), _op="var")

		def _backward():
			if self.requires_grad:
				diff = self.data - m # (... - mean)
				grad = out.grad  # type: ignore
				if axis is not None and not keepdims:
					grad = np.expand_dims(grad, axis=axis) #type:ignore
				g = 2*diff*np.broadcast_to(grad, self.shape)/(n-1) #type:ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def exp(self) -> "Tensor":
		val = np.exp(self.data)
		out = Tensor(val, requires_grad=self.requires_grad, _children=(self,), _op="exp")

		def _backward():
			if self.requires_grad:
				g = out.grad * val  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def log(self, suffix=1e-9) -> "Tensor":
		"""
		Log operation on tensor with autograd
		suffix → small constant (1e-9) to avoid log(0) which would be -inf.
		"""
		out = Tensor(np.log(self.data + suffix), requires_grad=self.requires_grad, _children=(self,), _op="log")

		def _backward():
			if self.requires_grad:
				g = out.grad/(self.data + suffix)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def sqrt(self) -> "Tensor":
		return self ** 0.5

	def abs(self) -> "Tensor":
		out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, _children=(self,), _op="abs")

		def _backward():
			if self.requires_grad:
				g = out.grad * np.sign(self.data)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def clip(self, min_val, max_val, mask_type=float32) -> "Tensor":
		out = Tensor(np.clip(self.data, min_val, max_val), requires_grad=self.requires_grad, _children=(self,), _op="clip")

		def _backward():
			if self.requires_grad:
				mask = ((self.data >= min_val) & (self.data <= max_val)).astype(mask_type)
				g = out.grad * mask  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out


	def reshape(self, *shape) -> "Tensor":
		out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")

		def _backward():
			if self.requires_grad:
				g = out.grad.reshape(self.shape)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward
		return out

	def view(self, *shape) -> "Tensor":
		return self.reshape(*shape)

	def transpose(self, ax1: int = -2, ax2: int = -1) -> "Tensor":
		axes = list(range(self.ndim))
		axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
		out = Tensor(
			self.data.transpose(axes), requires_grad=self.requires_grad,
			_children=(self,), _op="T"
		)

		def _backward():
			if self.requires_grad:
				inv_axes = np.argsort(axes)
				g = out.grad.transpose(inv_axes)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def permute(self, *axes) -> "Tensor":
		out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, _children=(self,), _op="permute")

		def _backward():
			if self.requires_grad:
				inv = np.argsort(axes)
				g = out.grad.transpose(inv)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def unsqueeze(self, dim:int) -> "Tensor":
		out = Tensor(np.expand_dims(self.data, axis=dim),
			requires_grad=self.requires_grad, _children=(self,), _op="unsqueeze")

		def _backward():
			if self.requires_grad:
				g = out.grad.squeeze(axis=dim)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def squeeze(self, dim: Optional[int] = None) -> "Tensor":
		out = Tensor(self.data.squeeze(axis=dim), requires_grad=self.requires_grad, _children=(self,), _op="squeeze")

		def _backward():
			if self.requires_grad:
				g = out.grad.reshape(self.shape)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def flatten(self, start_dim: int = 0) -> "Tensor":
		new_shape = self.shape[:start_dim] + (-1, )
		return self.reshape(*new_shape)

	def __getitem__(self, idx) -> "Tensor":
		out = Tensor(self.data[idx], requires_grad=self.requires_grad, _children=(self,), _op="index")

		def _backward():
			if self.requires_grad:
				g = np.zeros_like(self.data)
				np.add.at(g, idx, out.grad)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def masked_fill(self, mask: np.ndarray, value: float, dtype=float32) -> "Tensor":
		"""
		Fill positions where mask==True with value.
        mask is broadcast to self.shape automatically.
		"""
		fill_arr = np.full(self.shape, value, dtype=dtype)
		new_data = np.where(np.broadcast_to(mask, self.shape), fill_arr, self.data)
		out = Tensor(
			new_data.astype(dtype),
			requires_grad=self.requires_grad,
			_children=(self,),
			_op="masked_fill"
		)

		mask_bc = np.broadcast_to(mask, self.shape)

		def _backward():
			if self.requires_grad:
				g = out.grad.copy()  # type: ignore
				g[mask_bc] = 0.0
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	@staticmethod
	def cat(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
		needs_grad = any(t.requires_grad for t in tensors)
		out = Tensor(np.concatenate([t.data for t in tensors], axis=axis),
                     requires_grad=needs_grad,
                     _children=tuple(tensors), _op="cat")

		def _backward():
			sizes = [t.shape[axis] for t in tensors]
			grads = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)  # type: ignore
			for t,g in zip(tensors, grads):
				if t.requires_grad:
					t.grad = g if t.grad is None else t.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	@staticmethod
	def stack(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
		needs_grad = any(t.requires_grad for t in tensors)
		out = Tensor(np.stack([t.data for t in tensors], axis=axis),
                     requires_grad=needs_grad,
                     _children=tuple(tensors), _op="stack")

		def _backward():
			for i, t in enumerate(tensors):
				if t.requires_grad:
					g = np.take(out.grad, i, axis=axis)  # type: ignore
					t.grad = g if t.grad is None else t.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def relu(self) -> "Tensor":
		mask = self.data > 0
		out = Tensor(self.data * mask, requires_grad=self.requires_grad,
                     _children=(self,), _op="relu")

		def _backward():
			if self.requires_grad:
				g = out.grad * mask  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def gelu(self) -> "Tensor":
		x = self.data
		cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
		val = x*cdf
		out = Tensor(val, requires_grad=self.requires_grad,
                     _children=(self,), _op="gelu")

		def _backward():
			if self.requires_grad:
				tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
				tanh_val = np.tanh(tanh_arg)
				sech2    = 1.0 - tanh_val ** 2
				dcdf_dx  = 0.5 * sech2 * np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
				g = out.grad * (cdf + x * dcdf_dx)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def sigmoid(self) -> "Tensor":
		val = 1.0 / (1.0 + np.exp(-self.data))
		out = Tensor(val, requires_grad=self.requires_grad,
                     _children=(self,), _op="sigmoid")

		def _backward():
			if self.requires_grad:
				g = out.grad * val * (1.0 - val)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def tanh(self) -> "Tensor":
		val = np.tanh(self.data)
		out = Tensor(val, requires_grad=self.requires_grad,
                     _children=(self,), _op="tanh")

		def _backward():
			if self.requires_grad:
				g = out.grad * (1.0 - val ** 2)  # type: ignore
				self.grad = g if self.grad is None else self.grad + g

		if out.requires_grad:
			out._backward = _backward

		return out

	def softmax(self, axis: int = -1) -> "Tensor":
		"""
		Numerically stable softmax with correct Jacobian backward.

        For softmax s_i = exp(x_i) / sum_j exp(x_j):
          d(sum L_i s_i) / d(x_j) = s_j * (L_j - sum_i L_i * s_i)
        which simplifies to: p * (g - (g*p).sum(axis, keepdims=True))
		"""

		shifted = self.data - self.data.max(axis=axis, keepdims=True)
		e = np.exp(shifted)
		s = e.sum(axis=axis, keepdims=True)
		p = e / s
		out = Tensor(p, requires_grad=self.requires_grad,
                     _children=(self,), _op="softmax")

		def _backward():
			if self.requires_grad:
				g = out.grad  # type: ignore
				dot = (g * p).sum(axis=axis, keepdims=True)   # (..., 1)
				grad_in = p * (g - dot)                       # broadcast back
				self.grad = grad_in if self.grad is None else self.grad + grad_in

		if out.requires_grad:
			out._backward = _backward

		return out

	def log_softmax(self, axis: int = -1) -> "Tensor":
		"""
		Numerically stable log-softmax.

        log_softmax(x_i) = x_i - log(sum_j exp(x_j))

        Backward:
          If L = sum_i w_i * log_softmax(x_i)
          dL/dx_j = w_j - softmax(x_j) * sum_i w_i
                  = g_j - softmax(x_j) * sum_i g_i
		"""
		shifted = self.data - self.data.max(axis=axis, keepdims=True)
		log_sum_exp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
		val = shifted - log_sum_exp   # log_softmax values
		out = Tensor(val, requires_grad=self.requires_grad,
                     _children=(self,), _op="log_softmax")

		def _backward():
			if self.requires_grad:
				g = out.grad  # type: ignore
				softmax_val = np.exp(val)
				grad_in = g - softmax_val * g.sum(axis=axis, keepdims=True) #type:ignore
				self.grad = grad_in if self.grad is None else self.grad + grad_in

		if out.requires_grad:
			out._backward = _backward

		return out

	def __gt__(self, other):  return Tensor(self.data > (other.data if isinstance(other, Tensor) else other))
	def __lt__(self, other):  return Tensor(self.data < (other.data if isinstance(other, Tensor) else other))
	def __ge__(self, other):  return Tensor(self.data >= (other.data if isinstance(other, Tensor) else other))
	def __le__(self, other):  return Tensor(self.data <= (other.data if isinstance(other, Tensor) else other))
	def __eq__(self, other):  return Tensor(self.data == (other.data if isinstance(other, Tensor) else other)) #type:ignore

	def item(self):
		return self.data.item()

	def numpy(self) -> np.ndarray:
		return self.data.copy()

	def detach(self) -> "Tensor":
		return Tensor(self.data.copy(), requires_grad=False)


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

	def test_negation():
		x = Tensor(3.0, requires_grad=True)
		y = Tensor(4.0, requires_grad=True)
		z = x-y
		z.backward()
		print(x.grad)

	def test_multiplication():
		x = Tensor(40.0, requires_grad=True)
		y = Tensor(18.0, requires_grad=True)
		z = x*y
		z.backward()
		print(x.grad)

	test_functions: list[FunctionType] = [
		test_addition,
		test_negation,
		test_multiplication,
	]


	[f() for f in test_functions]
