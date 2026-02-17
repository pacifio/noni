"""
noni/backends/base.py
==========================
Abstract Backend interface and global registry.

Every backend must implement this interface.  The Tensor class calls these
methods instead of calling numpy/triton/opencl directly, which is what makes
the multi-backend architecture possible.

Design principles
-----------------
1. **All operations take and return numpy arrays** at the Python boundary.
   The backend is responsible for any host↔device transfer needed.
   This keeps the autograd engine (tensor.py) completely device-agnostic.

2. **Lazy transfer** — operations on the same device stay on device.
   The `to_device` / `from_device` methods are only called explicitly by
   `.to(device)` or when printing/inspecting a tensor.

3. **Graceful degradation** — if a backend isn't installed or no compatible
   hardware is found, it falls back to raising `BackendNotAvailableError`
   at construction time.  The user gets a clear message rather than a
   cryptic import error.

4. **Stateless methods** — all ops are pure functions of their numpy array
   inputs.  State (e.g. OpenCL command queue, CUDA stream) lives on the
   Backend instance, not on individual tensors.  This avoids reference
   cycles and simplifies serialisation.

Interface contract
------------------
Every concrete subclass must implement:

  Core linear algebra:
    matmul(a, b)              → C = A @ B
    add(a, b)                 → element-wise a + b  (with broadcasting)
    multiply(a, b)            → element-wise a * b  (with broadcasting)
    sum(a, axis, keepdims)    → reduction
    exp(a)                    → element-wise e^a
    log(a)                    → element-wise ln(a)
    relu(a)                   → max(0, a)
    softmax(a, axis)          → numerically stable softmax

  Data transfer:
    to_device(arr)            → "device array" (opaque to tensor.py)
    from_device(dev_arr)      → np.ndarray  (always float32)

  Metadata:
    name: str                 → human-readable name, e.g. "triton/cuda"
    is_available() → bool     → can we actually use this backend?
"""

import abc
from typing import Optional

import numpy as np

_GELU_CONSTANT = 0.044715

class BackendNotAvailableError(RuntimeError):
	"""Raised when a requested backend is not available on this machine."""
	pass

class Backend(abc.ABC):
	"""
    Abstract base class for Noni compute backends.

    Subclasses implement the actual computation using their specific library
    (numpy, triton, pyopencl, cupy, kompute…).  All methods receive numpy
    float32 arrays and return numpy float32 arrays — the backend handles all
    host-to-device and device-to-host transfers internally.

    Why numpy arrays at the boundary?
    ----------------------------------
    The autograd engine (tensor.py) was written with numpy in mind.  Rather
    than rewrite every backward pass for every backend, we keep the autograd
    graph in numpy space and only dispatch the *heavy compute* (forward ops)
    to the accelerator.  For training transformers, >95% of FLOPs are in
    matmul, so this gives most of the speedup with minimal complexity.

    A production implementation (PyTorch, JAX) keeps tensors on device for
    the entire forward + backward pass.  That requires rewriting every
    backward closure too — see FUTURE.md Phase 7 for that roadmap.
    """

	@property
	@abc.abstractmethod
	def name(self) -> str:
		"""backend identifier e.g triton/cuda/opencl"""

	@abc.abstractmethod
	def is_available(self) -> bool:
		"""

		"""

	@abc.abstractmethod
	def to_device(self, arr: np.ndarray) -> object:
		"""

		"""

	@abc.abstractmethod
	def from_device(self, dev_arr: object) -> np.ndarray:
		"""

		"""

	@abc.abstractmethod
	def matmul(self, a:np.ndarray, b: np.ndarray):
		"""

		"""

	@abc.abstractmethod
	def add(self, a:np.ndarray, b:np.ndarray) -> np.ndarray:
		""""""

	@abc.abstractmethod
	def sum(self, a:np.ndarray, axis: Optional[int]=None, keepdims:bool=False) -> np.ndarray:
		"""

		"""

	@abc.abstractmethod
	def exp(self, a: np.ndarray) -> np.ndarray:
		"""

		"""

	@abc.abstractmethod
	def relu(self, a:np.ndarray) -> np.ndarray:
		"""

		"""

	@abc.abstractmethod
	def softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
		"""

		"""

	def layer_norm(self, a:np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
		"""
		LayerNorm: normalise over the last dimension, apply affine

		Default implementation uses numpy, Backends can override this
		for a much more optimised fused kernel that avoids two passes over the data.
		"""

		mean = a.mean(axis=-1, keepdims=True)
		sq: np.ndarray = ((a-mean)**2)
		var = sq.mean(axis=-1, keepdims=True)
		norm = (a-mean)/np.sqrt(var+eps)

		return norm*weight + bias

	def gelu(self, a:np.ndarray) -> np.ndarray:
		"""
		GELU activation, Default uses numpy, backends can fuse with linear.

		GELU(x)=x⋅Φ(x)
		Φ(x) = cdf = approximation of the standard normal cumulative distribution function

		The fused linear+GELU pattern (typically used in the feed forward network of transformers)
			output = GELU(x@W+b)
		can be implemented as one kernel reading x and W once
		"""

		cdf = 0.5 * (1.0+np.tanh(np.sqrt(2.0/np.pi)*
				(a+_GELU_CONSTANT * a ** 3)
			))
		return a*cdf

_REGISTRY: dict[str, Backend] = {}

def register_backend(device: str, backend: Backend) -> None:
	_REGISTRY[device] = backend

def get_backend(device:str) -> Backend:
	if device not in _REGISTRY:
		available = list(_REGISTRY.keys())
		raise BackendNotAvailableError(
			f"Unknown device {device}. Supported Noni devices\n"
			f"{available}\n"
			"To add new device from noni.backends import Backend, register_device"
		)

	backend = _REGISTRY[device]

	if not backend.is_available():
		raise BackendNotAvailableError(
		    f"Backend '{device}' ({backend.name}) is registered but not "
		    f"available on this machine.  Make sure the required library is "
		    f"installed and hardware is present.\n"
		    f"  Triton/CUDA:  requires NVIDIA GPU + CUDA toolkit + triton\n"
		    f"  OpenCL:       requires any GPU/CPU with OpenCL drivers + pyopencl\n"
		    f"  CuPy:         requires NVIDIA GPU + CUDA toolkit + cupy\n"
		    f"  Vulkan:       requires Vulkan-capable GPU + kompute\n"
		    f"  CPU:          always available"
		)

	return backend

def get_backend_safe(device: str) -> Backend | None:
	return _REGISTRY[device] if device in _REGISTRY else None

def list_backends() -> dict:
	return {
		device: (b.name, b.is_available) for device, b in _REGISTRY.items()
	}

def only_available_backends() -> list[str]:
	return [device for device, b in _REGISTRY.items() if b.is_available()]
