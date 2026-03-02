"""
Noni — a tiny tensor library with autograd, for humans.
- Reverse-mode autodiff via dynamic computational graphs
- NumPy-only core — no C extensions, no CUDA required
- Multi-backend GPU acceleration via .to(device):
    'cpu'    → NumPy (always works)
    'opencl' → any GPU via pyopencl (AMD, Intel, NVIDIA, Apple)
    'cuda'   → NVIDIA GPU via Triton JIT kernels #WIP
    'cupy'   → NVIDIA GPU via CuPy (NumPy-compatible GPU arrays) #WIP
    'vulkan' → any GPU via Kompute + SPIR-V shaders #WIP
"""

from .backends.base import (
    BackendNotAvailableError,
    get_backend,
    list_available_backends,
    register_backend,
)
from .tensor import Tensor, _grad_enabled, no_grad


def zeros(*shape, requires_grad=False, device='cpu'):
    return Tensor.zeros(*shape, requires_grad=requires_grad)

def ones(*shape, requires_grad=False, device='cpu'):
    return Tensor.ones(*shape, requires_grad=requires_grad)

def randn(*shape, requires_grad=False, device='cpu'):
    return Tensor.randn(*shape, requires_grad=requires_grad)

def rand(*shape, requires_grad=False, device='cpu'):
    return Tensor.rand(*shape, requires_grad=requires_grad)

def arange(*args, requires_grad=False, device='cpu'):
    return Tensor.arange(*args, requires_grad=requires_grad)

def eye(n, requires_grad=False, device='cpu'):
    return Tensor.eye(n, requires_grad=requires_grad)

def tensor(data, requires_grad=False, device='cpu'):
    return Tensor(data, requires_grad=requires_grad, device=device)

def from_numpy(arr, requires_grad=False, device='cpu'):
    return Tensor.from_numpy(arr, requires_grad=requires_grad)

__version__ = "0.1.2"
__all__ = [
    "Tensor", "no_grad", "_grad_enabled",
    "zeros", "ones", "randn", "rand", "arange", "eye", "tensor", "from_numpy",
    "get_backend", "register_backend", "list_available_backends",
    "BackendNotAvailableError",
]
