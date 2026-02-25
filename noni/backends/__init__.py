from .base import (
    Backend,
    BackendNotAvailableError,
    get_backend,
    list_available_backends,
    list_backends,
    only_available_backends,
    register_backend,
)
from .numpy_backend import NumpyBackend
from .opencl_backend import OpenCLBackend

register_backend("cpu", NumpyBackend())
register_backend("opencl", OpenCLBackend())

__all__ = [
    "Backend", "BackendNotAvailableError",
    "get_backend", "register_backend", "list_available_backends",
    "only_available_backends",
    "list_backends",
    "NumpyBackend",
    "OpenCLBackend"
]
