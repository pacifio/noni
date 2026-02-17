from .base import Backend, BackendNotAvailableError, get_backend, register_backend
from .numpy_backend import NumpyBackend

register_backend("cpu", NumpyBackend())

__all__ = [
    "Backend", "BackendNotAvailableError",
    "get_backend", "register_backend",
    "NumpyBackend",
]
