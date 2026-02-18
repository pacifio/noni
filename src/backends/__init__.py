from .base import (
    Backend,
    BackendNotAvailableError,
    get_backend,
    list_backends,
    only_available_backends,
    register_backend,
)
from .numpy_backend import NumpyBackend

register_backend("cpu", NumpyBackend())

__all__ = [
    "Backend", "BackendNotAvailableError",
    "get_backend", "register_backend",
    "only_available_backends",
    "list_backends",
    "NumpyBackend",
]
