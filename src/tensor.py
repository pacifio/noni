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
