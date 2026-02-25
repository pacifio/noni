import math
from typing import Dict, Iterator, List, Optional, OrderedDict, Union, overload

import numpy as np

from ..tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    Mirrors the PyTorch Module API at a minimal level.
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Recursively collect all Tensors that require grad."""
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
            elif isinstance(attr, dict):
                for item in attr.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd = {}
        for name, attr in self.__dict__.items():
            if isinstance(attr, Tensor):
                sd[name] = attr.data.copy()
            elif isinstance(attr, Module):
                for k, v in attr.state_dict().items():
                    sd[f"{name}.{k}"] = v
        return sd

    def load_state_dict(self, sd: Dict[str, np.ndarray]):
        for name, val in sd.items():
            parts = name.split(".", 1)
            if len(parts) == 1:
                attr = getattr(self, name, None)
                if isinstance(attr, Tensor):
                    attr.data = val.astype(np.float32)
            else:
                sub = getattr(self, parts[0], None)
                if isinstance(sub, Module):
                    sub.load_state_dict({parts[1]: val})

    def train(self):
        self._training = True
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train()

    def eval(self):
        self._training = False
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.eval()

    def to(self, device: str) -> "Module":
        """
        Move all parameters and buffers to the specified device.

        Changes the `.device` attribute on every Tensor in this module
        (and all submodules) so future forward ops are dispatched to the
        given backend.

        Parameters
        ----------
        device : str
            'cpu', 'opencl', 'cuda', 'cupy', 'vulkan'

        Returns
        -------
        self (for method chaining: model.to('opencl').train())

        Example
        -------
        >>> model = MiniGPT(config)
        >>> model.to('opencl')           # move all params to OpenCL backend
        >>> logits = model(x)            # matmuls now run on OpenCL GPU
        >>> loss.backward()              # backward still on CPU

        Notes
        -----
        In this architecture, `device` is a routing tag, not a memory address.
        The actual numpy arrays stay on CPU host memory.  Only the dispatch
        of compute-heavy ops (matmul, softmax) changes.

        For a fully on-device model (all arrays in GPU memory), see
        FUTURE.md Phase 7.
        """
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                attr.device = device
            elif isinstance(attr, Module):
                attr.to(device)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Tensor):
                        item.device = device
                    elif isinstance(item, Module):
                        item.to(device)
        return self

    @property
    def training(self):
        return getattr(self, "_training", True)

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (Module, Tensor)):
                lines.append(f"  {k}={v},")
        lines.append(")")
        return "\n".join(lines)

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())


class Linear(Module):
    """
    y = x W^T + b

    Kaiming uniform initialization (fan_in) — good default for ReLU networks.
    For attention / embedding projections you may prefer a smaller init;
    we expose `scale` for that.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, scale: Optional[float] = None):
        self.in_features  = in_features
        self.out_features = out_features

        # Kaiming uniform: std = sqrt(2 / fan_in)
        std = math.sqrt(2.0 / in_features) if scale is None else scale
        self.weight = Tensor(
            np.random.randn(out_features, in_features).astype(np.float32) * std,
            requires_grad=True
        )
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32),
                               requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features)  →  (..., out_features)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return (f"Linear(in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None})")


class Embedding(Module):
    """
    Lookup table: integer indices → dense vectors.

    We store the table as a Tensor and use __getitem__ which has autograd,
    so gradients flow back through embeddings via scatter-add.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        # Small random init like PyTorch default
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )

    def forward(self, idx) -> Tensor:
        # idx can be a numpy int array or Tensor; we want the raw indices
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int32)
        return self.weight[idx]

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class LayerNorm(Module):
    """
    Normalize over the last dimension(s) and apply learned affine transform.

    Forward:
      y = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.bias   = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(axis=axis, keepdims=True)
        var  = ((x - mean) ** 2).mean(axis=axis, keepdims=True)
        x_hat = (x - mean) * ((var + self.eps) ** -0.5)
        return x_hat * self.weight + self.bias

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


class Dropout(Module):
    """
    Randomly zero out elements with probability p during training.
    Scaled by 1/(1-p) to keep expected values the same.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(mask)

    def __repr__(self):
        return f"Dropout(p={self.p})"


class SequentialSimple(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        self._training = True
        for l in self.layers: l.train()

    def eval(self):
        self._training = False
        for l in self.layers: l.eval()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.gelu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: return x.tanh()

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor: return x.sigmoid()


class CrossEntropyLoss(Module):
    """
    Combines LogSoftmax + NLL in one shot (numerically stable).

    inputs : (N, C) or (B, T, C) logits — any shape where last dim is classes
    targets: (N,) or (B, T)       int indices
    """

    def __init__(self, ignore_index: int = -1):
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets) -> Tensor:
        if isinstance(targets, Tensor):
            targets = targets.data.astype(np.int32)

        # Flatten batch dims: (..., C) → (N, C)
        C = logits.shape[-1]
        flat_logits  = logits.reshape(-1, C)
        flat_targets = targets.reshape(-1)

        log_probs = flat_logits.log_softmax(axis=-1)  # (N, C)

        # Gather correct class log-probs
        N = flat_targets.shape[0]
        idx = np.arange(N)
        correct_log_probs = log_probs[idx, flat_targets]  # (N,)

        # Mask out ignore_index
        if self.ignore_index >= 0:
            mask = (flat_targets != self.ignore_index).astype(np.float32)
            loss = -(correct_log_probs * Tensor(mask)).sum() / max(mask.sum(), 1e-9)
        else:
            loss = -correct_log_probs.mean()

        return loss


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean()


class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # log(1 + exp(-x)) for positive, log(1 + exp(x)) for negative (numerically stable)
        return (logits.relu() - logits * target + ((-logits.abs()).exp() + 1).log()).mean()


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor,
    mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tensor:
    """
    Vanilla attention:  softmax(Q K^T / sqrt(d_k)) V

    q, k, v: (B, H, T, d_k)
    mask   : boolean array (T, T) or (B, H, T, T)  — True = masked (fill -inf)

    Returns (B, H, T, d_k)
    """
    d_k = q.shape[-1]
    scale = math.sqrt(d_k)

    scores = q @ k.transpose(-2, -1) / scale   # (B, H, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    attn = scores.softmax(axis=-1)              # (B, H, T, T)

    if dropout_p > 0.0 and training:
        drop_mask = (np.random.rand(*attn.shape) > dropout_p).astype(np.float32)
        attn = attn * Tensor(drop_mask) / (1.0 - dropout_p)

    return attn @ v                             # (B, H, T, d_k)


class MultiHeadAttention(Module):
    """
    Multi-head self (or cross) attention.

    Supports a causal mask (for autoregressive GPT-style models).
    """

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1, causal: bool = False):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.causal   = causal
        self.dropout  = dropout

        # All four projections in one shot
        self.q_proj = Linear(d_model, d_model, scale=0.02)
        self.k_proj = Linear(d_model, d_model, scale=0.02)
        self.v_proj = Linear(d_model, d_model, scale=0.02)
        self.o_proj = Linear(d_model, d_model, scale=0.02)
        self.drop   = Dropout(dropout)

    def forward(self, x: Tensor,
                context: Optional[Tensor] = None,
                key_mask: Optional[np.ndarray] = None) -> Tensor:
        B, T, _ = x.shape
        kv_src = context if context is not None else x

        # Project & split heads: (B, T, d_model) → (B, H, T, d_k)
        def split(t):
            return t.reshape(B, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        q = split(self.q_proj(x))
        k = split(self.k_proj(kv_src))
        v = split(self.v_proj(kv_src))

        # Causal mask: upper-triangular = True (mask out future positions)
        mask = None
        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            # Broadcast to (1, 1, T, T) so it works with (B, H, T, T) scores
            mask = mask[None, None, :, :]

        attn_out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training
        )

        # Merge heads: (B, H, T, d_k) → (B, T, d_model)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.o_proj(self.drop(attn_out))

    def __repr__(self):
        return (f"MultiHeadAttention(d_model={self.d_model}, "
                f"n_heads={self.n_heads}, causal={self.causal})")


class FeedForward(Module):
    """
    Position-wise FFN used in Transformers:
      FFN(x) = GELU(x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        d_ff = d_ff or 4 * d_model
        self.fc1  = Linear(d_model, d_ff)
        self.fc2  = Linear(d_ff, d_model, scale=0.02)
        self.drop = Dropout(dropout)
        self.act  = GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TransformerBlock(Module):
    """
    Pre-norm Transformer block:
      x = x + Attn(LN(x))
      x = x + FFN(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: Optional[int] = None,
                 dropout: float = 0.1, causal: bool = False):
        self.ln1  = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, causal=causal)
        self.ln2  = LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

    def parameters(self) -> List[Tensor]:
        return (self.ln1.parameters() + self.attn.parameters() +
                self.ln2.parameters() + self.ff.parameters())

    def train(self):
        self._training = True
        for m in [self.ln1, self.attn, self.ln2, self.ff, self.drop]:
            m.train()

    def eval(self):
        self._training = False
        for m in [self.ln1, self.attn, self.ln2, self.ff, self.drop]:
            m.eval()


class Sequential(Module):
    """PyTorch-style Sequential container."""

    def __init__(self, *args: "Module | OrderedDict[str, Module]") -> None:
        self._modules: Dict[str, Module] = {}
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            od: OrderedDict[str, Module] = args[0]
            for key, module in od.items():
                self._modules[key] = module
        else:
            for i, module in enumerate(args):
                if not isinstance(module, Module):
                    raise TypeError(f"Expected Module, got {type(module)}")
                self._modules[str(i)] = module

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    @overload
    def __getitem__(self, idx: int) -> Module: ...
    @overload
    def __getitem__(self, idx: slice) -> "Sequential": ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "Sequential"]:
        modules = list(self._modules.values())
        if isinstance(idx, slice):
            return Sequential(*modules[idx])
        return modules[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        key = list(self._modules.keys())[idx]
        self._modules[key] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> "Sequential":
        idx = str(len(self._modules))
        self._modules[idx] = module
        return self

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd: Dict[str, np.ndarray] = {}
        for name, module in self._modules.items():
            for k, v in module.state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        for name, module in self._modules.items():
            prefix = f"{name}."
            sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if sub_sd:
                module.load_state_dict(sub_sd)

    def train(self) -> None:
        self._training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        self._training = False
        for module in self._modules.values():
            module.eval()

    def to(self, device: str) -> "Sequential":
        for module in self._modules.values():
            module.to(device)
        return self

    def __repr__(self) -> str:
        lines: List[str] = ["Sequential("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return "\n".join(lines)
