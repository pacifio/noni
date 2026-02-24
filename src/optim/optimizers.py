import math
from typing import List

import numpy as np

from ..tensor import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def clip_grad_norm_(self, max_norm: float) -> float:
        """Clip gradients by global L2 norm. Returns the pre-clip norm."""
        total_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                total_norm += (p.grad ** 2).sum()
        total_norm = math.sqrt(total_norm)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for p in self.params:
                if p.grad is not None:
                    p.grad = p.grad * scale
        return total_norm

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.
    v_t = momentum * v_{t-1} + (1 - momentum) * grad
    w_t = w_{t-1} - lr * v_t
    """

    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.velocity     = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0.0:
                v[:] = self.momentum * v + (1 - self.momentum) * g
                g = v
            p.data -= self.lr * g


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015).

    m_t = β1 * m_{t-1} + (1 - β1) * g
    v_t = β2 * v_{t-1} + (1 - β2) * g²
    m̂ = m_t / (1 - β1^t)   (bias correction)
    v̂ = v_t / (1 - β2^t)
    w  = w - lr * m̂ / (sqrt(v̂) + ε)
    """

    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * (g ** 2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW (Loshchilov & Hutter, 2019).

    Decoupled weight decay: the L2 penalty is applied directly to weights
    BEFORE the adaptive update, rather than folding it into the gradient.
    This is the standard optimizer for training transformers.
    """

    def __init__(self, params: List[Tensor], lr: float = 3e-4,
                 betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad.astype(np.float32)

            # Decoupled weight decay (applied directly to param, not grad)
            if self.weight_decay != 0.0:
                p.data *= (1.0 - self.lr * self.weight_decay)

            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * (g ** 2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CosineAnnealingLR:
    """
    Cosine annealing from max_lr → min_lr over T_max steps.
    Used by Karpathy's nanoGPT — very effective for LLM training.
    """

    def __init__(self, optimizer: Optimizer,
                 T_max: int,
                 min_lr: float = 0.0,
                 warmup_steps: int = 0):
        self.opt          = optimizer
        self.T_max        = T_max
        self.min_lr       = min_lr
        self.max_lr       = optimizer.lr
        self.warmup_steps = warmup_steps
        self._step        = 0

    def step(self):
        self._step += 1
        s = self._step
        if s <= self.warmup_steps:
            lr = self.max_lr * s / max(self.warmup_steps, 1)
        else:
            progress = (s - self.warmup_steps) / max(self.T_max - self.warmup_steps, 1)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.opt.lr = lr
        return lr
