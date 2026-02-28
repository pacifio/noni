import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataclasses import dataclass
from typing import Optional

from noni import Tensor
from noni.nn import (
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    TransformerBlock,
)
from noni.optim import AdamW, CosineAnnealingLR


@dataclass
class GPTConfig:
    vocab_size:    int   = 65
    context_len:   int   = 256
    d_model:       int   = 256
    n_heads:       int   = 8
    n_layers:      int   = 4
    dropout:       float = 0.1
    d_ff:          Optional[int] = None
    # Training
    batch_size:    int   = 32
    max_iters:     int   = 3000
    lr:            float = 3e-4
    min_lr:        float = 3e-5
    warmup_iters:  int   = 200
    grad_clip:     float = 1.0
    eval_interval: int   = 200
    eval_iters:    int   = 50


class GPT(Module):
    def __init__(self, cfg: GPTConfig):
        self.cfg      = cfg
        self.tok_emb  = Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb  = Embedding(cfg.context_len, cfg.d_model)
        self.drop     = Dropout(cfg.dropout)
        self.blocks   = [
            TransformerBlock(cfg.d_model, cfg.n_heads,
                             d_ff=cfg.d_ff, dropout=cfg.dropout, causal=True)
            for _ in range(cfg.n_layers)
        ]
        self.ln_f     = LayerNorm(cfg.d_model)
        self.lm_head  = Linear(cfg.d_model, cfg.vocab_size, bias=False, scale=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos  = np.arange(T)[None, :]
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        x    = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def parameters(self):
        params = []
        for m in [self.tok_emb, self.pos_emb, self.drop, self.ln_f, self.lm_head]:
            params.extend(m.parameters())
        for b in self.blocks:
            params.extend(b.parameters())
        return params

    def train(self):
        self._training = True
        for m in [self.tok_emb, self.pos_emb, self.drop, self.ln_f, self.lm_head]:
            m.train()
        for b in self.blocks: b.train()

    def eval(self):
        self._training = False
        for m in [self.tok_emb, self.pos_emb, self.drop, self.ln_f, self.lm_head]:
            m.eval()
        for b in self.blocks: b.eval()

    @property
    def num_params(self):
        return sum(p.data.size for p in self.parameters())


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=0):
    model.eval()
    ctx = idx.copy()
    for _ in range(max_new_tokens):
        ctx_crop   = ctx[:, -model.cfg.context_len:]
        logits     = model(ctx_crop)
        last       = logits.data[0, -1, :].copy()
        last       = last / max(temperature, 1e-8)
        if top_k > 0:
            k        = min(top_k, len(last))
            topk_idx = np.argpartition(last, -k)[-k:]
            mask     = np.ones(len(last), dtype=bool)
            mask[topk_idx] = False
            last[mask] = -1e9
        last -= last.max()
        probs    = np.exp(last)
        probs   /= probs.sum()
        next_tok = int(np.random.choice(len(probs), p=probs))
        ctx      = np.concatenate([ctx, [[next_tok]]], axis=1)
    return ctx
