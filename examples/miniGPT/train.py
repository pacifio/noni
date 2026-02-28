import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from model import GPT, GPTConfig, generate

from noni.nn import CrossEntropyLoss
from noni.optim import AdamW, CosineAnnealingLR
from noni.tensor import no_grad

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/shakespeare.txt")
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")

def download_shakespeare():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading tiny_shakespeare …")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"  saved to {DATA_PATH}")
    return open(DATA_PATH, encoding="utf-8").read()


def build_vocab(text):
    chars   = sorted(set(text))
    stoi    = {c: i for i, c in enumerate(chars)}
    itos    = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)


def get_batch(data, batch_size, context_len):
    ix = np.random.randint(0, len(data) - context_len, size=batch_size)
    x  = np.stack([data[i:i+context_len]   for i in ix])
    y  = np.stack([data[i+1:i+context_len+1] for i in ix])
    return x, y


def estimate_loss(model, train_data, val_data, cfg, loss_fn):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg.batch_size, cfg.context_len)
            with no_grad():
                logits = model(xb)
                loss   = loss_fn(logits, yb)
            losses.append(loss.data.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def train():
    np.random.seed(1337)

    text  = download_shakespeare()
    stoi, itos, vocab_size = build_vocab(text)
    data  = encode(text, stoi)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]

    print(f"Dataset: {len(text):,} chars, vocab_size={vocab_size}")
    print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

    cfg = GPTConfig(
        vocab_size   = vocab_size,
        context_len  = 256,
        d_model      = 256,
        n_heads      = 8,
        n_layers     = 4,
        dropout      = 0.1,
        batch_size   = 32,
        max_iters    = 3000,
        lr           = 3e-4,
        min_lr       = 3e-5,
        warmup_iters = 200,
        grad_clip    = 1.0,
        eval_interval= 200,
        eval_iters   = 50,
    )

    model  = GPT(cfg)
    loss_fn = CrossEntropyLoss()

    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                weight_decay=0.1)
    sched = CosineAnnealingLR(opt, T_max=cfg.max_iters,
                               min_lr=cfg.min_lr,
                               warmup_steps=cfg.warmup_iters)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val  = float("inf")
    t0        = time.time()

    print(f"\n{'='*60}")
    print("  Training miniGPT on Shakespeare")
    print(f"  {cfg.max_iters} iterations  batch={cfg.batch_size}  ctx={cfg.context_len}")
    print(f"{'='*60}\n")

    for step in range(cfg.max_iters + 1):

        # Evaluation
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, loss_fn)
            dt     = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {opt.lr:.2e} | {dt:.1f}s")

            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {k: v.copy() for k, v in _collect_state(model).items()}
                np.savez(os.path.join(CKPT_DIR, "best.npz"), **ckpt)

            # Sample text
            if step > 0:
                with no_grad():
                    seed = np.array([[stoi["\n"]]])
                    out  = generate(model, seed, max_new_tokens=200,
                                    temperature=0.8, top_k=20)
                sample = "".join(itos[i] for i in out[0])
                print(f"\n--- Sample ---\n{sample.strip()}\n{'-'*40}\n")

            t0 = time.time()

        if step == cfg.max_iters:
            break

        # Training step
        model.train()
        xb, yb = get_batch(train_data, cfg.batch_size, cfg.context_len)
        logits  = model(xb)
        loss    = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm_(cfg.grad_clip)
        opt.step()
        sched.step()

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved to {CKPT_DIR}/best.npz")

    # Save vocab
    with open(os.path.join(CKPT_DIR, "vocab.json"), "w") as f:
        json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f)

    return model, stoi, itos


def _collect_state(model):
    """Flat dict of all param arrays for numpy save."""
    state = {}
    for i, p in enumerate(model.parameters()):
        state[f"param_{i}"] = p.data
    return state


if __name__ == "__main__":
    train()
