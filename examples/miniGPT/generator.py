import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from model import GPT, GPTConfig, generate
from noni.tensor import no_grad

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_model():
    with open(os.path.join(CKPT_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}

    cfg = GPTConfig(vocab_size=len(stoi))
    model = GPT(cfg)
    model.eval()

    state = np.load(os.path.join(CKPT_DIR, "best.npz"))
    for i, p in enumerate(model.parameters()):
        p.data = state[f"param_{i}"]

    return model, stoi, itos


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generator.py <num_tokens>")
        sys.exit(1)

    n_tokens = int(sys.argv[1])

    model, stoi, itos = load_model()

    seed = np.array([[stoi["\n"]]])
    with no_grad():
        out = generate(model, seed, max_new_tokens=n_tokens, temperature=0.8, top_k=40)

    print("".join(itos[i] for i in out[0]))
