## Noni (WIP)

A minimal tensor library with autograd flexible for building good enough deep learning models.

### Familiar API

```python
a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = Tensor([[0.5, -1.], [2., 0.]], requires_grad=True)

# Each op records its backward function
c = a * b        # op="*",  backward: dc/da = b, dc/db = a
d = c.sum()      # op="sum", backward: ones

d.backward()     # topological sort → apply each _backward in reverse

print(a.grad)    # dL/da = b.data = [[0.5, -1.], [2., 0.]]
print(b.grad)    # dL/db = a.data = [[1., 2.], [3., 4.]]
```

### Common Modules for everything

```python
from noni.nn import Linear, LayerNorm, MultiHeadAttention, CrossEntropyLoss

# A simple 2-layer MLP
W1 = Linear(784, 256)
W2 = Linear(256, 10)

x = Tensor(some_batch)
h = W1(x).relu()
logits = W2(h)

loss = CrossEntropyLoss()(logits, targets)
loss.backward()   # gradients in W1.weight.grad, W2.weight.grad etc.
```

### Module list

- **Linear** — weight + bias, Kaiming init
- **Embedding** — lookup table with scatter-add backward
- **LayerNorm** — normalize over last N dims, learned affine
- **Dropout** — inverted dropout during training
- **MultiHeadAttention** — optional causal mask for autoregressive models
- **FeedForward** — position-wise FFN with GELU
- **TransformerBlock** — pre-norm residual block (Attn + FFN)
- **CrossEntropyLoss** — numerically stable log-softmax + NLL
- **Optimizers** — SGD, Adam, AdamW, CosineAnnealingLR
