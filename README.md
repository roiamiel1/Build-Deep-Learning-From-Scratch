# Build Deep Learning From Scratch

A project-based curriculum that teaches how PyTorch works *internally* by reimplementing it from the ground up. You start with scalar backpropagation, build a reverse-mode autodiff engine, grow it into an N-dimensional `Tensor`, then stack neural-network layers, optimizers, training loops, CNNs, attention, a full Transformer, a Vision Transformer, a small but real PyTorch-like framework, and finally capstone projects. The philosophy: **you are the autodiff library.** Nothing is imported that you haven't already built by hand, so every gradient and every chain-rule accumulation is code you wrote and understand.

## Tool restriction

The only permitted packages are **NumPy** (forward array math only — never to compute a derivative for you), **Matplotlib** (visualization), and **pytest** (tests). No `torch`, `tensorflow`, `jax`, `autograd`, `tinygrad`, `micrograd`, or any library that does autodiff/backprop for you — until you've built that concept by hand in an earlier stage. See `requirements.txt`.

## How a stage works

Each `stage_xx/` directory contains exactly three files:

- **`README.md`** — context, background (intuition + key equations/gradients), watch (videos), and a zero-ambiguity exercise.
- **`code.py`** — skeleton only: interfaces, signatures, docstrings, and TODOs. No working bodies.
- **`test.py`** — pytest tests, including numerical gradient checks (central differences) wherever gradients exist.

Implement `code.py` until the tests pass:

```bash
pytest stage_xx/test.py
```

Each stage builds on the code from prior stages (e.g. `Value` from `stage_06`, `Tensor` from `stage_11`).

## The 35 stages

| # | Stage | What you build |
|---|-------|----------------|
| 01 | Numerical Derivatives | Finite-difference gradients; the limit definition of a derivative. |
| 02 | The Chain Rule | Compose derivatives by hand; manual forward/backward on small functions. |
| 03 | Computational Graphs | Represent expressions as DAGs of operations. |
| 04 | Manual Backprop | Hand-code reverse passes through a fixed graph. |
| 05 | Topological Sort | Order nodes for correct gradient accumulation. |
| 06 | The `Value` Scalar Engine | Autodiff over scalars with `+`, `*`, `tanh`, `.backward()`. |
| 07 | More Scalar Ops | `exp`, `log`, `pow`, division, ReLU, broadcasting of scalars. |
| 08 | A Scalar Neuron & MLP | Wire `Value`s into neurons, layers, and a tiny MLP. |
| 09 | Loss & Manual SGD | MSE/cross-entropy on `Value`; hand-rolled gradient descent. |
| 10 | Vectorizing with NumPy | Move from scalars to arrays; why scalar graphs don't scale. |
| 11 | The `Tensor` Engine | N-dim autodiff tensor with grad tracking and `.backward()`. |
| 12 | Broadcasting Backward | Correct gradient reduction across broadcast dimensions. |
| 13 | Matmul & Reductions | `@`, `sum`, `mean`, `max` with their backward rules. |
| 14 | Activations | ReLU, Sigmoid, Tanh, GELU as autodiff ops. |
| 15 | Linear Layer & `Module` | `Parameter`, `Module` base class, `nn.Linear`. |
| 16 | Softmax & Cross-Entropy | Numerically stable softmax + its fused gradient. |
| 17 | Training Loop | Forward, backward, zero-grad, step on a real dataset. |
| 18 | Optimizers | SGD+momentum, RMSProp, Adam, weight decay. |
| 19 | Initialization | Xavier/Glorot, He init, and why scale matters. |
| 20 | Regularization | Dropout and L2; train vs eval mode. |
| 21 | BatchNorm | Batch normalization forward + backward by hand. |
| 22 | LayerNorm | Layer normalization and its gradients. |
| 23 | Convolution | `Conv2d` via im2col with full backward. |
| 24 | Pooling | Max/average pooling forward and backward. |
| 25 | A CNN Classifier | Stack conv/pool/linear; train on image data. |
| 26 | Embeddings & Tokenizer | Embedding lookup + gradients; a byte/char tokenizer. |
| 27 | RNN | Vanilla recurrent cell and backprop-through-time. |
| 28 | Self-Attention | Scaled dot-product attention forward + backward. |
| 29 | Multi-Head Attention | Split/concat heads; the full MHA module. |
| 30 | Positional Encoding | Sinusoidal and learned position embeddings. |
| 31 | Transformer Block | Residuals, LayerNorm, FFN; a full encoder/decoder block. |
| 32 | A GPT-style Transformer | Causal masking; train a small char-level language model. |
| 33 | Vision Transformer | Patch embeddings + Transformer for image classification. |
| 34 | The Mini-Framework | Package it all into a clean PyTorch-like API. |
| 35 | Capstones | End-to-end projects: GPT, ViT, or your own architecture. |

## Expected effort

Roughly **150-250 hours** end to end, depending on background.
