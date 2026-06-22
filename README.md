# Build Deep Learning From Scratch

A project-based curriculum that teaches how PyTorch works *internally* by reimplementing it from the ground up. You start with scalar backpropagation, build a reverse-mode autodiff engine, grow it into an N-dimensional `Tensor`, then stack neural-network layers, optimizers, training loops, CNNs, attention, a full Transformer, a Vision Transformer, a small but real PyTorch-like framework, and finally capstone projects. The philosophy: **you are the autodiff library.** Nothing is imported that you haven't already built by hand, so every gradient and every chain-rule accumulation is code you wrote and understand.

The scalar autodiff engine in **stages 01-05** is a from-scratch reimplementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd): `Value(data)` (stage 01) → computational graph (stage 02) → per-op `_backward` closures (stage 03) → the `backward()` reverse pass (stage 04) → the remaining ops `tanh`/`exp`/`relu` (stage 05). The same engine micrograd packs into one file, built up one concept per stage. Everything afterward generalizes it to tensors.

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

Each stage builds on the code from prior stages (e.g. `Value` from `stage_05`, `Tensor` from `stage_08`).

## The 34 stages

| # | Stage | What you build |
|---|-------|----------------|
| 01 | Scalar Values | `Value`: wrap one number; forward arithmetic via operator overloading. |
| 02 | Computational Graph | Record each result's parents (`_prev` set), op (`_op`), and a no-op `_backward` hook. |
| 03 | Local Derivatives | Install per-op `_backward` closures (the local-derivative push). |
| 04 | Chain Rule | `backward()`: topological sort + seed grad=1 + reverse-walk the closures. |
| 05 | Backprop Engine | Add `tanh`, `exp`, `relu` on the scalar `Value`; the complete micrograd engine. |
| 06 | Vector Operations | A `Vec` container of `Value`s: elementwise ops, `dot`, `sum`. |
| 07 | Matrix Operations | A `Mat` of `Value`s: `matmul`/`@`, transpose, reshape, sum, mean. |
| 08 | Tensor Engine | Collapse scalar graphs onto one N-dim NumPy-backed autodiff `Tensor`. |
| 09 | Neuron | A single learnable neuron `y = phi(x @ w + b)` on the `Tensor`. |
| 10 | Dense Layer | Vectorized fully-connected layer `Z = X @ W + b`. |
| 11 | MLP | Stack `Dense` layers with activations between them. |
| 12 | Loss Functions | MSE/MAE/cross-entropy (+ stable softmax) and `sum`/`mean` reductions. |
| 13 | SGD Optimizer | The `Optimizer`/`SGD` update step abstraction. |
| 14 | First Training Loop | Wire MLP + loss + SGD into the canonical learn loop. |
| 15 | Weight Initialization | Xavier/Glorot and He/Kaiming init, and why scale matters. |
| 16 | Momentum | SGD with momentum. |
| 17 | Adam | RMSProp/Adam with bias correction and weight decay. |
| 18 | Batch Training | Minibatching, epochs, shuffling; gradient-variance intuition. |
| 19 | DataLoader | `Dataset`/`DataLoader` batching abstraction. |
| 20 | Regularization | L2 / weight decay; train vs eval mode. |
| 21 | Dropout | Dropout forward + backward; inverted scaling. |
| 22 | BatchNorm | Batch normalization forward + backward by hand. |
| 23 | Conv2D Math | Convolution arithmetic and gradients via im2col. |
| 24 | Conv2D Implementation | `Conv2D`/pooling/flatten as `Tensor` layers. |
| 25 | CNN Project | Stack conv/pool/linear; train on image data. |
| 26 | Attention Math | Scaled dot-product attention forward + backward (pure NumPy). |
| 27 | Self-Attention | Self-attention on the `Tensor` autodiff engine. |
| 28 | Multi-Head Attention | Split/concat heads; the full MHA module. |
| 29 | Transformer | Residuals + LayerNorm + FFN; a full Transformer block. |
| 30 | Vision Transformer | Patch embeddings + Transformer for image classification. |
| 31 | Framework Refactor | Package it all into a clean PyTorch-like `Module`/`Parameter` API. |
| 32 | Capstone: MNIST | End-to-end MNIST classifier. |
| 33 | Capstone: CIFAR-10 | End-to-end CIFAR-10 classifier. |
| 34 | Capstone: Transformer | End-to-end Transformer language model. |

## Expected effort

Roughly **150-250 hours** end to end, depending on background.
