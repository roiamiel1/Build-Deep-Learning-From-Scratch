# Stage 33: Capstone — MNIST

**Imports** the whole `stage_32` `mytorch` API (`Module`/`Sequential`/`Linear`/`ReLU`/`CrossEntropyLoss`/`Adam`/`DataLoader`/`Dataset`/`Tensor`) via `dlfs.stage_import`; **adds** the MNIST IDX loader, preprocessing, a `Sequential(Linear, ReLU, Linear)` `784→128→10` MLP builder, and an end-to-end train/evaluate capstone script (with a synthetic-digit fallback).

**Context** — The capstone of the framework track. You point the `mytorch` framework you consolidated in
`stage_32` — `Module`/`Sequential`/`Linear`/`ReLU`/`CrossEntropyLoss`/`Adam`/`DataLoader` — at the
canonical benchmark: **real MNIST handwritten digits**. No new gradient math, and no new framework class:
the whole point is that *your* library trains a real model on real data to a target test accuracy with
zero help from PyTorch. The classifier is a plain `Sequential(Linear, ReLU, Linear)`.

**Background** — MNIST is 70k grayscale `28x28` digit images (60k train / 10k test), labels `0..9`.
An image is a point $x\in\mathbb{R}^{784}$ (flattened) fed to the MLP, which outputs logits
$z\in\mathbb{R}^{10}$ and minimizes the mean softmax cross-entropy (the `stage_32` `CrossEntropyLoss`,
wrapping `stage_13`), $L=-\frac1B\sum_b \log p_{b,y_b}$ with $p=\mathrm{softmax}(z)$. The only
equation that matters here is the one fused gradient your engine already produces,
$$\frac{\partial L}{\partial z}=\frac{1}{B}\big(\mathrm{softmax}(z)-Y\big),\qquad Y=\text{one-hot}(y),$$
and it flows back through every `Linear`/`ReLU` via a single `loss.backward()`. **Preprocessing** is the
new practical skill: read the raw IDX bytes with stdlib + NumPy, scale pixels to $[0,1]$ (divide by 255),
and reshape each image to a length-784 row for the MLP. Normalized inputs keep the `Adam` updates
well-scaled and let the initialization train stably. No autodiff library, no `torchvision`, no `sklearn`
— pixels in, your gradients out.

**Watch**
- [But what is a neural network? (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk) — MNIST framed exactly as the 784→…→10 classifier you are training here.
- [Neural Networks: Backpropagation (StatQuest)](https://www.youtube.com/watch?v=IN2XmBhILt4) — the cross-entropy + softmax gradient that drives every step, end to end.

**Exercise** — Implement, in `code.py`, an MNIST loader and a capstone training script that reaches a
target test accuracy using ONLY the `stage_32` framework. Pull the whole `mytorch` API
(`Module`/`Sequential`/`Linear`/`ReLU`/`CrossEntropyLoss`/`Adam`/`DataLoader`/`Dataset`/`Tensor`) from
`stage_32` via `dlfs.stage_import`, and expose `cross_entropy_loss` as a `CrossEntropyLoss()` instance.
Define `accuracy` locally. Allowed tools: `numpy` (array math + byte parsing), `matplotlib` (optional
curves), the stdlib (`gzip`, `struct`, `os`), and the `stage_32` framework. **No PyTorch / TensorFlow /
JAX / autograd / torchvision / sklearn.** No hand-written layer gradients — every backward comes from the
imported `stage_32` layers.

- `load_mnist_idx(images_path, labels_path) -> (X, y)`: parse the canonical IDX format (optionally
  gzipped). Validate the magic numbers (`2051` images, `2049` labels) via `struct.unpack(">II...")`;
  return `X` of shape `(N, 28, 28)` `uint8`→`float64` and `y` of shape `(N,)` `int`.
- `preprocess(X, *, flatten, normalize=True) -> np.ndarray`: divide by 255 when `normalize`; if `flatten`
  return `(N, 784)`, else return channels-first `(N, 1, 28, 28)`. Pure NumPy, no copy of label data.
- `make_loaders(X, y, *, batch_size, val_frac=0.0, flatten, seed=None) -> (train_loader, val_loader)`:
  `preprocess`, optional NumPy validation split, wrap each split in a `Dataset`+`DataLoader` (`stage_32`).
  `val_loader` is `None` when `val_frac == 0`. Labels stay integer (cross-entropy expects indices). Train
  loader shuffles; val loader does not.
- `build_mlp(seed=None) -> Sequential`: `Sequential(Linear(784, 128, seed), ReLU(), Linear(128, 10, seed))`
  — raw logits, no output activation (the loss applies softmax).
- `accuracy(logits, targets) -> float`: fraction where `argmax(logits.data, axis=1) == targets`
  (integer labels); defined locally.
- `evaluate(model, loader) -> dict`: `model.eval()`, iterate the loader, return
  `{"loss": mean_cross_entropy, "acc": mean_accuracy}` (size-weighted).
- `train_mnist(model, train_loader, *, epochs, lr=1e-3, val_loader=None, optimizer=None, verbose=False)
  -> dict`: per epoch `model.train()`, then for each `(X_b, y_b)`: `logits = model(X_b)` →
  `loss = cross_entropy_loss(logits, y_b)` → `loss.backward()` → `optimizer.step()` →
  `optimizer.zero_grad()` (build `Adam(model.parameters(), lr)` if `optimizer is None`). After each epoch,
  `evaluate` on `val_loader` if given. Return
  `{"train_loss":[...], "val_loss":[...], "val_acc":[...], "steps":int}`.
- `run_capstone(data_dir=None, *, model="mlp", epochs=..., batch_size=..., subset=None, seed=0) -> dict`:
  the end-to-end script. If `data_dir` holds the IDX files, load real MNIST (optionally a `subset` of
  train rows for speed); otherwise fall back to a small synthetic digit set so the script always runs.
  Build the MLP, train, evaluate on the held-out test split, and return the final
  `{"test_acc":..., "test_loss":..., "history":..., "model":..., "source":...}`. Print a one-line report
  when `verbose`.

**Done when**
- `pytest stage_33_capstone_mnist/test.py` passes.
- `load_mnist_idx` reconstructs `(N,28,28)` images + labels from a synthetic IDX byte buffer; bad magic
  numbers raise.
- `preprocess` yields `(N,784)` in `[0,1]` when `flatten=True` and `(N,1,28,28)` otherwise.
- End-to-end gradcheck: on one batch, each parameter's `grad` from
  `cross_entropy_loss(model(X_b), y_b).backward()` matches the central difference
  `(f(p+eps)-f(p-eps))/(2*eps)` within `atol ~ 1e-4` (loosened for the deep stack).
- `train_mnist` drives training loss down and reaches `>= 0.90` accuracy on a small digit set within a
  few epochs; `run_capstone` returns a `test_acc` above its target.
