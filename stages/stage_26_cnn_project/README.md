# Stage 26: CNN classifier

**Imports** `Tensor` (stage_09), `Dense` (stage_11), `cross_entropy_loss` (stage_13), `Adam` (stage_18), `DataLoader`/`Dataset` (stage_20), `Conv2D`/`MaxPool2D`/`Flatten` (stage_25) — **adds** a `CNN` that composes the conv tower + dense head and a `train_cnn` train/eval loop (plus `accuracy` and a synthetic `make_digit_blobs` set). No new gradient math.

**Context** — The first end-to-end *project* stage: you assemble every building block of the framework into a real convolutional image classifier and train it on MNIST-like digit data. There is no new gradient math here — the heavy lifting (the `Conv2D`, `MaxPool2D` and `Flatten` backward from `stage_25`) is already done. Your job is to *compose* a `LeNet`-style net (conv→relu→pool, conv→relu→pool, flatten, dense head) on the autodiff `Tensor` (`stage_09`), feed it batches from the `DataLoader` (`stage_20`), and drive a full train/eval loop to a target accuracy.

**Background** — A CNN for classification is just the same chain rule you have used since `stage_06`, applied to layers that share weights spatially. A 2-D conv layer (`Conv2D`, `stage_25`) maps an input image stack $X\in\mathbb{R}^{B\times C_{in}\times H\times W}$ to feature maps $Y\in\mathbb{R}^{B\times C_{out}\times H'\times W'}$; a pooling layer (`MaxPool2D`, `stage_25`) downsamples each map, shrinking $H',W'$ and giving translation tolerance. After the conv/pool tower you **flatten** each example's $(C,H,W)$ maps into a vector (the `Flatten` layer, also from `stage_25`) and apply the `Dense` head (`stage_11`) to produce $C$ class logits, then the stable softmax cross-entropy from `stage_13`. The forward stack is
$$\text{logits}=\text{Dense}\big(\text{flatten}(\,\text{pool}(\phi(\text{conv}_2(\text{pool}(\phi(\text{conv}_1(X))))))\,)\big),\quad \phi=\text{ReLU},$$
and the loss is the mean cross-entropy $L=-\frac1B\sum_{i}\log p_{i,y_i}$ with $p=\mathrm{softmax}(\text{logits})$. Every layer's `_backward` is reused verbatim, so the whole network differentiates by one call to `Tensor.backward()` — the chain rule composes the per-layer Jacobians automatically. The `Flatten` reshape (its backward routes the incoming gradient back to $(B,C,H,W)$) is already built in `stage_25`, so this stage adds **no** new gradient code. Optimization uses `Adam` (`stage_18`); `train()`/`eval()` mode toggles propagate to every sub-layer that defines them.

**Watch**
- [But what is a convolution? (3Blue1Brown)](https://www.youtube.com/watch?v=KuXjwB4LzSA) — the convolution operation itself, the core primitive your tower stacks.
- [MIT 6.S191: Convolutional Neural Networks](https://www.youtube.com/watch?v=NmLK_WQBxB4) — why conv→relu→pool→dense is the canonical classifier architecture you are assembling.

**Exercise** — Implement a small CNN image classifier in `code.py` by composing prior stages. Import via `dlfs.stage_import`, binding each prior symbol to a `StageN_Symbol` alias, e.g. `Stage9_Tensor = stage_import("stage_09", "Tensor")`, `Stage11_Dense = stage_import("stage_11", "Dense")`, `Stage25_Conv2D, Stage25_MaxPool2D, Stage25_Flatten = stage_import("stage_25", "Conv2D", "MaxPool2D", "Flatten")`, `Stage13_cross_entropy_loss = stage_import("stage_13", "cross_entropy_loss")`, `Stage18_Adam = stage_import("stage_18", "Adam")`, and `DataLoader`/`Dataset` (`stage_20`) in the tests. Re-export the canonical public names (`Tensor`, `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, `cross_entropy_loss`, `Adam`). Allowed tools: `numpy` (forward array math only), `matplotlib` (optional curves), the stdlib, and your prior-stage code. No PyTorch / autograd. No hand-written layer gradients — every backward comes from the imported layers.

- `Flatten` is **imported** from `stage_25` (the parameter-free reshape layer), not redefined here.
- `CNN(in_shape, n_classes, *, conv_channels=(8, 16), kernel_size=3, hidden=64, seed=None)`: a `Module`-style classifier.
  - `in_shape` is `(C, H, W)`. Build, in order: `Conv2D(conv_channels[0], C, kernel_size, padding=...)` → ReLU → `MaxPool2D(2)` → `Conv2D(conv_channels[1], conv_channels[0], kernel_size, padding=...)` → ReLU → `MaxPool2D(2)` → `Flatten()` → `Dense(flat_dim, hidden)` → ReLU → `Dense(hidden, n_classes)`. (Note stage_25 `Conv2D` takes `out_channels` first.) Compute `flat_dim` from the post-pool spatial size (derive it; do **not** hardcode for one input size).
  - `forward(x) -> Tensor`: coerce `x` to a `Tensor` of shape `(B, C, H, W)`, run the stack, return logits of shape `(B, n_classes)`. ReLU via `Tensor.relu()`.
  - `parameters() -> list[Tensor]`: flatten the parameters of every sub-layer in order.
  - `train()` / `eval()`: set `self.training` and propagate to every sub-layer; return `self` (chainable).
  - `zero_grad()`: zero every parameter's grad.
- `accuracy(logits, targets) -> float`: fraction where `argmax(logits.data, axis=1) == targets` (integer labels).
- `train_cnn(model, train_loader, *, epochs, lr=1e-3, val_loader=None, optimizer=None, verbose=False) -> dict`: build `Adam(model.parameters(), lr)` if `optimizer is None`. Per epoch, `model.train()`, then for `X_b, y_b in train_loader`: `logits = model(X_b)` → `loss = cross_entropy_loss(logits, y_b)` → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`. If `val_loader`, `model.eval()` and compute mean val loss + accuracy over it. Return `{"train_loss": [...], "val_loss": [...], "val_acc": [...], "steps": int}`.

**Done when**
- `pytest stage_26_cnn_project/test.py` passes.
- `Flatten` round-trips shape: `(B, C, H, W) -> (B, C*H*W)`, and central-difference gradcheck on `flatten` gives `dL/dx` matching `(f(x+eps)-f(x-eps))/(2*eps)` within `atol ~ 1e-6` (a reshape, so the gradient is identity up to layout).
- `CNN.forward` returns `(B, n_classes)` logits for any valid `in_shape`; `flat_dim` is derived, not hardcoded.
- End-to-end gradcheck: for one batch, every parameter's `grad` from `cross_entropy_loss(model(X_b), y_b).backward()` matches the central-difference estimate within `atol ~ 1e-4` (loosened for the deep stack).
- On a small synthetic 2-class digit set, `train_cnn` drives training loss down and reaches `>= 0.90` train accuracy within a few epochs.
