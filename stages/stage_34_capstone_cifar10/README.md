# Stage 34: Capstone — CIFAR-10

**Imports** the stage_32 `mytorch` API (`Tensor`/`cross_entropy_loss`/`Adam`/`DataLoader`) + `Conv2D`/`MaxPool2D`/`Flatten` (stage_25) + `Dense` (stage_11) via `dlfs.stage_import`; **adds** `BatchNorm2d`, augmentation, LR schedules, `ConvNet`, and the CIFAR-10 load + train script.

**Context** — The final image capstone: train a real convolutional classifier on **CIFAR-10** (32x32 RGB, 10 classes) using *only* `mytorch` — the engine and layers you built in stages 09-26 plus the training utilities from `stage_32`. No new autodiff primitives; the one new gradient is `BatchNorm2d`, the 2-D sibling of the `BatchNorm1d` you derived in `stage_23`. You add data augmentation, batch normalization, and a learning-rate schedule, then push past a target accuracy.

**Background** — A deeper net needs three things the shallow CNN of `stage_26` lacked. (1) **Batch norm** stabilizes training by standardizing each channel across the batch *and* spatial positions, then rescaling. For a 4-D activation $x\in\mathbb{R}^{N\times C\times H\times W}$, `BatchNorm2d` reduces over the $M=N\cdot H\cdot W$ positions of channel $c$: $\mu_c=\frac1M\sum x,\;\sigma_c^2=\frac1M\sum(x-\mu_c)^2,\;\hat x=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\;y=\gamma_c\hat x+\beta_c$. Its input gradient is the same collapsed form you derived in `stage_23`, with the sum now over all $M$ positions of the channel:
$$\frac{\partial L}{\partial x}=\frac{1}{M\,\sqrt{\sigma^2+\epsilon}}\Big(M\,g_{\hat x}-\textstyle\sum g_{\hat x}-\hat x\textstyle\sum(g_{\hat x}\hat x)\Big),\quad g_{\hat x}=g\,\gamma .$$
(2) **Augmentation** (random crop + horizontal flip + per-channel normalization) is applied to the *raw arrays before* they enter the graph, so it carries no gradient — it just multiplies the effective dataset size. (3) A **cosine LR schedule** $\eta_t=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$ anneals the step size for a cleaner final descent. Everything else — `Conv2D`/`MaxPool2D`/`Flatten` (`stage_25`), `Dense` (`stage_11`), and the `cross_entropy_loss` / `Adam` / `DataLoader` plumbing pulled through the `stage_32` `mytorch` API (re-exporting `stage_13` / `stage_18` / `stage_20`) — is reused verbatim and differentiates with one `Tensor.backward()` (`stage_09`).

**Watch**
- [CNNs / CIFAR-style classifiers (MIT 6.S191)](https://www.youtube.com/watch?v=NmLK_WQBxB4) — the conv→BN→relu→pool→dense architecture you are scaling up here.
- [But what is a convolution? (3Blue1Brown)](https://www.youtube.com/watch?v=KuXjwB4LzSA) — the primitive your tower stacks, for intuition.

**Exercise** — Implement a CIFAR-10 capstone in `code.py`, composing prior stages via `dlfs.stage_import`. Allowed tools: `numpy` (forward array math / storage only), `matplotlib` (optional curves), the stdlib, and your prior-stage code. **No PyTorch / TensorFlow / JAX / autograd.** No hand-written layer gradients except the one new BN backward.

- `BatchNorm2d(num_features, *, eps=1e-5, momentum=0.1)`: per-channel BN over `(N, C, H, W)`. `gamma`/`beta` are learnable `Tensor`s of shape `(C,)`; `running_mean`/`running_var` are non-learnable buffers (NumPy). `__call__(x) -> Tensor` returns a `(N, C, H, W)` Tensor whose `_backward` populates `x.grad`, `gamma.grad`, `beta.grad` using the collapsed formula above (TRAIN uses batch stats + updates buffers; EVAL uses buffers, no update). `parameters() -> [gamma, beta]`; `train()`/`eval()` toggle `self.training` and return `self`.
- Augmentation (NumPy on `(N, C, H, W)` float arrays, gradient-free): `random_crop(x, *, pad, rng)` zero-pads by `pad` then crops a random `H×W` window per image; `random_horizontal_flip(x, *, p=0.5, rng)` flips the W axis per image with prob `p`; `normalize(x, mean, std)` standardizes per channel; `Augment(*, pad=4, flip_p=0.5, mean, std, seed=None)` is a callable pipeline (`train`/`eval` mode: skip crop+flip in eval, always normalize).
- Schedule: `cosine_lr(step, total_steps, *, base_lr, min_lr=0.0) -> float` and `step_lr(step, *, base_lr, drop_every, gamma=0.1) -> float`.
- `ConvNet(in_shape=(3,32,32), n_classes=10, *, channels=(32,64,128), hidden=128, seed=None)`: blocks of `Conv2D(pad=1)→BatchNorm2d→relu→Conv2D(pad=1)→BatchNorm2d→relu→MaxPool2D(2)` per stage in `channels`, then `Flatten→Dense(flat_dim, hidden)→relu→Dense(hidden, n_classes)`. `flat_dim` is **derived** from the post-pool spatial size, never hardcoded. `forward`, `parameters`, `train`/`eval`, `zero_grad` as in `stage_26`.
- `accuracy(logits, targets) -> float` (reuse the `stage_26` definition).
- `make_cifar_like(n_per_class, *, img_size=32, n_classes=10, noise, seed) -> (X, y)`: synthetic 3-channel `(N,3,H,W)` set (per-class colored templates) for the convergence smoke test.
- `train_cifar(model, train_loader, *, epochs, base_lr=1e-3, schedule="cosine", augment=None, val_loader=None, optimizer=None, verbose=False) -> dict`: per step set `optimizer.lr` from the schedule, optionally apply `augment` to each batch's `X.data` before the forward pass, then `forward→cross_entropy_loss→backward→step→zero_grad`. Return `{"train_loss":[...], "val_loss":[...], "val_acc":[...], "lr":[...], "steps":int}`.

**Done when**
- `pytest stage_34_capstone_cifar10/test.py` passes.
- `BatchNorm2d` central-difference gradcheck (`dL/dx`, `dL/dgamma`, `dL/dbeta`) matches within `atol ~ 1e-5`, and TRAIN vs EVAL forward differ as specified.
- End-to-end gradcheck through `ConvNet` matches numeric slopes within `atol ~ 1e-4` (loosened for the deep BN stack).
- Augmentation preserves shape, is gradient-free, and is identity in `eval` mode (modulo normalization); schedules are monotone/correct at boundaries.
- `flat_dim` is derived; `ConvNet.forward` returns `(B, n_classes)` for any valid `in_shape`.
- On the synthetic CIFAR-like set, `train_cifar` drives loss down and reaches the target train accuracy within a few epochs.
