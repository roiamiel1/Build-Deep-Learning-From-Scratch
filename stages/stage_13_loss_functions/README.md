# Stage 13: Loss functions

**Context** — A model produces predictions; a *loss* turns the gap between predictions and targets
into a single scalar to descend. This stage builds the three workhorse losses — **MSE**, **MAE**, and
**Cross-Entropy** (with a numerically stable **softmax** via **log-sum-exp**) — as functions over the
`Tensor` from `stage_09`, relying on the broadcast-correct backward from `stage_12`. These are the
objectives every training loop in later stages minimizes.

**Background** — A loss is a function $L(\hat y, y) \to \mathbb{R}$ reduced over the batch (mean here).
Builds on the autodiff `Tensor` from `stage_09` and the broadcasting gradient reduction from
`stage_12` (so `(B, C)` minus `(B, 1)` backprops correctly).

*MSE* (regression): $L = \frac{1}{N}\sum_i (\hat y_i - y_i)^2$, so $\dfrac{\partial L}{\partial \hat y_i} = \dfrac{2}{N}(\hat y_i - y_i)$.

*MAE* (regression, robust): $L = \frac{1}{N}\sum_i |\hat y_i - y_i|$, so $\dfrac{\partial L}{\partial \hat y_i} = \dfrac{1}{N}\,\mathrm{sign}(\hat y_i - y_i)$ (subgradient $0$ at the kink).

*Softmax* maps logits $z \in \mathbb{R}^{C}$ to a distribution $p_c = \dfrac{e^{z_c}}{\sum_k e^{z_k}}$.
Computing $e^{z}$ directly overflows; subtract the row max $m=\max_k z_k$ (it cancels):
$$p_c = \frac{e^{z_c - m}}{\sum_k e^{z_k - m}}, \qquad \log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}.$$
This last identity is **log-sum-exp**, the stable way to get the normalizer.

*Cross-Entropy* with one-hot target $y$ (true class $t$): $L = -\sum_c y_c \log p_c = -\log p_t$. Using
log-sum-exp, the per-example loss is $L = \mathrm{LSE}(z) - z_t = \big(m + \log\sum_k e^{z_k-m}\big) - z_t$.
Its gradient w.r.t. the logits is the clean, famous result
$$\frac{\partial L}{\partial z_c} = p_c - y_c,$$
averaged over the batch. Deriving this (softmax Jacobian + the $\log$) is the point of the stage.

**Watch**
- [Neural Networks Part 5: ArgMax and SoftMax](https://www.youtube.com/watch?v=KpKog-L9veg) — StatQuest: what softmax computes and why.
- [The spelled-out intro to neural networks (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy; the cross-entropy + softmax loss segment shows the `p - y` gradient in code.

**Cumulative build** — Imports `Tensor` from `stage_09` via `dlfs.stage_import` (reused unchanged, with stage_12's broadcast-correct backward); ADDS the loss functions `mse_loss`, `mae_loss`, `cross_entropy_loss` (+ `softmax`/`log_softmax` via log-sum-exp, and the `one_hot` helper) on top of it.

**Exercise** — In `code.py`, implement these as **functions** that take and return `Tensor`s from
`stage_09` (import it via `dlfs.stage_import`; do not reimplement it). Build every loss out of `Tensor` ops so gradients flow
through `Tensor.backward()` — do **not** hand-write any `.grad`. Allowed tools: `numpy` (forward array
creation only), Python stdlib, and your `stage_09` `Tensor` (+ `stage_12` broadcasting). No
PyTorch/autograd.

- `mse_loss(pred, target) -> Tensor`: mean of $(\hat y - y)^2$ over all elements; returns a scalar
  `Tensor`. `pred`/`target` are `Tensor`/array-like of equal shape `(N,)` or `(B, D)`.
- `mae_loss(pred, target) -> Tensor`: mean of $|\hat y - y|$; scalar `Tensor`. Implement `abs` via
  `Tensor` ops or an `abs`/`sign`-based op already in your engine (no autodiff helper).
- `log_softmax(logits) -> Tensor`: input `(B, C)` logits; return `(B, C)` of $\log p$ using the
  stable log-sum-exp (subtract per-row max, which must be treated as a constant — no grad through the
  max shift). Equals `logits - logsumexp(logits, axis=1, keepdims=True)`.
- `softmax(logits) -> Tensor`: `(B, C)` probabilities; rows sum to 1. May be `exp(log_softmax(...))`.
- `cross_entropy_loss(logits, targets) -> Tensor`: `logits` `(B, C)`; `targets` either a 1-D array of
  `B` integer class indices **or** a `(B, C)` one-hot `Tensor`/array. Return the scalar mean
  $-\frac{1}{B}\sum_b \log p_{b,t_b}$. Build it from `log_softmax` so the gradient is `(p - y)/B`.
- All reductions use **mean** over the batch. Inputs are coerced to `Tensor` if needed.

**Done when**
- `pytest stage_13_loss_functions/test.py` passes.
- Forward values match NumPy reference losses; softmax rows sum to 1; cross-entropy is stable for
  logits like `[1000, 1001, 1002]` (no `inf`/`nan`).
- Central-difference gradcheck of each loss w.r.t. its prediction/logits matches the analytic
  `Tensor.grad` within `atol ~ 1e-6` (MAE checked away from the kink).
- Cross-entropy gradient w.r.t. logits equals `(softmax(logits) - onehot(targets)) / B`.
