# Stage 21: L1 & L2 regularization

**Context** — Models with many parameters overfit: they drive the *training* loss to zero by memorizing noise, then generalize badly. **Regularization** fights this by adding a penalty on the weights to the loss, biasing the optimizer toward simpler (smaller-norm) solutions. This stage builds the two classic penalties — **L2 (ridge / weight decay)** and **L1 (lasso, sparsity-inducing)** — as `Tensor` functions on top of the losses from `stage_13`, and connects L2 to the **weight-decay** term you already wrote into the optimizers in `stage_18`.

**Background** — You minimize a *regularized* objective $\tilde L(\theta)=L(\theta)+\lambda R(\theta)$, where $L$ is a `stage_13` loss (e.g. `mse_loss`/`cross_entropy_loss`), $\lambda\ge 0$ is the penalty strength, and $R$ penalizes large weights. Builds directly on the `Tensor` engine (`stage_09`/`stage_12`) — the penalty is just more graph ops, so `Tensor.backward()` derives its gradient automatically.

*L2 / ridge:* $R_2(\theta)=\tfrac12\sum_j \theta_j^2 = \tfrac12\lVert\theta\rVert_2^2$, with gradient
$$\frac{\partial R_2}{\partial \theta_j}=\theta_j,\qquad \frac{\partial(\lambda R_2)}{\partial\theta_j}=\lambda\,\theta_j.$$
So the gradient step becomes $\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)=(1-\eta\lambda)\theta-\eta\nabla L$ — every step **shrinks** $\theta$ toward 0 by a factor $(1-\eta\lambda)$. That is *exactly* the `weight_decay` term `g = p.grad + weight_decay * p.data` from `stage_18`: **L2 penalty in the loss ≡ coupled weight decay in the optimizer** (with $\lambda=$ `weight_decay`). The $\tfrac12$ makes the gradient clean.

*L1 / lasso:* $R_1(\theta)=\sum_j |\theta_j|=\lVert\theta\rVert_1$, with sub-gradient
$$\frac{\partial R_1}{\partial \theta_j}=\operatorname{sign}(\theta_j)\quad(\text{undefined at }0,\ \text{take }0).$$
L1 pushes each weight toward 0 by a *constant* amount $\eta\lambda$ regardless of magnitude, so small weights hit exactly 0 — L1 yields **sparse** weights (feature selection); L2 only shrinks them smoothly without zeroing.

*Decoupled note:* folding $\lambda\theta$ into the gradient means adaptive optimizers (Adam, `stage_18`) divide it by $\sqrt{\hat v}$, so the effective decay varies per coordinate. **Decoupled** weight decay (AdamW, `stage_18`) instead applies $\theta\leftarrow\theta-\eta\lambda\theta$ separately — equivalent to plain L2 only for SGD, not for Adam.

**Watch**
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — StatQuest; what the L2 penalty does to the fit, clearly.
- [Regularization Part 2: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) — StatQuest; why L1 drives weights to exactly 0 (sparsity) and how it differs from L2.

**Exercise** — In `code.py`, implement the penalties as **functions over `Tensor`s** from `stage_09` (import it; do not reimplement). Build each penalty from `Tensor` ops so its gradient flows through `Tensor.backward()` — do **not** hand-write any `.grad`. Allowed tools: `numpy` (forward array math only, e.g. reading `p.data` for the *coupled-decay equivalence* helper), Python stdlib, your `stage_09` `Tensor`, and the `stage_13` losses / `stage_18` optimizers for the integration helpers. No PyTorch/autograd.

- `l2_penalty(params, lam=1.0) -> Tensor`: given an iterable of weight `Tensor`s, return the scalar `Tensor` $\lambda\cdot\tfrac12\sum_j\theta_j^2$ summed over **all** elements of **all** params. Built from `Tensor` ops (`**2`, sum, scale). `lam` may be 0 (returns a 0 `Tensor`).
- `l1_penalty(params, lam=1.0) -> Tensor`: return the scalar `Tensor` $\lambda\sum_j|\theta_j|$ over all params. Implement `abs` via `Tensor` ops only (e.g. `relu(t) + relu(-t)`, as in `stage_13`'s `mae_loss`) — no NumPy `abs` on the graph, no hand-written grad.
- `regularized_loss(loss, params, *, l1=0.0, l2=0.0) -> Tensor`: take an already-computed scalar data `loss` `Tensor` (from a `stage_13` loss) and return `loss + l1_penalty(params, l1) + l2_penalty(params, l2)` as a scalar `Tensor`. With `l1=l2=0` it must return a `Tensor` equal to `loss`.
- `l2_grad_equals_weight_decay(params, lam) -> np.ndarray | list`: a **verification helper** (no autodiff): for each param return the analytic L2 penalty gradient `lam * p.data` — the array that `stage_18`'s coupled `weight_decay` adds into `g`. Used by tests to show the equivalence.
- Penalties act on **weights only** (the caller chooses which `Tensor`s to pass — typically exclude biases); your functions just sum over whatever iterable they receive.

**Done when**
- `pytest stage_21_regularization/test.py` passes.
- Forward values match NumPy references: `l2 == lam*0.5*sum(w**2)`, `l1 == lam*sum(|w|)`.
- Central-difference gradcheck of `l2_penalty`/`l1_penalty` w.r.t. each param matches the analytic `Tensor.grad` within `atol ~ 1e-6` (L1 checked away from 0; grad equals `lam*sign(w)`).
- The L2 penalty gradient equals `lam * p.data`, i.e. matches `stage_18`'s coupled `weight_decay` term (`l2_grad_equals_weight_decay`).
- `regularized_loss` with `l1=l2=0` reproduces the bare loss; increasing `l2` strictly shrinks the resulting fitted weights toward 0.
