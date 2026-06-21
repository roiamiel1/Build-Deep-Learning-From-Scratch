# Stage 18: Adam optimizer

**Context** — In `stage_17` you wrote a training loop that, each step, runs forward, calls `Tensor.backward()` (the `stage_09` engine), then nudges every parameter by plain SGD: `p.data -= lr * p.grad`. That works but is slow and fiddly: one global learning rate, no memory of past gradients. This stage builds **Adam** (Adaptive Moment Estimation), the default optimizer for almost every modern net, by composing two ideas — **momentum** (a velocity that smooths the gradient) and **RMSProp** (a per-parameter adaptive step from the gradient's running magnitude). You implement them as plain NumPy update rules over the parameter `Tensor`s produced by `parameters()` (from `Dense`/`MLP`, stages 11–12).

**Background** — All optimizers update parameter `\theta` from its gradient `g_t = \partial \mathcal{L}/\partial \theta` (already in `p.grad`). **Momentum** keeps an exponential moving average (EMA) of the gradient, $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, and steps along $m_t$ — averaging out noise and building speed in consistent directions. **RMSProp** keeps an EMA of the *squared* gradient, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, and divides the step by $\sqrt{v_t}$ so each coordinate gets a learning rate scaled to its own recent magnitude. **Adam fuses both.** With $m_0 = v_0 = 0$, the EMAs start biased toward zero, so Adam applies **bias correction** $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$ before updating:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad \theta_{t} = \theta_{t-1} - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.$$
The $1-\beta^t$ factor exists because unrolling the EMA from zero gives $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i} g_i$, whose weights sum to $1-\beta_1^t$, not 1; dividing by that makes $\hat m_t$ an unbiased average. Defaults: $\eta=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$. **Weight decay** adds an L2 pull toward zero; *decoupled* decay (AdamW) does $\theta \mathrel{-}= \eta\,\lambda\,\theta$ separately from the adaptive step rather than folding $\lambda\theta$ into $g_t$. There is **no new gradient to derive** here — Adam is an update rule applied to gradients the `stage_09` engine already computes.

**Watch**
- [Adam Optimizer Explained](https://www.youtube.com/watch?v=JXQT_vxqwIs) — momentum + RMSProp fused, with the bias-correction intuition.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest: the descent step Adam improves on.

**Builds on prior stages** — imports `Optimizer` + `SGD` (`stage_14`) and `Momentum` (`stage_17`) via `from dlfs import stage_import`, re-exports them, and ADDS only `RMSProp`, `Adam(Optimizer)`, and `AdamW(Adam)` (1st/2nd-moment EMAs + bias correction). Adam's first moment is stage_17's velocity, normalized.

**Exercise** — Implement the new optimizer classes in `code.py`, using **only** NumPy (for the array math on `p.data`/`p.grad`) and the Python stdlib. Operate on the parameter `Tensor`s from `stage_11`/`stage_12` (`p.data` and `p.grad` are NumPy arrays). Do **not** use any autodiff/optimizer library, and do **not** call `backward()` inside an optimizer — gradients are an input. The `Optimizer` base, `SGD` (both `stage_14`) and `Momentum` (`stage_17`) are imported, not re-derived.

- `class RMSProp(Optimizer)` — `RMSProp(params, lr=1e-2, beta=0.99, eps=1e-8, weight_decay=0.0)`:
  - Keep a squared-grad EMA `v` per parameter. Each step: `v = beta*v + (1-beta)*g**2`; `p.data -= lr * g / (sqrt(v) + eps)`.
- `class Adam(Optimizer)` — `Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`:
  - Keep per-parameter `m`, `v` (zeros) and an integer step counter `self.t` (starts 0; increment to 1 on the first `step`).
  - `weight_decay` is **coupled** (added into `g`, the classic Adam). Apply the EMA, bias-correction, and update rule above exactly.
- `class AdamW(Adam)` — same signature; **decoupled** weight decay: do the plain Adam update on `g` (no `weight_decay` in `g`), then additionally `p.data -= lr * weight_decay * p.data`.
- All optimizers: `step` mutates `p.data` in place and leaves `p.grad` untouched; per-parameter state lives in the optimizer (e.g. keyed by `id(p)` or parallel lists), never on the `Tensor`.
- Allowed tools: `numpy`, Python stdlib. **No** PyTorch/autograd/optimizer libs.

**Done when**
- `pytest stage_18_adam/test.py` passes.
- `Adam` with `betas=(0,0)` (pure rescale) reduces to a signed step `-lr * sign(g)` after bias correction; verified.
- One `Adam.step` on a quadratic `L = 0.5 * sum(theta**2)` (so `g = theta`) matches a hand-computed reference update for `m`, `v`, bias correction, and the new `theta`.
- Running `Adam`/`RMSProp`/`Momentum` on a convex quadratic drives the loss below a small threshold within a fixed budget of steps.
- Bias correction is checked: with constant gradients, `m`/`v` EMAs are unbiased after dividing by `1-beta**t`.
