# Stage 17: Momentum

**Context** — In `stage_14` you built plain `SGD`, which steps each parameter straight down the local gradient: `p -= lr * p.grad`. On ravines, badly-scaled or noisy loss surfaces this zig-zags and crawls. This stage adds **momentum** — a velocity buffer that accumulates an exponential moving average of past gradients — and its **Nesterov** variant, then has you measure the convergence speed-up. This is the first of the "smarter optimizer" stages that culminate in RMSProp/Adam later.

**Background** — Plain SGD has no memory: each update only sees the current
gradient $g_t = \partial L/\partial p$. Heavy-ball momentum gives the update a
*velocity* $v$ that remembers where it has been, damping oscillations across a
ravine while accelerating along consistent directions. The classic
("PyTorch-style") form is
$$v_t = \beta\, v_{t-1} + g_t, \qquad p_t = p_{t-1} - \eta\, v_t,$$
where $\beta \in [0,1)$ is the momentum coefficient (typically $0.9$) and
$\eta$ is the learning rate. Unrolling the recursion gives
$$v_t = \sum_{k=0}^{t} \beta^{k}\, g_{t-k},$$
an (unnormalized) **EMA** of past gradients: recent grads weigh $1$, older ones
decay by $\beta$ per step. So on a steady slope the effective step size grows
toward $\eta/(1-\beta)$ (a $10\times$ boost at $\beta=0.9$) — the heavy-ball
picture of a ball gaining speed downhill, while opposing grads on a zig-zag
cancel in the sum. **Nesterov** accelerated gradient instead evaluates the
gradient at a *look-ahead* point $p_{t-1} - \eta\beta v_{t-1}$, anticipating the
upcoming move and correcting sooner. With the standard reformulation that keeps
the look-ahead implicit, the Nesterov step is
$$v_t = \beta\, v_{t-1} + g_t, \qquad p_t = p_{t-1} - \eta\,(g_t + \beta\, v_t).$$
This stage computes **no new analytic gradients**: $g_t$ is just `p.grad`
produced by `Tensor.backward()` from `stage_09`. You only restructure how the
optimizer turns those grads into updates, building directly on the `SGD` API
from `stage_14`. Setting $\beta=0$ must recover that plain `SGD` exactly.

**Watch**
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown: what the raw gradient step does, so you see exactly what momentum is improving on.
- [Stanford CS231n lecture series](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) — the optimization lecture covers SGD vs momentum vs Nesterov and why the velocity buffer helps; ground truth for the update rules here.

**Cumulative chain** — imports `SGD` (stage_14) via `dlfs.stage_import` and **subclasses** it as `SGDMomentum` (aliased `Momentum`), adding only the velocity buffers, `beta`, the `nesterov` flag, and `reset()`. `zero_grad` / `params` / `lr` are inherited unchanged.

**Exercise** — Implement `class SGDMomentum(SGD)` (exported also as `Momentum`) in `code.py`, subclassing `SGD` from `stage_14` — do **not** rewrite SGD. Use **only** NumPy (array math on `.data`/`.grad`), the stdlib, and your prior-stage `Tensor`/`SGD`. Do **not** import any autodiff or optimizer library.

- `SGDMomentum(params, lr, beta=0.9, nesterov=False)` (call `super().__init__(params, lr)` to reuse stage_14's param/lr setup):
  - `params`: an iterable of leaf `Tensor`s (each has `.data` and `.grad`, as in `stage_09`); store as a list.
  - Store `lr`, `beta` (validate `0 <= beta < 1`), and the `nesterov` flag.
  - Allocate one **velocity buffer** per parameter, `np.zeros_like(p.data)`, in `self.velocities` (aligned with `self.params`). Buffers persist across `step()` calls.
- `step(self) -> None`:
  - For each `(p, v)`: read `g = p.grad`, update `v = beta * v + g` (heavy-ball) in place / stored back into `self.velocities`.
  - Update `p.data -= lr * v` when `nesterov=False`, or `p.data -= lr * (g + beta * v)` when `nesterov=True`.
  - Skip a parameter whose `grad` is `None`/all-zero only if `stage_14` did; otherwise treat zero grad normally.
- `zero_grad`: inherited unchanged from `stage_14`'s `SGD` (no override needed).
- `reset(self) -> None`: zero every velocity buffer (useful between independent runs).
- `__repr__`: e.g. `SGDMomentum(lr=0.1, beta=0.9, nesterov=False)`.
- Also implement the free function `quadratic_descent(optimizer_factory, x0, A, b, steps)`:
  - Minimize the quadratic $f(x) = \tfrac12 x^\top A x - b^\top x$ (so $\nabla f = A x - b$) by running `steps` optimizer steps on a single parameter `Tensor` initialized to `x0`; set `p.grad = A @ p.data - b` each iteration (no autodiff needed — it is the exact gradient).
  - Return the list of `f(x)` values per step. Used by the tests to compare convergence of `SGD` vs `Momentum` vs Nesterov.
- Allowed tools: `numpy`, Python stdlib, your `stage_09` `Tensor` and `stage_14` `SGD`. **No** PyTorch / autograd / etc.

**Done when**
- `pytest stage_17_momentum/test.py` passes.
- The velocity recursion matches a hand-unrolled EMA of the supplied gradients (checked numerically).
- With `beta=0`, `Momentum` reproduces plain `SGD` step-for-step.
- On the ill-conditioned quadratic, `Momentum` (and Nesterov) reach a lower $f(x)$ in the same number of steps than `SGD` — convergence speed-up confirmed.
- Gradient values fed to the optimizer match a central-difference gradient of $f$ within `atol ~ 1e-6`.
