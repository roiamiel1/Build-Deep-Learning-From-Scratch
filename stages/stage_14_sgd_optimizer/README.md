# Stage 14: SGD optimizer

**Imports + adds** — Imports `Tensor` from `stage_09` (via `dlfs.stage_import`) and uses it unchanged; ADDS a new `Optimizer` base class + `SGD` subclass — the optimizer contract later stages (momentum/Adam in `stage_18`) extend.

**Context** — You now have a `Tensor` engine (`stage_09`) with broadcasting-correct backward (`stage_12`) and matmul/reductions (`stage_13`), plus layers (`stage_11`'s `Dense`) that expose `.parameters()`. So far you have updated weights by hand: call `.backward()`, then write `p.data -= lr * p.grad` for each parameter inline. This stage factors that update out into a reusable **optimizer** object — the same `torch.optim.SGD` interface every later stage (the training loop, momentum/Adam in `stage_18`, CNNs, the Transformer) will plug into.

**Background** — Stochastic gradient descent is the simplest first-order optimizer. After a backward pass, every parameter tensor `p` holds `p.grad = ∂L/∂p` (accumulated correctly by `Tensor.backward()` from `stage_09`/`stage_12`/`stage_13`). SGD nudges each parameter a small step *down* the gradient:
$$\theta \leftarrow \theta - \eta\, \nabla_\theta L,$$
where $\eta$ (the learning rate `lr`) sets the step size. There is **no new gradient to derive** here — the optimizer only *consumes* gradients the autodiff engine produced; the only "math" is the update rule above applied elementwise to each parameter's `.data` using its `.grad`. Two responsibilities make this reusable: (1) `step()` applies the update to all collected params, and (2) `zero_grad()` resets every `p.grad` to zero *before* the next forward/backward, because `Tensor.backward()` **accumulates** (`+=`) into `.grad` — forget it and gradients from old batches leak into new ones. The optimizer is deliberately decoupled from the model: it is handed a flat *list of parameter `Tensor`s* (e.g. `list(model.parameters())`) and never needs to know the model's structure. This `Optimizer` base class + `SGD` subclass is the contract `stage_18` extends with momentum, weight decay, RMSProp, and Adam.

**Watch**
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown: what "step downhill by `lr * grad`" actually does to the loss surface.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy; the manual `p.data -= lr * p.grad` loop this stage packages into an optimizer.

**Exercise** — Implement `class Optimizer` (base) and `class SGD(Optimizer)` in `code.py`, using **only** the `Tensor` from `stage_09` (imported via `dlfs.stage_import`), NumPy (array math on `.data`/`.grad` only), and the stdlib. **No** PyTorch/autograd. Do not compute any gradient here — you only read `p.grad` that `Tensor.backward()` already filled.

- `Optimizer(params)`:
  - Accept `params` as an iterable of `Tensor`s (e.g. from `model.parameters()`). Materialize it into `self.params = list(params)` (an iterator must not be exhausted by one pass — store a concrete list).
  - `zero_grad(self) -> None`: set `p.grad = np.zeros_like(p.data)` for every `p` in `self.params`.
  - `step(self) -> None`: `raise NotImplementedError` — subclasses define the update.
- `SGD(Optimizer)`:
  - `SGD(params, lr: float = 0.01)`: store `self.lr` (validate `lr > 0`), call the base ctor to collect params.
  - `step(self) -> None`: for each `p` in `self.params`, apply `p.data -= self.lr * p.grad` **in place** (mutate `p.data`; do not rebind `p` to a new `Tensor`, and do not touch `p.grad` here). Skip a param only if its `.grad` is `None`.
  - `__repr__`: e.g. `SGD(lr=0.01, n_params=2)`.
- The optimizer must hold the **same** `Tensor` objects the model holds, so `step()` mutates the model's live parameters.
- Allowed tools: `numpy`, Python stdlib, your `stage_09` `Tensor`. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_14_sgd_optimizer/test.py` passes.
- One `SGD` step moves each `p.data` by exactly `-lr * p.grad` (checked elementwise).
- `zero_grad()` clears every `p.grad` to zeros.
- A short end-to-end check: on a convex quadratic loss, repeated `backward(); step(); zero_grad()` drives the loss monotonically down toward its minimum, and the analytic gradient at each step matches a central-difference gradcheck (`atol ~ 1e-6`).
