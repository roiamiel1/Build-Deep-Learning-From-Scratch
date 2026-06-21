# Stage 15: First training loop

**Context** — Every prior stage built one piece in isolation: the `Tensor` engine (`stage_09`), broadcasting backward (`stage_12`), matmul + reductions (`stage_13`), and activations (`stage_14`). Now you wire them into the canonical loop that actually *learns*: build an MLP, push a batch forward to a scalar loss, call `.backward()`, take a gradient-descent step, zero the grads, repeat. By the end you train a classifier on a non-linearly-separable toy dataset (moons or spiral) and watch the loss curve fall.

**Background** — Training is fixed-point iteration on the loss surface. With model parameters $\theta$ (the weight/bias `Tensor`s of every layer) and a per-batch scalar loss $L(\theta)$, **stochastic gradient descent** repeats four steps each iteration: forward ($\hat y = f_\theta(X)$, then $L$), backward (`L.backward()` fills every `p.grad` $=\partial L/\partial p$ via the autodiff from `stage_09`), step, and zero-grad. The step is
$$\theta \leftarrow \theta - \eta\, \frac{\partial L}{\partial \theta},$$
where $\eta$ is the learning rate. For a 2-class problem use mean-squared error against $\pm 1$ targets,
$$L = \frac{1}{B}\sum_{i=1}^{B}\big(\hat y_i - y_i\big)^2, \qquad \frac{\partial L}{\partial \hat y_i} = \frac{2}{B}\big(\hat y_i - y_i\big),$$
but you do **not** code that gradient: build $L$ from `Tensor` ops (`-`, `**2`, `.mean()` from `stage_13`) and let `.backward()` derive it. **Zero-grad matters**: `Tensor.grad` *accumulates* (`+=`) across `.backward()` calls (`stage_09`), so without resetting to zero each iteration you would sum gradients from every past batch and the step would be wrong. The MLP itself is a stack of `Dense` layers (`stage_11`) with a non-linear activation (`stage_14`) between them — the non-linearity is what lets it separate moons/spirals that no single `Dense` (an affine map) ever could.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy's final section *is* this loop: forward, `loss.backward()`, nudge params, **zero grad**, repeat; he shows the bug when you forget to zero.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown: the geometric picture of stepping downhill that the loop implements.

**Cumulative framework** — This stage does **not** redefine the model/loss/optimizer: it imports them via `dlfs.stage_import` (`Tensor` from `stage_09`, `MLP` from `stage_12`, `mse_loss` from `stage_13`, `SGD` from `stage_14`) and **adds** only the `train` driver, the `make_moons`/`make_spiral` toy datasets, the `accuracy` metric, and `plot_loss`.

**Exercise** — Implement the `make_moons`/`make_spiral` datasets, the `accuracy` metric, the `train` loop, and `plot_loss` in `code.py`, wiring the imported `MLP`/`mse_loss`/`SGD` into the canonical forward → loss → `backward()` → `step()` → `zero_grad()` loop. Use **only** the imported `stage_09`–`stage_14` framework, NumPy (data generation + forward array math only), Matplotlib (plot only), and the stdlib. Do **not** compute any gradient by hand; every gradient comes from `Tensor.backward()`.

- `make_moons(n, noise, seed) -> (X, y)` and `make_spiral(n_per_class, noise, seed) -> (X, y)`: return `X` of shape `(N, 2)` (float64) and `y` of shape `(N,)` in `{-1, +1}` (moons) or class labels you map to `{-1, +1}`. Deterministic given `seed`.
- `accuracy(pred: Tensor, y) -> float`: fraction where `sign(pred) == sign(y)` (read-only metric on `pred.data`).
- `train(model, X, y, *, lr, epochs, optimizer=None) -> list[float]`: run the full loop — forward, `mse_loss`, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()` — once per epoch (full-batch is fine), appending `float(loss.data)` each epoch; **return the loss history**. If `optimizer is None`, construct an `SGD` over `model.parameters()`.
- `plot_loss(history, path=None)`: Matplotlib line plot of loss vs epoch; save to `path` if given (no plotting inside `train`).
- **Imported, not implemented here:** `MLP` (`MLP([2, 16, 1], activation="tanh")`, with `__call__`/`parameters`/`zero_grad`) from `stage_12`; `mse_loss(pred, target) -> Tensor` (scalar MSE) from `stage_13`; `SGD(params, lr)` (in-place `p.data -= lr * p.grad`, plus `step`/`zero_grad`) from `stage_14`; `Tensor` from `stage_09`. Just import and wire them.
- Allowed tools: `numpy`, `matplotlib`, Python stdlib, and your imported `stage_09`–`stage_14` code. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_15_first_training_loop/test.py` passes.
- One training iteration strictly decreases the loss for a small step; loss history is monotone-ish and ends well below its start.
- The MLP reaches ≥ 90% accuracy on a noiseless moons set within a few hundred epochs.
- Central-difference gradcheck of `mse_loss` w.r.t. a parameter `Tensor` matches the autodiff `.grad` within tol (`atol ~ 1e-6`).
- After `optimizer.zero_grad()` every `p.grad` is all-zeros; two `.backward()` calls without a zero in between provably double the grad (test asserts this).
