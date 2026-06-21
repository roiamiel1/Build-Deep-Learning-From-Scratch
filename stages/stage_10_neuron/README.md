# Stage 10: Neuron

**Context** — You have a working N-dimensional autodiff `Tensor` from `stage_09` (with `.data`, `.grad`, `.backward()`, and ops like `+`, `*`, `@`/matmul, `tanh`, `relu`). This stage assembles the first *learnable unit*: a single artificial neuron that computes a weighted sum of its inputs, adds a bias, and squashes the result through a nonlinearity. It is the atom every layer, MLP, CNN, and Transformer in later stages is built from.

**Background** — A neuron is just an affine map followed by a pointwise activation. Given input vector $x \in \mathbb{R}^{n}$, weights $w \in \mathbb{R}^{n}$, and bias $b \in \mathbb{R}$, the pre-activation (logit) is the dot product plus bias and the output is the activation applied to it:
$$z = w \cdot x + b = \sum_{i=1}^{n} w_i x_i + b, \qquad y = \phi(z).$$
Because this is built on the `Tensor` engine from `stage_09`, you do **not** hand-write gradients here — you express $z$ with `Tensor` ops (`@` for the dot product, `+` for the bias) and let `.backward()` flow through. For understanding/checking, the local derivatives are:
$$\frac{\partial z}{\partial w_i} = x_i,\quad \frac{\partial z}{\partial x_i} = w_i,\quad \frac{\partial z}{\partial b} = 1,\quad \frac{\partial y}{\partial z} = \phi'(z).$$
For $\phi = \tanh$, $\phi'(z) = 1 - \tanh^2(z)$; for $\phi = \mathrm{relu}$, $\phi'(z) = \mathbb{1}[z > 0]$; for identity ($\phi = \text{none}$), $\phi'(z) = 1$. The full gradient w.r.t. a weight is then $\partial y/\partial w_i = \phi'(z)\, x_i$ by the chain rule from `stage_09`. The neuron must also support a **batch** input of shape `(batch, n)`, where the matmul produces a `(batch,)` vector of pre-activations and the bias broadcasts.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds a neuron on an autodiff engine; the exact pattern you implement here.
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown's geometric intuition for what a single neuron / weighted sum does.

**Builds on** — imports `Tensor` from `stage_09` (via `dlfs.stage_import`) and **adds** a new `Neuron` (weighted sum + bias + activation) on top of it; `Tensor` is reused unchanged, not redefined.

**Exercise** — Implement `class Neuron` in `code.py`, using **only** `Tensor` from `stage_09` (imported with `from dlfs import stage_import`), NumPy (forward array creation only), and the stdlib. Do **not** compute any gradient by hand; all gradients must come from `Tensor.backward()`.

- `Neuron(n_in: int, activation: str = "tanh", seed: int | None = None)`:
  - Create `self.w`: a `Tensor` of shape `(n_in,)` initialized with small random values (e.g. uniform in `[-1, 1]`); use `seed` for reproducibility when given.
  - Create `self.b`: a scalar `Tensor` (shape `()` or `(1,)`) initialized to `0`.
  - Store `activation` ∈ `{"tanh", "relu", "none"}`.
  - Both `w` and `b` must be leaf `Tensor`s that participate in autodiff (i.e. they accumulate `.grad`).
- `__call__(self, x) -> Tensor`:
  - Accept `x` as a `Tensor` (or array-like convertible to one) of shape `(n_in,)` **or** `(batch, n_in)`.
  - Compute the pre-activation `z = x @ self.w + self.b` using `Tensor` ops (`@` then `+`). For a `(n_in,)` input, `z` is a scalar `Tensor`; for `(batch, n_in)`, `z` has shape `(batch,)`.
  - Apply the activation: `tanh` → `z.tanh()`, `relu` → `z.relu()`, `none` → `z`. Return the resulting `Tensor`.
- `parameters(self) -> list[Tensor]`: return `[self.w, self.b]` (used by future optimizers).
- `zero_grad(self) -> None`: reset `.grad` of every parameter to zeros.
- `__repr__`: e.g. `Neuron(n_in=3, activation='tanh')`.
- Allowed tools: `numpy`, Python stdlib, and your `stage_09` `Tensor`. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_10_neuron/test.py` passes.
- Forward output shapes are correct for both single-vector and batched inputs.
- `.backward()` populates `w.grad`, `b.grad`, and (when the input requires grad) `x.grad`.
- Central-difference gradcheck of the neuron output w.r.t. `w`, `b`, and `x` matches the autodiff gradients within tolerance (`atol ~ 1e-6`) for every activation.
