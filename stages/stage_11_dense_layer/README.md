# Stage 11: Dense/Linear layer

**Context** — In `stage_10` you built a single `Neuron` (`y = phi(x @ w + b)`) on top of the autodiff `Tensor` from `stage_09`. A `Neuron` produces one output. A *dense* (a.k.a. fully-connected / linear) layer produces `n_out` outputs at once by stacking `n_out` neurons that share the same input — exactly the `nn.Linear` building block. This is the workhorse layer of every MLP, and later the projections inside attention.

**Background** — A dense layer is a single affine map. Given a batch of inputs $X \in \mathbb{R}^{B \times n_\text{in}}$, a weight matrix $W \in \mathbb{R}^{n_\text{in} \times n_\text{out}}$, and a bias $b \in \mathbb{R}^{n_\text{out}}$, the output is
$$Z = X W + \mathbf{1}_B\, b^\top, \qquad Z \in \mathbb{R}^{B \times n_\text{out}},$$
where $\mathbf{1}_B$ is a length-$B$ column of ones, so the same bias row is added to every example in the batch. Each row $Z_{i,:} = X_{i,:}W + b^\top$ is just `n_out` of the `stage_10` neurons evaluated on input $i$. As in `stage_10`, you do **not** hand-write any gradient: you express the forward pass with `Tensor` ops (`@` from `stage_09`'s matmul, `+` for the bias) and `Tensor.backward()` flows the chain rule. For reference / gradchecking, with upstream gradient $G = \partial L/\partial Z \in \mathbb{R}^{B \times n_\text{out}}$ the layer's gradients are
$$\frac{\partial L}{\partial W} = X^\top G, \qquad \frac{\partial L}{\partial b} = \mathbf{1}_B^\top G = \sum_{i=1}^{B} G_{i,:}, \qquad \frac{\partial L}{\partial X} = G\, W^\top.$$
Because `stage_09` only supports equal-shaped elementwise ops (full broadcasting backward is `stage_12`), add the bias **without** relying on broadcasting: build it up to `(B, n_out)` via `Tensor`s — e.g. `ones((B, 1)) @ b.reshape(1, n_out)` — so every op sees matching shapes and the bias gradient is summed over the batch automatically by the matmul backward.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy stacks neurons into a `Layer`; this stage is the matmul-vectorized version.
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown: a layer as a matrix transforming a vector of activations.

**Cumulative chain** — `code.py` does `Stage9_Tensor = stage_import("stage_09", "Tensor")` and `Stage10_Neuron = stage_import("stage_10", "Neuron")` (via `dlfs.stage_import`), re-exporting `Tensor` under its canonical name; this stage **adds** `Dense`, the vectorized matmul+bias layer (n_out neurons stacked) built on that `Tensor`.

**Exercise** — Implement `class Dense` in `code.py`, using **only** `Tensor` from `stage_09` (and the `Neuron` idea from `stage_10`), NumPy (forward array creation only), and the stdlib. Do **not** compute any gradient by hand; every gradient must come from `Tensor.backward()`.

- `Dense(n_in: int, n_out: int, bias: bool = True, seed: int | None = None)`:
  - Create `self.W`: a `Tensor` of shape `(n_in, n_out)` initialized with small random values (e.g. uniform in `[-1, 1]`); use `seed` for reproducibility when given.
  - Create `self.b`: a `Tensor` of shape `(n_out,)` initialized to `0` when `bias=True`; set `self.b = None` when `bias=False`.
  - Both `W` and `b` (when present) must be leaf `Tensor`s that accumulate `.grad`.
  - Store `n_in`, `n_out`, and the `bias` flag.
- `__call__(self, x) -> Tensor`:
  - Accept `x` as a `Tensor` (or array-like) of shape `(n_in,)` (single example) **or** `(B, n_in)` (a batch). Internally treat a 1-D input as a batch of size 1 (reshape to `(1, n_in)`).
  - Compute `Z = x @ self.W` using `Tensor`'s `@`. When `bias=True`, add the bias so each row gets `b`, **without** assuming broadcasting backward works (see Background). When `bias=False`, return `Z` unbiased.
  - Output shape: `(n_out,)` for a 1-D input, `(B, n_out)` for a 2-D input.
- `parameters(self) -> list[Tensor]`: return `[self.W, self.b]` if `bias` else `[self.W]` (used by future optimizers).
- `zero_grad(self) -> None`: reset `.grad` of every parameter to zeros.
- `__repr__`: e.g. `Dense(n_in=4, n_out=3, bias=True)`.
- Allowed tools: `numpy`, Python stdlib, and your `stage_09` `Tensor`. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_11_dense_layer/test.py` passes.
- Forward shapes are correct for single-vector and batched inputs, with and without bias.
- `.backward()` populates `W.grad`, `b.grad`, and (when the input requires grad) `x.grad`.
- Central-difference gradcheck of a scalar reduction of the output w.r.t. `W`, `b`, and `X` matches the autodiff gradients within tolerance (`atol ~ 1e-6`), including the batched case where `b.grad` sums over the batch.
