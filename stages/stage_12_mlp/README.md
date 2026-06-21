# Stage 12: MLP

**Context** — You have a single `Neuron` from `stage_10` and a vector-valued `Dense` layer from `stage_11` — a *pure linear* map `Z = X @ W + b` (no activation), riding on the `Tensor` autodiff engine from `stage_09`. This stage stacks several `Dense` layers and inserts the nonlinearity *between* them to form a **multilayer perceptron (MLP)** with one clean `forward()`. This is the first *deep* network in the curriculum and the workhorse you will train in `stage_17` and reuse as the feed-forward block inside Transformers (`stage_31`).

**Background** — An MLP is a chain of $L$ affine-then-activation layers. With input $x$, layer $\ell$ computes
$$h^{(\ell)} = \phi^{(\ell)}\!\big(h^{(\ell-1)} W^{(\ell)} + b^{(\ell)}\big), \qquad h^{(0)} = x,$$
and the network output is $h^{(L)}$. Stacking *linear* maps alone collapses to a single linear map ($W_2(W_1 x) = (W_2 W_1)x$), so the **nonlinear** $\phi$ between layers is what gives the MLP its power. The **universal approximation theorem** says a single hidden layer with enough units and a non-polynomial $\phi$ can approximate any continuous function on a compact set to arbitrary accuracy; depth makes this far more *parameter-efficient* in practice. You write **no gradients** here — each `Dense` already backprops via `Tensor.backward()` from `stage_09`. The MLP's gradient is just the chain rule composed across layers, handled automatically: for a scalar loss $\mathcal{L}$, $\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$ is accumulated by the engine as the upstream gradient flows back through each layer's local Jacobian $\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} = \mathrm{diag}\!\big(\phi'(z^{(\ell)})\big)\,W^{(\ell)\top}$. Your only job is to wire the layers in order and collect their parameters.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy stacks layers into an MLP on top of an autodiff engine; the exact composition you build here.
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown's intuition for hidden layers and why depth + nonlinearity represents rich functions.

**Cumulative framework** — This stage imports `Tensor` (`stage_09`) and `Dense` (`stage_11`) via `dlfs.stage_import` and **adds** the new `MLP` class that composes `Dense` layers + activations on top of them (no redefinition of `Tensor`/`Dense`).

**Exercise** — Implement `class MLP` in `code.py`, using **only** the `Dense` layer from `stage_11`, the `Tensor` from `stage_09` (both imported for you via `dlfs.stage_import`), NumPy (forward array creation only), and the stdlib. Do **not** hand-write any gradient — all gradients must flow through `Tensor.backward()`. Remember: `Dense` is *linear only*; the MLP supplies the activation.

- `MLP(sizes: list[int], activation="tanh", out_activation="none", seed=None)`:
  - `sizes` is the full width list `[n_in, h1, h2, ..., n_out]` (length $\ge 2$). Build `len(sizes) - 1` `Dense` layers, where layer $i$ is `Dense(sizes[i], sizes[i+1], seed=layer_seed)`.
  - `activation` and `out_activation` are each in `{"tanh", "relu", "none"}`. Hidden layers use `activation`; the final layer uses `out_activation` (default `"none"`, i.e. linear output for regression/logits).
  - Use `seed` for reproducibility: derive a distinct per-layer seed (e.g. `seed + i`) so repeated construction with the same `seed` gives identical weights.
  - Store the layers in `self.layers` (a list) in forward order.
- `forward(self, x) -> Tensor`:
  - Accept `x` as a `Tensor` or array-like of shape `(n_in,)` (single example) or `(batch, n_in)` (a batch).
  - For each layer in order: compute the affine output `z = layer(x)`, then apply the layer's activation via the `Tensor`'s own method (`z.tanh()` / `z.relu()`, or `z` for `"none"`) — hidden layers use `activation`, the last uses `out_activation`. Return the final `Tensor`. Output shape is `(n_out,)` for a single example or `(batch, n_out)` for a batch.
- `__call__(self, x)`: alias for `forward(x)`.
- `parameters(self) -> list[Tensor]`: return **all** parameter `Tensor`s from every layer, concatenated in layer order (for future optimizers).
- `zero_grad(self) -> None`: reset `.grad` to zeros for every parameter.
- `__repr__`: e.g. `MLP([2, 16, 16, 1], activation='tanh', out_activation='none')`.
- Allowed tools: `numpy`, Python stdlib, your `stage_11` `Dense`, your `stage_09` `Tensor`. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_12_mlp/test.py` passes.
- Forward output shapes are correct for single-vector and batched inputs.
- `len(parameters())` equals the total number of `Dense` params (2 per layer: a weight and a bias).
- `.backward()` on a scalar reduction of the output populates `.grad` for every parameter and for a `requires_grad` input.
- Central-difference gradcheck of a scalar loss w.r.t. each layer's weights/biases and w.r.t. the input matches the autodiff gradients within tolerance (`atol ~ 1e-6`).
- Stacking layers with `out_activation="none"` is verified to be nonlinear (an MLP with a hidden nonlinearity is not equal to any single affine map).
