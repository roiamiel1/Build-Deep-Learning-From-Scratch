# Stage 11: MLP

**Context** — You have a single `Neuron` from `stage_09` and a vector-valued `Dense` layer from `stage_10` — a *pure linear* map `Z = X @ W + b` (no activation), riding on the `Tensor` autodiff engine from `stage_08`. This stage stacks several `Dense` layers and inserts the nonlinearity *between* them to form a **multilayer perceptron (MLP)** with one clean `forward()`. This is the first *deep* network in the curriculum and the workhorse you will train in `stage_16` and reuse as the feed-forward block inside Transformers (`stage_30`).

**Background** — An MLP is a chain of $L$ affine-then-activation layers. With input $x$, layer $\ell$ computes
$$h^{(\ell)} = \phi^{(\ell)}\!\big(h^{(\ell-1)} W^{(\ell)} + b^{(\ell)}\big), \qquad h^{(0)} = x,$$
and the network output is $h^{(L)}$. Stacking *linear* maps alone collapses to a single linear map ($W_2(W_1 x) = (W_2 W_1)x$), so the **nonlinear** $\phi$ between layers is what gives the MLP its power. The **universal approximation theorem** says a single hidden layer with enough units and a non-polynomial $\phi$ can approximate any continuous function on a compact set to arbitrary accuracy; depth makes this far more *parameter-efficient* in practice. You write **no gradients** here — each `Dense` already backprops via `Tensor.backward()` from `stage_08`. The MLP's gradient is just the chain rule composed across layers, handled automatically: for a scalar loss $\mathcal{L}$, $\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$ is accumulated by the engine as the upstream gradient flows back through each layer's local Jacobian $\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} = \mathrm{diag}\!\big(\phi'(z^{(\ell)})\big)\,W^{(\ell)\top}$. Your only job is to wire the layers in order and collect their parameters.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy stacks layers into an MLP on top of an autodiff engine; the exact composition you build here.
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown's intuition for hidden layers and why depth + nonlinearity represents rich functions.

**Cumulative framework** — This stage imports `Tensor` (`stage_08`) and `Dense` (`stage_10`) via `dlfs.stage_import` and **adds** the new `MLP` class that composes `Dense` layers + activations on top of them (no redefinition of `Dense`). It *also* adds a broadcasting-aware `Tensor` subclass: `class Tensor(Stage8_Tensor)` overrides the elementwise binary ops (`__add__`/`__mul__`, and the derived/reflected forms that compose out of them) so that differently-shaped-but-broadcastable operands forward via NumPy broadcasting *and* backprop correctly — each parent's gradient is reduced ("unbroadcast") back to its own shape. This is exactly the "general broadcasting gradient reduction" that `stage_08` promised to defer to here, and `stage_12`'s stable softmax (`logits - logsumexp(..., keepdims=True)`, i.e. `(B,C) - (B,1)`, and bias-row broadcasting) relies on it.

**Background (unbroadcast)** — NumPy broadcasting lets `z = f(x, y)` operate on operands of different but compatible shapes by virtually *copying* a smaller operand across the broadcast axes. The chain rule says a value copied across an axis must receive the **sum** of the upstream gradient over those copies. So when the forward broadcasts, the upstream gradient `g = ∂L/∂z` comes back at the *broadcast* shape and must be *reduced* to each operand's original shape before accumulating. The rule (the "unbroadcast"): (1) sum away any extra **leading** axes the gradient gained from rank promotion (`(3,) → (2,3)`), then (2) for every axis where the operand had size 1 but the broadcast shape was larger, `sum(axis=i, keepdims=True)` (that size-1 axis was stretched), then (3) `reshape` to the operand's shape. Example: for `z = a + b` with `a:(2,3)`, `b:(3,)`, `a` gets `g` unchanged while `b` gets `g.sum(axis=0)` of shape `(3,)`. For multiply, apply the local factor first (`g * other`, `g * self`, evaluated at the broadcast shape) and *then* unbroadcast each to its parent's shape.

**Exercise** — Implement `class MLP` in `code.py`, using **only** the `Dense` layer from `stage_10`, the `Tensor` from `stage_08` (both imported for you via `dlfs.stage_import`), NumPy (forward array creation only), and the stdlib. Do **not** hand-write any gradient — all gradients must flow through `Tensor.backward()`. Remember: `Dense` is *linear only*; the MLP supplies the activation.

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
- Allowed tools: `numpy`, Python stdlib, your `stage_10` `Dense`, your `stage_08` `Tensor`. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_11_mlp/test.py` passes.
- Broadcasting binary ops between differently-shaped operands forward via NumPy broadcasting *and* gradcheck correctly — after `backward()` each operand's `.grad` is reduced to that operand's own shape. E.g. `(2,3) + (3,)`: the `(3,)` operand's grad is the column-sums `(3,)` and the `(2,3)` operand's grad is ones; `(B,C) - (B,1)`: the `(B,1)` operand's grad sums across the `C` axis to shape `(B,1)`.
- Forward output shapes are correct for single-vector and batched inputs.
- `len(parameters())` equals the total number of `Dense` params (2 per layer: a weight and a bias).
- `.backward()` on a scalar reduction of the output populates `.grad` for every parameter and for a `requires_grad` input.
- Central-difference gradcheck of a scalar loss w.r.t. each layer's weights/biases and w.r.t. the input matches the autodiff gradients within tolerance (`atol ~ 1e-6`).
- Stacking layers with `out_activation="none"` is verified to be nonlinear (an MLP with a hidden nonlinearity is not equal to any single affine map).
