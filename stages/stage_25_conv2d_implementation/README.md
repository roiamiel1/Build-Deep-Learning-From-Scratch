# Stage 25: Conv2D implementation

**Context** — `stage_24` motivated convolution as the right inductive bias for images (local, weight-shared filters) and sketched the naive sliding-window forward. This stage makes it real and fast: a full `Conv2D` layer with forward **and** backward, plus max/avg **pooling**, all built on the `Tensor` autodiff engine from `stage_09`. The trick is **im2col** — unfold every receptive field into a column so the whole convolution becomes one big matrix multiply, reusing the matmul machinery you already trust.

**Background** — A conv layer holds weights $W$ of shape $(C_{out}, C_{in}, k_h, k_w)$ and bias $b\in\mathbb R^{C_{out}}$. For input $X$ of shape $(N, C_{in}, H, W)$ with stride $s$ and padding $p$, the output has spatial size $H' = \lfloor (H+2p-k_h)/s\rfloor+1$ (and same for $W'$). **im2col** gathers each $C_{in}\cdot k_h\cdot k_w$ receptive field into a row, producing a matrix $\mathrm{col}$ of shape $(N\cdot H'\cdot W',\ C_{in}k_hk_w)$. Reshaping $W$ to $W_r$ of shape $(C_{in}k_hk_w,\ C_{out})$, the forward is just an affine map:
$$Y_{\text{mat}} = \mathrm{col}\,W_r + b, \qquad Y = \text{reshape/transpose to } (N, C_{out}, H', W').$$
Because this is a matmul, the gradients are the standard matmul gradients (`stage_08`/`stage_09`):
$$\frac{\partial L}{\partial W_r} = \mathrm{col}^\top\,G,\quad \frac{\partial L}{\partial \mathrm{col}} = G\,W_r^\top,\quad \frac{\partial L}{\partial b} = \sum_{\text{rows}} G,$$
where $G$ is the upstream gradient reshaped to $(N H'W',\ C_{out})$. The one new piece is **col2im**: $\partial L/\partial\mathrm{col}$ lives in the unfolded layout, so you must *scatter-add* each column back to the pixels it came from (overlapping windows accumulate). This is the exact transpose of im2col. **Pooling** needs no weights: avg-pool backward spreads each output grad equally ($1/(k_hk_w)$) over its window; max-pool backward routes the whole grad to the arg-max position only (others get 0), exactly like ReLU's gate from `stage_09`.

**Watch**
- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown; what the sliding kernel actually computes, with clean visuals.
- [MIT 6.S191: Convolutional Neural Networks](https://www.youtube.com/watch?v=NmLK_WQBxB4) — strides/padding/pooling and why CNNs beat dense nets on images.

**Exercise** — In `code.py`, build Conv2D + pooling on the `stage_09` `Tensor` (import it via the `_load_from_stage` loader; do **not** reimplement it). Conv forward/backward go through im2col + matmul and you wire the custom `_backward` by hand (this stage's whole point), so they hook into `Tensor.backward()`. Allowed tools: `numpy` (forward array math, gather/scatter via fancy indexing or `np.add.at`), Python stdlib, your `stage_09` `Tensor`. No PyTorch/TensorFlow/JAX/autograd.

- `im2col(x, kh, kw, stride, pad) -> (cols, x_padded_shape)`: pure-NumPy. `x` is a raw `(N, C, H, W)` array; return `cols` of shape `(N*H'*W', C*kh*kw)` (row = one flattened receptive field, channel-major) and the padded shape for col2im. Pad with zeros.
- `col2im(cols, x_padded_shape, kh, kw, stride, pad) -> np.ndarray`: the transpose of `im2col`; scatter-add `cols` back into a `(N, C, H, W)` array (overlaps **add**, then strip padding). `col2im(im2col(x)…)` is NOT identity (overlaps), but it IS the adjoint — tests check $\langle im2col(x), y\rangle = \langle x, col2im(y)\rangle$.
- `class Conv2D(out_channels, in_channels, kernel_size, stride=1, padding=0, bias=True, seed=None)`: `kernel_size` is an int (square). Holds `W` (`Tensor`, shape `(C_out, C_in, kh, kw)`, He/`stage_16`-style init) and `b` (`Tensor`, shape `(C_out,)` or `None`). `__call__(x) -> Tensor` accepts a `(N, C_in, H, W)` array/`Tensor`, returns a `(N, C_out, H', W')` `Tensor` whose `_backward` fills `W.grad`, `b.grad`, and the input's grad via the matmul + col2im rules above. `parameters()`/`zero_grad()` like `stage_11`'s `Dense`.
- `class MaxPool2D(kernel_size, stride=None)` and `class AvgPool2D(kernel_size, stride=None)`: `stride` defaults to `kernel_size` (non-overlapping). `__call__(x) -> Tensor` on `(N, C, H, W)` → `(N, C, H', W')`, no learnable params, with hand-wired `_backward` (max → route to arg-max; avg → spread $1/(k_hk_w)$). Cache arg-max indices on the forward pass.

**Done when**
- `pytest stage_25_conv2d_implementation/test.py` passes.
- Conv forward matches an independent naive sliding-window NumPy reference (random `N,C,k,stride,pad`) within `atol ~ 1e-10`.
- im2col/col2im pass the adjoint (inner-product) test.
- Central-difference gradcheck of a scalar loss `sum(out)` w.r.t. `Conv2D.W`, `Conv2D.b`, and the input matches `Tensor.grad` within `atol ~ 1e-6` (`eps ~ 1e-5`).
- Max/avg pool forward match references; their gradcheck (sum of outputs) matches analytic `.grad` within `atol ~ 1e-6` (max checked with distinct window values so the arg-max is unambiguous).
