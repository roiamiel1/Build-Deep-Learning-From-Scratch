# Stage 23: Batch Normalization

**Context** — Good `He`/`Xavier` init from `stage_16` only fixes the *starting*
activation statistics; once weights move during training the per-layer
distribution drifts again ("internal covariate shift"). BatchNorm fixes this
*continuously*: it standardizes each feature across the batch, then rescales
with learnable `gamma`/`beta`. This is the first layer whose backward you must
**derive by hand** (the engine can't see through the batch statistics cheaply),
and the first with distinct **train vs eval** behaviour, building on the
`Dense`/`Module` conventions from `stage_16` and the regularization mindset of
`stage_22`.

**Background** — For a batch $x_1,\dots,x_B$ of one feature, BN computes
$$\mu=\tfrac1B\sum_i x_i,\quad \sigma^2=\tfrac1B\sum_i (x_i-\mu)^2,\quad
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y_i=\gamma\,\hat x_i+\beta.$$
Standardizing makes the layer scale-invariant and the loss surface smoother;
`gamma`/`beta` restore representational power (BN can learn the identity). The
parameter grads are easy sums: $\partial L/\partial\gamma=\sum_i g_i\hat x_i$ and
$\partial L/\partial\beta=\sum_i g_i$ where $g_i=\partial L/\partial y_i$. The
hard part is $\partial L/\partial x_i$, because $\mu$ and $\sigma^2$ both depend
on **every** $x_j$. Writing $\hat g_i=g_i\gamma$ and
$\text{istd}=(\sigma^2+\epsilon)^{-1/2}$, the collapsed result is
$$\frac{\partial L}{\partial x_i}=\frac{\text{istd}}{B}\Big(B\,\hat g_i
-\sum_j \hat g_j-\hat x_i\sum_j \hat g_j\hat x_j\Big).$$
The two subtracted sums are exactly the gradients that flow back through $\mu$
and $\sigma^2$; derive them by routing $g$ through each intermediate
($\hat x,\sigma^2,\mu$) and adding the paths. At **eval** time there is no batch:
BN uses a running mean/var (an EMA accumulated during training) so a single
example is normalized deterministically, with $\text{istd}$ folded into fixed
affine coefficients.

**Watch**
- [Building makemore Part 3: Activations, Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
  — Karpathy derives the exact BN backward and the running-stats/eval split from
  scratch; this stage is that derivation, isolated.
- [Batch Normalization (DeepLearningAI / Andrew Ng)](https://www.youtube.com/watch?v=tNIpEZLv_eg)
  — clean intuition for why standardizing activations stabilizes training.

**Cumulative chain** — imports `Tensor` (`stage_09`, via `dlfs.stage_import`) and
reuses the ones/zeros init convention (`stage_16`) and the `train()`/`eval()`
mode split (`stage_22`); **adds** a brand-new `BatchNorm1d` (hand-written
forward + backward, running stats, learnable `gamma`/`beta`) on top — it does not
redefine any earlier class.

**Exercise** — In `code.py`, implement a `BatchNorm1d` layer **without** autodiff:
you hand-write both `forward` and `backward`. Allowed tools: NumPy (array math
only) and the Python stdlib; `Tensor` is imported from `stage_09` via
`dlfs.stage_import` for shape/parity use. **No** PyTorch / TensorFlow / JAX /
autograd.

- `BatchNorm1d(num_features, *, eps=1e-5, momentum=0.1)`: learnable
  `gamma` (init ones) and `beta` (init zeros), each shape `(num_features,)`;
  buffers `running_mean` (zeros) and `running_var` (ones); attribute `training`
  (default `True`).
- `forward(x) -> np.ndarray`, `x` shape `(B, num_features)`:
  - **train**: normalize with the *batch* mean/var (biased, `1/B`), then update
    `running_mean = (1-momentum)*running_mean + momentum*batch_mean` (and var
    likewise, using the **unbiased** `1/(B-1)` var for the running estimate).
    Cache everything `backward` needs.
  - **eval** (`training=False`): normalize with `running_mean`/`running_var`
    (no updates, no cache needed). Both modes return shape `(B, num_features)`.
- `backward(grad_out) -> np.ndarray`: given `dL/dy` of shape `(B, num_features)`,
  return `dL/dx` (same shape) using the collapsed formula above, and set
  `self.gamma_grad`, `self.beta_grad` (shape `(num_features,)`). Backward is only
  defined after a **train**-mode forward.
- `train()` / `eval()` set `self.training`; `parameters()` returns
  `[gamma, beta]`; `zero_grad()` resets `gamma_grad`/`beta_grad` to zeros.
- Implement params/grads as plain `np.ndarray` (this layer owns its backward; it
  does not flow through the `stage_09` `Tensor` engine).

**Done when**
- `pytest stage_23_batchnorm/test.py` passes.
- Train-mode output has per-feature mean ≈ 0, var ≈ 1 (for `gamma=1,beta=0`);
  `gamma`/`beta` rescale as expected.
- Central-difference gradcheck of a scalar loss w.r.t. `x`, `gamma`, and `beta`
  matches `backward` within `atol ~ 1e-6`.
- Running stats update across train steps; eval mode uses them, is deterministic
  per-example, and ignores batch statistics; `train()`/`eval()` toggle behaviour.
