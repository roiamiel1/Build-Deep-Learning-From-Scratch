# Stage 22: Dropout

**Context** — A regularizer that fights overfitting by randomly *dropping* units during training, forcing the network to spread its representation across many features instead of co-adapting a few. This is the second train/eval-mode layer in the curriculum, after the normalization layer in `stage_21`: like it, `Dropout` behaves differently in `train` vs `eval` mode and plugs into the `MLP` from `stage_12` between its `Dense` layers. You build it on the autodiff `Tensor` from `stage_09`, so backprop through the mask is automatic.

**Background** — Standard ("inverted") dropout, with keep probability $p$ (drop probability $1-p$): at **training** time, sample a per-element Bernoulli mask $m_{ij}\sim\mathrm{Bernoulli}(p)$, then output
$$y = \frac{m \odot x}{p},$$
where $\odot$ is elementwise product. Dividing by $p$ (the *inverted* part) keeps the expectation unchanged, $\mathbb{E}[y_{ij}] = \frac{1}{p}\,\mathbb{E}[m_{ij}]\,x_{ij} = \frac{1}{p}\,p\,x_{ij} = x_{ij}$, so the layer is mean-preserving and **no rescaling is needed at test time**. At **eval** time dropout is the identity, $y = x$. Because the forward pass is just an elementwise multiply by the constant factor $s = m/p$ (constant *given the sampled mask*), the gradient is trivial:
$$\frac{\partial y_{ij}}{\partial x_{ij}} = s_{ij} = \frac{m_{ij}}{p}\quad\Rightarrow\quad \frac{\partial L}{\partial x_{ij}} = \frac{\partial L}{\partial y_{ij}}\cdot \frac{m_{ij}}{p},$$
i.e. dropped units ($m_{ij}=0$) get zero gradient and kept units get their incoming gradient scaled by $1/p$. This is exactly `Tensor.__mul__` from `stage_09` against a freshly-sampled mask Tensor, so you reuse the engine's `_backward` rather than hand-deriving anything. The mask is resampled on every training forward pass; in eval mode you must *not* sample and *not* scale.

**Watch**
- [Dropout explained (DeepLearningAI / Andrew Ng)](https://www.youtube.com/watch?v=ARq74QuavAo) — what dropout does and why inverted scaling keeps test-time activations consistent.
- [Regularization in a neural network (StatQuest)](https://www.youtube.com/watch?v=6g0t3Phly2M) — the overfitting / co-adaptation intuition dropout is fighting.

**Cumulative** — imports `Tensor` (`stage_09`) and `MLP` (`stage_12`) via `dlfs.stage_import`; ADDS a new `Dropout` (inverted, train/eval mask) on top of `Tensor`, and `MLPDropout` which SUBCLASSES `MLP` to drop in a `Dropout` after each hidden layer.

**Exercise** — Implement inverted dropout in `code.py` on top of the `Tensor` from `stage_09` and `MLP` from `stage_12` (import them with `from dlfs import stage_import`). Allowed tools: `numpy` (mask sampling / forward arrays only), the Python stdlib, and your `stage_09` / `stage_12` code. No PyTorch / autograd.

- `Dropout(p_keep=0.5, *, seed=None)`: a layer with keep probability `p_keep` in `(0, 1]`. Store `p_keep`, a private RNG `np.random.default_rng(seed)`, a `training` flag (default `True`), and `self.mask` (the last sampled `m/p` array, or `None`). Validate `0 < p_keep <= 1` (else `ValueError`).
- `Dropout.__call__(x) -> Tensor` (and `forward` as an alias): coerce `x` to a `Tensor`.
  - **train mode** (`self.training`): sample `m ~ Bernoulli(p_keep)` with shape `x.data.shape` via the RNG; build the scale `s = m / p_keep`; store it in `self.mask`; return `x * Tensor(s)` so the engine's multiply backward routes `g * s` into `x.grad`. (`p_keep == 1.0` ⇒ all-ones mask ⇒ identity, but still a fresh multiply.)
  - **eval mode** (`not self.training`): set `self.mask = None` and return `x` unchanged (the identity — no sampling, no scaling, no new graph node needed).
- `Dropout.train()` / `Dropout.eval()`: set `self.training = True` / `False` and return `self` (chainable). `Dropout.parameters()` returns `[]` (dropout has no learnable params). Add a `__repr__` mentioning `p_keep`.
- `MLPDropout(sizes, *, p_keep=0.5, activation="tanh", out_activation="none", seed=None)`: **subclass** the `MLP` from `stage_12` — call `super().__init__(...)` to build the `Dense` stack and inherit `_apply_activation` / `parameters` / `zero_grad`. Keep a `Dropout(p_keep, seed=...)` to apply **after each hidden activation** (not after the output layer). `forward(x)` runs `dense -> activation -> dropout` per hidden layer, then the final `dense -> out_activation`. Add `train()` / `eval()` that flip the mode of the model **and every owned `Dropout`** (and return `self`); `parameters()` is inherited and returns only the `Dense` params (unchanged from `stage_12`).

**Done when**
- `pytest stage_22_dropout/test.py` passes.
- Eval mode is the exact identity: `eval()`-mode output equals the input, and repeated eval calls are deterministic.
- Train mode zeros a fraction `≈ 1-p_keep` of elements and scales survivors by `1/p_keep`; over many draws the mean output `≈` the input (inverted-dropout expectation).
- Central-difference gradcheck: with the mask **held fixed**, `dL/dx` from `backward()` matches `(f(x+eps)-f(x-eps))/(2*eps)` within `atol ~ 1e-6` — dropped coords get `0`, kept coords get `1/p_keep`.
