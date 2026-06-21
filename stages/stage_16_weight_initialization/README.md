# Stage 16: Weight initialization

**Context** — You can now stack `Dense` layers from `stage_11` and pass signals
through the activations from `stage_15`. But *how* you fill the weight matrices
before training decides whether a deep net learns at all. This stage derives the
variance-propagation rule for a forward pass and uses it to build the two
canonical schemes — **Xavier/Glorot** (for `tanh`/linear) and **He/Kaiming**
(for `relu`) — then demonstrates the failure modes (vanishing/exploding
activations, dead ReLUs, saturated tanh) that bad init produces.

**Background** — Consider one linear layer $z_j = \sum_{i=1}^{n_\text{in}}
W_{ji}\,x_i$ with i.i.d. zero-mean inputs and weights, mutually independent.
The output variance is then a sum of $n_\text{in}$ independent product terms:
$$\operatorname{Var}(z_j) = \sum_{i=1}^{n_\text{in}}
\operatorname{Var}(W_{ji})\operatorname{Var}(x_i)
= n_\text{in}\,\operatorname{Var}(W)\,\operatorname{Var}(x).$$
To keep the *forward* signal variance stable ($\operatorname{Var}(z) =
\operatorname{Var}(x)$) we need $\operatorname{Var}(W) = 1/n_\text{in}$; running
the same argument on the backward pass (where the multiplier is $n_\text{out}$)
asks for $\operatorname{Var}(W) = 1/n_\text{out}$. **Xavier/Glorot** compromises
between the two:
$$\operatorname{Var}(W) = \frac{2}{n_\text{in} + n_\text{out}}
\;\Rightarrow\; U\!\left[-\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}},\,
\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}}\right]
\;\text{or}\; \mathcal N\!\left(0,\tfrac{2}{n_\text{in}+n_\text{out}}\right).$$
A ReLU zeros (in expectation) half its inputs, halving the output variance, so
**He/Kaiming** doubles the numerator to compensate:
$$\operatorname{Var}(W) = \frac{2}{n_\text{in}}
\;\Rightarrow\; \mathcal N\!\left(0,\tfrac{2}{n_\text{in}}\right)
\;\text{or}\; U\!\left[-\sqrt{\tfrac{6}{n_\text{in}}},\,
\sqrt{\tfrac{6}{n_\text{in}}}\right].$$
If $\operatorname{Var}(W)$ is too small the per-layer factor
$n_\text{in}\operatorname{Var}(W) < 1$ shrinks activations toward 0 across depth
(vanishing signal, dead ReLUs); too large pushes $\tanh$ into its flat tails
(saturation) where $\phi'(z)\approx 0$ and gradients die. Crucially, the
gradient of a `Dense` layer is unchanged from `stage_11` — $\partial L/\partial
W = X^\top G$ — so init only sets the *starting point* of $W$, and a freshly
initialized layer must still gradcheck exactly.

**Watch**
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
  — Karpathy shows dead/saturated activations and Kaiming init on a real net;
  this stage is the from-scratch version.
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
  — 3Blue1Brown: signal flowing layer-to-layer, the intuition variance
  propagation formalizes.

**Framework chain** — `code.py` imports `Dense` (`stage_11`) and `Tensor`
(`stage_09`) via `dlfs.stage_import` and uses them unchanged, adding *on top*
the `xavier_*`/`he_*` samplers plus `init_dense` (apply an init to a `Dense`
in place) and the `forward_activation_stats` harness. No class is redefined.

**Exercise** — In `code.py`, implement the initializers and a measurement
harness, using **only** NumPy, the stdlib, `Dense` from `stage_11`, and `Tensor`
from `stage_09`. Initializers return raw `np.ndarray` weights (no gradients — an
initializer just *samples* numbers); the harness reuses `Dense`, whose backward
already exists. Do **not** hand-write any gradient.

- `xavier_uniform(n_in, n_out, *, gain=1.0, seed=None) -> np.ndarray` and
  `xavier_normal(n_in, n_out, *, gain=1.0, seed=None) -> np.ndarray`: shape
  `(n_in, n_out)`. Uniform uses limit $a = \text{gain}\sqrt{6/(n_\text{in} +
  n_\text{out})}$; normal uses std $\text{gain}\sqrt{2/(n_\text{in} +
  n_\text{out})}$.
- `he_normal(n_in, n_out, *, seed=None) -> np.ndarray` and
  `he_uniform(n_in, n_out, *, seed=None) -> np.ndarray`: shape `(n_in, n_out)`.
  Normal std $\sqrt{2/n_\text{in}}$; uniform limit $\sqrt{6/n_\text{in}}$.
- `init_dense(layer, W, b=None) -> None`: overwrite `layer.W.data` (and
  `layer.b.data` if `b` is given) of a `stage_11` `Dense` **in place**, keeping
  them the same leaf `Tensor`s so they still accumulate `.grad`; reset their
  `.grad` to zeros. Raise `ValueError` on a shape mismatch.
- `forward_activation_stats(sizes, init_fn, activation, *, n_samples=512,
  seed=None) -> list[dict]`: build `len(sizes)-1` `Dense` layers (layer `k` maps
  `sizes[k] -> sizes[k+1]`), init each weight with `init_fn(n_in, n_out)` and
  zero bias, push `n_samples` of $\mathcal N(0,1)$ input through the stack
  applying `activation` ("tanh"/"relu"/"none") after each layer, and return one
  per-layer dict with keys `mean`, `std`, `saturated` (fraction with $|v|>0.98$,
  for tanh) and `dead` (fraction `== 0`, for relu).
- Allowed tools: `numpy`, Python stdlib, `stage_11` `Dense`, `stage_09`
  `Tensor`. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stage_16_weight_initialization/test.py` passes.
- Each initializer returns shape `(n_in, n_out)` and its empirical variance
  matches the target $\operatorname{Var}(W)$ within tolerance; `seed` makes
  draws reproducible and `gain` scales the std.
- Xavier-init `tanh` (and He-init `relu`) stacks keep per-layer `std` roughly
  constant with depth, a tiny init collapses `std`→0 (vanishing), and a large
  init drives `saturated`/`dead` fractions up — all asserted in tests.
- Central-difference gradcheck of a scalar reduction of a He/Xavier-initialized
  `Dense`'s output w.r.t. `W`, `b`, and `X` matches the `stage_11` autodiff
  gradients within `atol ~ 1e-6`.
