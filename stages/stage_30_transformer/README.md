# Stage 30: Transformer block

**Imports & adds** — imports `MultiHeadAttention` (stage_29), `MLP` (stage_12), and `Tensor` (stage_09) via `dlfs.stage_import`; adds hand-derived `LayerNorm` (+backward), residual wiring, and `TransformerBlock` (pre/post-norm) on top — attention/FFN are reused, not rewritten.

**Context** — You now have all the parts: multi-head attention (`MultiHeadAttention` from
`stage_29`), a position-wise feed-forward net (the `MLP`/`Dense` stack from `stage_12`), and the
hand-derived normalization pattern from `stage_23` (BatchNorm). This stage assembles them into the
canonical **Transformer block** — sublayer + residual + normalization, repeated twice — and derives
**LayerNorm backward by hand** (the per-token analogue of `stage_23`'s per-feature BatchNorm). One
block is the unit that gets stacked into a GPT (`stage_32`) or ViT (`stage_33`).

**Background** — A Transformer block is two residual sublayers: token mixing (attention) then a
position-wise MLP. Each sublayer is wrapped with a residual connection and a **LayerNorm**. Unlike
BatchNorm (`stage_23`), which standardizes each feature **across the batch**, LayerNorm standardizes
each **token across its $D$ features** — so it is batch-size independent and works at inference on a
single token. For a token vector $x\in\mathbb{R}^{D}$:
$$\mu=\tfrac1D\sum_j x_j,\quad \sigma^2=\tfrac1D\sum_j (x_j-\mu)^2,\quad
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\odot\hat{x}+\beta.$$
The two block variants differ only in *where* the norm sits. **Post-norm** (original paper):
$x\leftarrow \mathrm{LN}(x+\mathrm{Sublayer}(x))$. **Pre-norm** (GPT-2 onward, more stable to train):
$x\leftarrow x+\mathrm{Sublayer}(\mathrm{LN}(x))$. Residuals give gradients a short path (the
$+x$ branch passes upstream grad through untouched), which is why deep stacks train.

**LayerNorm backward (key gradient).** With $g=\partial L/\partial y$, $\hat g=g\odot\gamma$, and
$\text{istd}=1/\sqrt{\sigma^2+\epsilon}$, summing/averaging **over the feature axis** $D$ (not the
batch axis as in `stage_23`):
$$\frac{\partial L}{\partial x}=\frac{\text{istd}}{D}\Big(D\,\hat g-\textstyle\sum_j \hat g_j
-\hat x\odot\textstyle\sum_j \hat g_j\,\hat x_j\Big),\qquad
\frac{\partial L}{\partial\gamma}=\sum_{\text{tokens}} g\odot\hat x,\quad
\frac{\partial L}{\partial\beta}=\sum_{\text{tokens}} g.$$
This is `stage_23`'s collapsed formula with the reduction axis moved from batch to features.

**Watch**
- [Let's build GPT: from scratch (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the block: residual + LayerNorm + FFN, and pre-norm placement.
- [But what is a GPT? Transformers visually (3Blue1Brown)](https://www.youtube.com/watch?v=wjZofJX0v4M) — how attention + MLP blocks stack into the full model.

**Exercise** — Implement, in `code.py`, hand-derived `LayerNorm` and a `TransformerBlock` that wires
it with MHA + FFN. Allowed tools: NumPy (array math only) + stdlib, plus `MultiHeadAttention` from
`stage_29` and the `MLP` from `stage_12`, both imported through `dlfs.stage_import`. **No
autodiff library.** Like `stage_23`, the norm owns explicit `forward`/`backward` over raw
`np.ndarray`; the reduction axis is the **last** (feature) axis. Inputs are `(B, L, D)` (or `(L, D)`
unbatched); tokens = all positions flattened over batch and length.

- `LayerNorm(normalized_dim, *, eps=1e-5)`:
  - `gamma` ones `(D,)`, `beta` zeros `(D,)`; `gamma_grad`, `beta_grad` zeros `(D,)`.
  - `forward(x) -> np.ndarray`: compute $\mu,\sigma^2$ over the **last** axis (biased, $1/D$),
    `x_hat`, `y = gamma*x_hat + beta`. Cache `(x_hat, istd)` for backward. Shape preserved.
  - `backward(grad_out) -> np.ndarray`: return `dL/dx`; set `gamma_grad = sum(g*x_hat)` and
    `beta_grad = sum(g)` reduced over **all axes except the last**, using the boxed `dx` formula.
  - `parameters() -> [gamma, beta]`; `zero_grad()`; no batch/running stats (LayerNorm has none).
- `TransformerBlock(d_model, n_heads, d_ff, *, norm="pre", eps=1e-5, seed=None)`:
  - Holds `attn` (a `MultiHeadAttention(d_model, n_heads)` from `stage_29`), two `LayerNorm(d_model)`
    (`ln1`, `ln2`), and a position-wise `ffn` = `MLP([d_model, d_ff, d_model], activation="relu")`
    from `stage_12`. `norm` is `"pre"` or `"post"`.
  - `forward(x, mask=None) -> np.ndarray`: with `norm="pre"`,
    `h = x + attn(ln1(x), mask)`, then `out = h + ffn(ln2(h))`; with `norm="post"`,
    `h = ln1(x + attn(x, mask))`, then `out = ln2(h + ffn(h))`. Shape `(B,L,D)` in and out.
  - `parameters()`: all of attn's, both LayerNorms', and the FFN's params, in a stable order.
- Acceptance: LayerNorm output is per-token zero-mean/unit-var (when `gamma=1,beta=0`); LayerNorm
  `dx`, `dgamma`, `dbeta` match central-difference gradcheck within `1e-5`; pre vs post norm produce
  different outputs; a residual block with a zeroed sublayer is the identity (plus norm); the block
  preserves shape `(B, L, D)`.

**Done when**
- `pytest stage_30_transformer/test.py` passes.
- LayerNorm forward standardizes over features (mean ~0, var ~1 per token).
- LayerNorm `dx`/`dgamma`/`dbeta` match central-difference gradcheck within `1e-5`.
- Pre-norm and post-norm wiring both run and preserve `(B, L, D)`; outputs differ.
