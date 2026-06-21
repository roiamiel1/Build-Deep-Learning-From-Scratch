# Stage 28: Self-Attention

> Imports `Tensor` (`stage_09`) and references `softmax`/`cross_entropy_loss` (`stage_13`); adds single-head scaled dot-product self-attention (Q/K/V projections, stable row-wise softmax, weighted sum) plus the `causal_mask` helper.

**Context** — You have a `Tensor` autodiff engine (`stage_09`, broadcast-correct backward from `stage_12`) and have just built a recurrent cell (`RNN`, `stage_27`) that mixes information across a sequence *step by step*. Self-attention mixes every position with every other position *in one shot*, with no recurrence. This is the core primitive of the Transformer; `stage_29` splits it into multiple heads and `stage_31` wraps it in a full block.

**Background** — Given a sequence of $T$ token vectors $X \in \mathbb{R}^{T \times d_{\text{model}}}$, project it three ways with learned weights to get queries, keys, values:
$$Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\qquad W_\bullet \in \mathbb{R}^{d_{\text{model}} \times d_k}.$$
A query asks "what am I looking for?", a key advertises "what do I offer?", and a value carries "what I pass on if selected". Each query scores every key by a **dot product** (high when their directions align, i.e. similarity), giving a $T\times T$ score grid. The scores are scaled by $\sqrt{d_k}$ because a dot product of two $d_k$-dim vectors has variance $\propto d_k$; dividing keeps the logits' variance $\approx 1$ so the softmax does not saturate into a near one-hot (which would kill its gradient). A row-wise softmax turns scores into attention weights $A$ (each row a probability distribution over positions), and the output is the corresponding weighted average of values:
$$S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T},\qquad A = \operatorname{softmax}_{\text{row}}(S),\qquad O = AV \in \mathbb{R}^{T \times d_k}.$$
A **causal mask** sets $S_{ij} = -\infty$ for $j > i$ before the softmax, so position $i$ can only attend to positions $\le i$ (needed for autoregressive language models, `stage_32`); in practice add $-10^9$, not literal $-\infty$, so $\exp$ underflows to $0$ cleanly. This builds directly on the `stage_08` matmul gradients ($dL/dQ = G\,K$, $dL/dK = G^\top Q$ for $G = dL/dS$) and the `stage_13` log-sum-exp softmax. The only genuinely new gradient idea is softmax's Jacobian; for row $i$ with weights $a$ and logits $s$,
$$\frac{\partial a_k}{\partial s_j} = a_k(\delta_{kj} - a_j)\;\Rightarrow\; \frac{\partial L}{\partial s_j} = a_j\!\left(\frac{\partial L}{\partial a_j} - \sum_k a_k \frac{\partial L}{\partial a_k}\right),$$
a "subtract the weighted mean" rule. You write **none** of these by hand: build $S$, the mask, the softmax, and $O$ entirely from `Tensor` ops (`@`, scalar multiply, `exp`, `sum(axis=, keepdims=)`, broadcast `-`/`/`) and let `Tensor.backward()` from `stage_09` (with the broadcast-correct backward of `stage_12`) flow the gradients through. Reuse the stable detached-max softmax pattern from `stage_13` so no gradient leaks through the per-row shift.

**Watch**
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy derives scaled dot-product self-attention and the causal mask line by line; the exact construction here.
- [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown's geometric intuition for Q/K/V and the attention matrix.

**Exercise** — Implement `class SelfAttention` in `code.py`. Operate on a **single sequence** of shape `(T, d_model)` (batching/multi-head is `stage_29`). Use **only** the `Tensor` from `stage_09` (loaded for you), NumPy for *forward array creation only*, and the stdlib. Do **not** hand-write any `.grad`; every gradient must flow through `Tensor.backward()`.

- `SelfAttention(d_model: int, d_k: int, causal: bool = False, seed: int | None = None)`:
  - Create three weight `Tensor`s `W_q`, `W_k`, `W_v`, each shape `(d_model, d_k)`, small random init (scale ~ `1/sqrt(d_model)`), reproducible from `seed`.
  - Store `self.d_k`, `self.causal`.
- `forward(self, x) -> Tensor`:
  - Coerce `x` (shape `(T, d_model)`) to a `Tensor`.
  - Compute `Q = x @ W_q`, `K = x @ W_k`, `V = x @ W_v` (each `(T, d_k)`).
  - Scores `S = (Q @ K.T) * (1.0 / sqrt(d_k))`, shape `(T, T)`.
  - If `self.causal`, add a constant mask `M` (a non-differentiable `Tensor`) with `0` on/below the diagonal and a large negative number (e.g. `-1e9`) strictly above it, so masked logits vanish after softmax: `S = S + M`.
  - `A = softmax_rows(S)` using the detached-max + `exp` + `sum(axis=1, keepdims=True)` division pattern (numerically stable, `stage_13`).
  - Return `O = A @ V`, shape `(T, d_k)`. Expose the last attention matrix as `self.last_attn` (a `Tensor`) for inspection.
- `softmax_rows(self, s) -> Tensor`:
  - Stable row-wise softmax built from `Tensor` ops only: subtract a detached per-row max, `exp`, divide by the row sum. No NumPy gradient, no hand-written Jacobian.
- `attention_weights(self, x) -> np.ndarray`:
  - Convenience — run `forward` and return `self.last_attn.data` (rows sum to 1; with `causal=True`, strictly-upper entries are 0). Useful for plotting attention maps.
- `parameters(self) -> list[Tensor]`: `[W_q, W_k, W_v]`.
- `zero_grad(self) -> None`: reset every parameter's `.grad`.
- Module-level helper `causal_mask(T) -> np.ndarray`: `(T, T)` array, `0` on/below diagonal, `-1e9` above.
- Allowed tools: `numpy` (forward only), Python stdlib, your `stage_09` `Tensor`. **No** PyTorch / TensorFlow / JAX / autograd.

**Done when**
- `pytest stage_28_self_attention/test.py` passes.
- Forward shapes: `Q/K/V/O` are `(T, d_k)`, `S/A` are `(T, T)`; each row of `A` sums to 1.
- With `causal=True`, `attention_weights` is lower-triangular (strictly-upper entries are exactly 0) and rows still sum to 1.
- Scaling by `1/sqrt(d_k)` is present (a test reconstructs `S` from the public weights).
- `backward()` on a scalar reduction of `O` populates `.grad` for `W_q`, `W_k`, `W_v` and for a `requires_grad` input `x`.
- Central-difference gradcheck of a scalar loss w.r.t. each of `W_q`, `W_k`, `W_v` and w.r.t. `x` matches the autodiff gradients within tolerance (`atol ~ 1e-5`), for both `causal=False` and `causal=True`.
