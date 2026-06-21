# Stage 27: Attention mathematics

**Context** — Before wiring a real attention *layer* (next stages), you derive scaled
dot-product attention on paper and as pure NumPy reference functions — the attention analogue of
`stage_24` (convolution math). This is the single operation at the heart of the Transformer: a
content-based, differentiable lookup where every query reads a weighted blend of values. No layer,
no learned projections yet — just the Q/K/V roles, the $\sqrt{d_k}$ scale, the softmax over keys,
and the full backward pass.

**Background** — Attention is a soft dictionary lookup. Given queries $Q\in\mathbb{R}^{L_q\times d_k}$,
keys $K\in\mathbb{R}^{L_k\times d_k}$, and values $V\in\mathbb{R}^{L_k\times d_v}$, each query scores
every key by a dot product, normalizes those scores into weights over keys, and returns the
weighted sum of values:
$$S = \frac{QK^{\top}}{\sqrt{d_k}}\in\mathbb{R}^{L_q\times L_k},\qquad
A = \mathrm{softmax}(S)\ \text{(row-wise over keys)},\qquad O = AV\in\mathbb{R}^{L_q\times d_v}.$$
**Why scale by $\sqrt{d_k}$?** If the entries of $Q,K$ are independent with mean 0 and variance 1,
then $q\!\cdot\!k=\sum_{i=1}^{d_k} q_ik_i$ has variance $d_k$. Large-magnitude logits push softmax
into a near one-hot regime where its gradient $\to 0$ (saturation), so we divide by the standard
deviation $\sqrt{d_k}$ to keep $S$ at unit scale and the gradients healthy. This reuses the stable
**softmax / log-sum-exp** from `stage_13` (row max subtracted before exponentiating) and the matmul
gradient rules from `stage_08`: for $C=AB$ with upstream $G$, $\partial L/\partial A=G B^{\top}$ and
$\partial L/\partial B=A^{\top}G$ ("dimension analysis").

**Backward.** Let $G=\partial L/\partial O$. Through $O=AV$ (a `stage_08` matmul):
$$\frac{\partial L}{\partial A}=G\,V^{\top},\qquad \frac{\partial L}{\partial V}=A^{\top}G.$$
Through the row-wise softmax $A=\mathrm{softmax}(S)$, with $g^A=\partial L/\partial A$, the
Jacobian-vector product is, **per row** (the off-diagonal terms come from the shared normalizer):
$$\frac{\partial L}{\partial S_{i}} = A_i \odot\!\Big(g^A_i - (g^A_i\!\cdot\! A_i)\,\mathbf{1}\Big),
\quad\text{i.e.}\quad \frac{\partial L}{\partial S_{ij}}=A_{ij}\Big(g^A_{ij}-\sum_{k}g^A_{ik}A_{ik}\Big).$$
Finally through $S=QK^{\top}/\sqrt{d_k}$ (matmul again), with $g^S=\partial L/\partial S$:
$$\frac{\partial L}{\partial Q}=\frac{1}{\sqrt{d_k}}\,g^S K,\qquad
\frac{\partial L}{\partial K}=\frac{1}{\sqrt{d_k}}\,(g^S)^{\top}Q.$$
An optional additive **mask** $M$ (with $-\infty$ at forbidden positions) is added to $S$ before
softmax; masked entries get weight $\approx 0$ and pass $0$ gradient.

**Watch**
- [Attention in transformers, visually explained (3Blue1Brown)](https://www.youtube.com/watch?v=eMlx5fFNoYc) — Q/K/V as lookup, the $QK^\top$ score matrix, and softmax over keys.
- [Let's build GPT: from scratch (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the self-attention math block, scaling, and masking in code.

**Exercise** — Math only: implement the pure NumPy functions in `code.py`. Allowed tools: Python
stdlib and NumPy for array math only. **No autodiff library, no layer class, no learned projections.**
All arrays are plain `np.ndarray`; reuse the stable-softmax idea from `stage_13` and the matmul-grad
reasoning from `stage_08`. Support an unbatched case `(L, d)` and an optional leading batch axis
`(B, L, d)`.

- `softmax_rows(x) -> np.ndarray`: numerically stable softmax over the **last** axis (subtract the
  per-row max as a constant before `exp`). Rows sum to 1.
- `softmax_backward(dA, A) -> np.ndarray`: given upstream `dA` and the softmax output `A`, return
  `dS` using the per-row JVP above ($dS = A\odot(dA - \mathrm{sum}(dA\odot A,\text{last axis},\text{keepdims})))$.
  No recomputation of the softmax.
- `attention_forward(Q, K, V, mask=None) -> (O, A)`: compute $S=QK^\top/\sqrt{d_k}$, add `mask` if
  given, `A = softmax_rows(S)`, `O = A @ V`. Return output `O` and weights `A`. Infer $d_k$ from
  `Q.shape[-1]`. Works for shapes `(L_q,d_k)/(L_k,d_k)/(L_k,d_v)` and the batched `(B,...)` variants.
- `attention_backward(dO, Q, K, V, A, mask=None) -> (dQ, dK, dV)`: analytical gradients via the four
  boxed rules — `dV = A^T @ dO`, `dA = dO @ V^T`, `dS = softmax_backward(dA, A)`, then
  `dQ = (dS @ K)/sqrt(d_k)`, `dK = (dS^T @ Q)/sqrt(d_k)`. Shapes must match `Q`, `K`, `V`.
- Acceptance: rows of `A` sum to 1 and are non-negative; `attention_forward` matches a naive
  loop reference; large logits do not overflow; `dQ, dK, dV` match central-difference numerical
  gradients within `1e-4`; with a `-inf` mask, masked weights are `~0` and carry `0` gradient.

**Done when**
- `pytest stage_27_attention_math/test.py` passes.
- `softmax_rows` is stable for logits like `[1000, 1001, 1002]` (no `inf`/`nan`) and rows sum to 1.
- `attention_forward` equals the naive per-query reference; the $1/\sqrt{d_k}$ scale is present.
- `dQ`, `dK`, `dV` from `attention_backward` match central-difference gradcheck within `1e-4`.
