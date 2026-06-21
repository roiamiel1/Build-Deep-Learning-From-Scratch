# Stage 31: Vision Transformer

*Imports* `im2col`/`col2im` (`stage_25`) and `TransformerBlock`/`LayerNorm` (`stage_30`) via `dlfs.stage_import`; *adds* patch embedding, a learnable class token + positional embeddings, and a `ViT` model stacking the imported blocks with a linear classifier head.

**Context** — You have a working `TransformerBlock` (`stage_30`: residual MHA +
FFN + hand-derived `LayerNorm`) and the `im2col` patch-unfolding machinery
(`stage_25`). A **Vision Transformer (ViT)** drops the convolutional inductive
bias entirely: it cuts an image into a grid of fixed-size patches, treats each
patch as a "token", and runs the *exact* Transformer encoder you already built.
This stage wires the image-specific front end (patch embedding, class token,
learnable positional embeddings) and the classifier head around that stack.

**Background** — Given an image $X\in\mathbb R^{C\times H\times W}$ split into
$P\times P$ patches, there are $N_p=(H/P)(W/P)$ patches, each flattened to a
vector of length $C P^2$. A **patch embedding** is a single linear map
$E\in\mathbb R^{(CP^2)\times D}$ (plus bias) projecting every patch to the model
dimension $D$ — identical to a stride-$P$, kernel-$P$ convolution, which is why
you reuse `im2col` from `stage_25` to extract the patches. A learnable
**class token** $x_{\text{cls}}\in\mathbb R^{D}$ is prepended, giving a sequence
of length $N_p+1$, and a learnable **positional embedding**
$P_{\text{pos}}\in\mathbb R^{(N_p+1)\times D}$ is *added* (attention is otherwise
permutation-invariant, so position must be injected). The sequence
$$Z_0 = [\,x_{\text{cls}};\, p_1E;\dots;\, p_{N_p}E\,] + P_{\text{pos}}$$
is fed through $L$ `TransformerBlock`s (`stage_30`); a final `LayerNorm` is
applied and the **class token row only** is sent to a linear head $W_h$ for the
class logits. The new gradients here are all of the *linear/add* kind from
`stage_08`/`stage_11`: for the patch embed $Z=\mathrm{cols}\,E+b$,
$\partial L/\partial E=\mathrm{cols}^\top G$, $\partial L/\partial b=\sum_{\text{rows}}G$,
$\partial L/\partial\mathrm{cols}=G\,E^\top$ (scattered back with `col2im`);
the positional add routes its upstream grad straight into $P_{\text{pos}}$, and
the class token grad is just the upstream grad of row 0 summed over the batch.

**Watch**
- [An Image is Worth 16x16 Words: ViT explained (Yannic Kilcher)](https://www.youtube.com/watch?v=TrdevFK_am4) — the paper walkthrough: patches as tokens, class token, positional embeddings.
- [Vision Transformer (ViT) — paper + intuition](https://www.youtube.com/watch?v=j3VNqtJUoz0) — why patch+Transformer replaces convolution and how the pieces connect.

**Exercise** — Implement, in `code.py`, the ViT front end / head as explicit
NumPy `forward`/`backward` (the `stage_27`–`stage_30` paradigm: parameters are
`np.ndarray`, backward is hand-written from a forward cache). Reuse `im2col`
(`stage_25`), `TransformerBlock` + `LayerNorm` (`stage_30`) via `dlfs.stage_import`.
Allowed tools: NumPy (array math only) + stdlib + your prior-stage code. **No
autodiff library.** Inputs are images `(N, C, H, W)`; `H`, `W` must be divisible
by `P`.

- `class PatchEmbed(in_channels, patch_size, d_model, seed=None)`:
  - `E` shape `(in_channels*P*P, d_model)` (small random init, scale `1/sqrt(C*P*P)`), `b` zeros `(d_model,)`; grads `E_grad`, `b_grad` zeros.
  - `forward(x) -> np.ndarray`: use `im2col` with `kh=kw=P`, `stride=P`, `pad=0` to get `cols (N*Np, C*P*P)`, then `cols @ E + b` reshaped to `(N, Np, d_model)`. Cache `(cols, x_padded_shape, x.shape)`.
  - `backward(grad_out) -> np.ndarray`: `grad_out` is `(N, Np, d_model)`; set `E_grad = cols.T @ G`, `b_grad = G.sum(0)` (`G = grad_out.reshape(N*Np, d_model)`), and return `dx (N, C, H, W)` via `col2im(G @ E.T, …)`.
  - `parameters() -> [E, b]`; `zero_grad()`.
- `class ViT(image_size, patch_size, in_channels, d_model, n_heads, d_ff, depth, n_classes, *, norm="pre", seed=None)`:
  - Holds `patch_embed` (above), a learnable `cls_token` `(1, 1, d_model)`, learnable `pos_embed` `(1, Np+1, d_model)` (`Np=(image_size/patch_size)**2`), `depth` `TransformerBlock`s, a final `LayerNorm(d_model)`, and a head `W_head (d_model, n_classes)` + `b_head (n_classes,)`.
  - `forward(x) -> np.ndarray`: patch-embed → prepend `cls_token` (broadcast over batch) → add `pos_embed` → run the blocks in order → final `LayerNorm` → take **row 0** (class token) of every sequence → `@ W_head + b_head`. Returns logits `(N, n_classes)`.
  - `parameters()`: every learnable array, stable order (patch_embed, cls_token, pos_embed, each block's, final norm's, head's).
  - `zero_grad()`.
- Module-level helper `num_patches(image_size, patch_size) -> int`: `(image_size // patch_size) ** 2`; raise `ValueError` if not divisible.

**Done when**
- `pytest stage_31_vision_transformer/test.py` passes.
- `PatchEmbed.forward` output shape is `(N, Np, d_model)`; the embedding equals an independent flatten-then-matmul reference within `atol ~ 1e-10`.
- `PatchEmbed` gradcheck: central differences of `sum(W * forward(·))` w.r.t. `E`, `b`, and the input image match `*_grad`/`dx` within `atol ~ 1e-5`.
- `ViT.forward` returns logits `(N, n_classes)`; `cls_token` (1,1,D), `pos_embed` (1,Np+1,D) shapes are correct, sequence length into the blocks is `Np+1`.
- The classifier head + final LayerNorm + cls-token-select path gradchecks w.r.t. `W_head`, `b_head` within `atol ~ 1e-5` (with `depth=0`).
- The positional embedding is actually used: zeroing `pos_embed` vs. perturbing it changes the logits, and `pos_embed_grad` matches central differences within `atol ~ 1e-5`.
