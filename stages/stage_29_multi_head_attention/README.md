# Stage 29: Multi-Head Attention

Imports `Tensor` (`stage_09`) and the single-head `SelfAttention` + `causal_mask` (`stage_28`); adds `MultiHeadAttention` (alias `MHA`) that holds `h` `SelfAttention` heads of width `d_k = d_model // h`, concatenates their outputs, and applies a learned `W_o` projection. No attention math is rewritten — every gradient flows through `Tensor.backward()`.

**Done when** `pytest stage_29_multi_head_attention/test.py` passes: forward shape `(T, d_model)`; `h=1` reduces to one `SelfAttention` head followed by `W_o`; `d_model % h != 0` raises `ValueError`; and a central-difference gradcheck of a scalar loss w.r.t. `W_o`, each head's `W_q/W_k/W_v`, and the input `x` matches the autodiff gradients within tolerance.
