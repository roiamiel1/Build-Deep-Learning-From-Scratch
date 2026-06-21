# Stage 35: Capstone — tiny Transformer (char-level LM)

**Context** — The finale. You have a working framework: the `mytorch` surface from
`stage_32` (`Module`, `Parameter`, `Linear`, `Sequential`, `CrossEntropyLoss`, `Adam`,
`DataLoader`) and the `TransformerBlock` + hand-derived `LayerNorm` from `stage_30`
(which stacks `MultiHeadAttention` from `stage_29`). Here you wire them into a complete
**character-level language model (GPT-style)**, train it on a small text corpus until the
loss drops well below the uniform baseline, and **sample** new text autoregressively.
Nothing new is derived — this stage is *integration*: embeddings + positional encoding +
N transformer blocks + a tied/untied output head + next-token cross-entropy.

**Background** — A char-LM models $p(c_{t+1}\mid c_{\le t})$. Map each of $V$ characters to a
$D$-dim vector via a token embedding $E\in\mathbb{R}^{V\times D}$ (a learnable lookup),
add a learned positional embedding $P\in\mathbb{R}^{L\times D}$ so the model knows order
(attention is permutation-equivariant on its own — `stage_27`), run $N$ pre-norm
`TransformerBlock`s with a **causal mask** (`causal_mask` from `stage_28`: position $t$ may
only attend to $\le t$), apply a final `LayerNorm`, then project to $V$ logits with a
`Linear` head. Train with shifted next-token targets: input `x = chars[:-1]`, target
`y = chars[1:]`. The loss is mean softmax cross-entropy over all $B\cdot L$ positions,
$$\mathcal{L}=-\frac{1}{BL}\sum_{b,t}\log p_{b,t,\,y_{b,t}},\qquad p=\operatorname{softmax}(\text{logits}).$$
You do **not** hand-code its gradient — build it from `CrossEntropyLoss` (`stage_13`/`stage_32`),
so `backward()` (`stage_09`) yields the fused
$$\frac{\partial\mathcal{L}}{\partial\text{logits}}=\frac{\operatorname{softmax}(\text{logits})-Y}{BL}.$$
A random model has loss $\approx\ln V$; success means dropping clearly below that.
Sampling: feed a context, take the **last** position's logits, divide by `temperature`,
softmax, draw a character, append, slide the window to the last $L$ chars, repeat.

**Watch**
- [Let's build GPT: from scratch, in code, spelled out (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the exact thing you are building: char dataset, blocks, training loop, sampling.
- [Let's reproduce GPT-2 (124M) (Karpathy)](https://www.youtube.com/watch?v=l8pRSuU81PU) — scaling the same recipe; weight tying, init, the training loop in detail.

**Imports & adds** — imports `Tensor` (`stage_09`), `cross_entropy_loss` (`stage_13`),
`causal_mask` (`stage_28`), `TransformerBlock`/`LayerNorm` (`stage_30`), and
`Module`/`Parameter`/`Linear`/`Adam` (`stage_32`) via `dlfs.stage_import`; adds only the tiny
char-LM (`TransformerLM`) train + sampling script on top.

**Exercise** — Implement, in `code.py`, the char-LM end to end on top of prior stages. Allowed
tools: NumPy (forward array math only), Python stdlib, Matplotlib (plotting only), and your
own prior stages via `dlfs.stage_import`. **No** PyTorch/TensorFlow/JAX/autograd library.

- `class CharTokenizer`: `__init__(text)` builds a sorted vocab of unique chars with
  `stoi`/`itos`; `encode(s) -> np.ndarray[int]`; `decode(ids) -> str`; `vocab_size` property.
- `get_batch(data, block_size, batch_size, *, rng) -> (X, Y)`: sample `batch_size` random
  windows; `X` is `(B, L)` int ids, `Y` is `X` shifted by one (`(B, L)` int ids).
- `class TokenEmbedding(Module)` — `(vocab_size, d_model, seed=None)`: a `Parameter` table
  `E` of shape `(V, D)`; `forward(ids) -> Tensor` of shape `(B, L, D)` by row lookup
  (gather), grad-correct so repeated ids accumulate.
- `positional_embedding(block_size, d_model, seed=None) -> Parameter`: learned `(L, D)` table
  `P`; the model adds `P[:L]` (broadcast over batch) to the token embeddings.
- `class TransformerLM(Module)` —
  `(vocab_size, d_model, n_heads, d_ff, n_layers, block_size, *, seed=None)`:
  holds `TokenEmbedding`, positional `Parameter`, `n_layers` `TransformerBlock(... norm="pre")`
  from `stage_30`, a final `LayerNorm(d_model)`, and a `Linear(d_model, vocab_size)` head.
  - `forward(ids) -> Tensor`: logits of shape `(B, L, V)`; build the causal mask with
    `causal_mask(L)` (`stage_28`) and pass it to every block.
  - `parameters()`: every `Parameter` of embeddings, all blocks, final norm, and head.
- `lm_loss(logits, targets) -> Tensor`: flatten `(B, L, V) -> (B*L, V)` and `(B, L) -> (B*L,)`,
  return the scalar mean `cross_entropy_loss` (`stage_13`/`stage_32`).
- `train_lm(model, data, *, block_size, batch_size, steps, lr, seed=None) -> list[float]`:
  the loop — `get_batch`, `forward`, `lm_loss`, `loss.backward()`, `Adam.step()`,
  `zero_grad()` — appending `float(loss.data)` each step; return the history.
- `sample(model, tokenizer, *, prompt="", max_new_tokens, block_size, temperature=1.0, rng) -> str`:
  autoregressive generation as described above (last-position logits, temperature, softmax,
  multinomial draw, slide window).

**Done when**
- `pytest stage_35_capstone_transformer/test.py` passes.
- `forward` returns `(B, L, V)` logits and `lm_loss` returns a scalar `Tensor`; a fresh model's
  loss is `≈ ln(vocab_size)`.
- Central-difference gradcheck of `lm_loss` w.r.t. the output-head weights matches
  `Tensor.backward()` within tol (`atol ~ 1e-4`).
- `train_lm` on a tiny corpus drives the loss clearly below the `ln(vocab_size)` baseline.
- `sample` returns a `str` of the requested length drawn only from the tokenizer's vocab.
