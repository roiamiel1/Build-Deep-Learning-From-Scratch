# Stage 20: DataLoader

**Context** — In `stage_19` you wrote `iterate_minibatches`, a one-shot generator
that shuffled and chopped `(X, y)` into batches. That works, but every real framework
wraps this in a reusable, *re-iterable* object: PyTorch's `Dataset` + `DataLoader`.
This stage builds that object — a `Dataset` you index and `len()`, and a `DataLoader`
that implements the **iterator protocol** so `for X_b, y_b in loader:` re-shuffles
each epoch and yields tensor batches — plus a deterministic `train_val_split`. No new
gradients enter here; the autodiff `Tensor` (`stage_09`) and the `train_minibatch`
loop (`stage_19`) are reused verbatim, with the loader feeding them.

**Background** — Mini-batch SGD (`stage_19`) needs, every epoch, a fresh random partition of the $N$ examples into batches. The math is unchanged: a batch of size $B$ gives the unbiased gradient estimate
$$\hat g_B=\frac{1}{B}\sum_{i\in\text{batch}}\nabla_\theta\ell_i,\qquad \mathbb{E}[\hat g_B]=\nabla_\theta L,$$
the same `mse_loss().backward()` from `stage_15`/`stage_19`. What this stage adds is
**engineering**, not calculus:
(1) a `Dataset` abstraction — `__len__` returns $N$, `__getitem__(i)` returns one
`(x_i, y_i)` row — so the loader never needs to know where the data lives;
(2) an *iterable* `DataLoader` that separates one-time config (batch size, shuffle,
drop_last, seed) from a per-epoch **iterator** built by `__iter__`, so the *same*
loader object can be looped over for many epochs and re-shuffles each time (a fresh
permutation per `__iter__`);
(3) `drop_last`, which drops the final ragged batch of size $N\bmod B$ so every
yielded batch is exactly $B$ wide — important once BatchNorm (`stage_21`) needs a
fixed batch shape;
(4) a `train_val_split` that partitions the row indices **once** (held-out
validation never leaks into training).
With `shuffle=True` the loader yields $\lfloor N/B\rfloor$ batches when `drop_last`
else $\lceil N/B\rceil$; the per-coordinate gradient variance still scales like
$\sigma^2/B$ exactly as in `stage_19`.

**Watch**
- [Datasets & DataLoaders (PyTorch official)](https://www.youtube.com/watch?v=Zvd276j9sZ8) — what `Dataset.__getitem__`/`__len__` and `DataLoader` batching/shuffling do; you are reimplementing exactly this.
- [The spelled-out intro to neural networks: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy; the training loop these batches feed (forward/backward/step/zero_grad).

**Cumulative chain** — this stage imports `Tensor` (`stage_09`) and `MLP, mse_loss, SGD, train_minibatch` (`stage_19`) via `dlfs.stage_import` and **adds** the data-feeding layer on top: `Dataset`, `DataLoader` (batch / shuffle / drop_last / iterator protocol), `train_val_split`, and a `train_with_loader` driver.

**Exercise** — Implement a `Dataset`, a `DataLoader`, and `train_val_split` in `code.py`. Reuse `Tensor` (`stage_09`), and `MLP`, `mse_loss`, `SGD`, `train_minibatch` (`stage_19`) — import them; do **not** reimplement. Allowed tools: `numpy` (permutation + indexing only), the stdlib. No PyTorch / autograd. No hand-written gradients.

- `Dataset(X, y)`: wrap two array-likes of equal first dimension $N$.
  - `__len__()` -> `N`. `__getitem__(i)` -> `(X[i], y[i])` as NumPy arrays (single row each). Mismatched `len(X) != len(y)` raises `ValueError`. Supports integer-array / slice indexing returning sub-arrays.
- `DataLoader(dataset, batch_size, *, shuffle=False, drop_last=False, seed=None)`:
  - Validate `batch_size >= 1`; else `ValueError`. Store config; do **not** shuffle in `__init__`.
  - `__len__()` -> number of batches per epoch: `N // batch_size` if `drop_last` else `ceil(N / batch_size)`.
  - `__iter__()` -> a **fresh iterator** each call: build the index order (a new `np.random.default_rng(seed)` permutation if `shuffle`, else `arange(N)`), then yield `(X_b, y_b)` as **`Tensor`** objects (stack the dataset rows for the batch indices). With `drop_last`, skip the trailing batch of size `< batch_size`. Two passes over the *same* loader must each cover every row (order may differ when shuffling).
  - The yielded `X_b` has shape `(b, n_in)` and `y_b` shape `(b,)` or `(b, 1)` — consistent with what `train_minibatch` expects.
- `train_val_split(X, y, val_frac, *, seed=None) -> (X_tr, y_tr, X_val, y_val)`:
  - Permute indices once (`np.random.default_rng(seed)`), hold out the last `round(val_frac * N)` rows as validation, the rest as train. `0 <= val_frac < 1`; else `ValueError`. The two index sets are **disjoint** and together cover all `N`. Deterministic for a fixed seed.
- `train_with_loader(model, loader, *, lr, epochs, optimizer=None) -> dict`: thin wrapper that, for each epoch, loops `for X_b, y_b in loader:` and runs `pred = model(X_b)` -> `loss = mse_loss(pred, y_b)` -> `loss.backward()` -> `optimizer.step()` -> `optimizer.zero_grad()` (build `SGD(model.parameters(), lr)` if `optimizer is None`). Return `{"batch_loss": [...], "epoch_loss": [...], "steps": int}` (size-weighted `epoch_loss` per epoch), matching `stage_19`'s `train_minibatch` contract.

**Done when**
- `pytest stage_20_dataloader/test.py` passes.
- `len(loader)` equals `N // B` with `drop_last=True` and `ceil(N / B)` otherwise; with `drop_last=True` every yielded batch has exactly `B` rows.
- Iterating the **same** `DataLoader` twice yields all `N` rows each time; with `shuffle=True` the two epoch orderings differ; with `shuffle=False` they are identical and in `arange` order.
- `train_val_split` index sets are disjoint, cover all `N`, sized `round(val_frac*N)`, and deterministic per seed.
- `train_with_loader` runs `epochs * len(loader)` steps and the epoch loss trends down; central-difference gradcheck: `p.grad` from one batch's `mse_loss(...).backward()` matches `(f(p+eps)-f(p-eps))/(2*eps)` within `atol ~ 1e-5`.
