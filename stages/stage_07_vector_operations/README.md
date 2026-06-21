# Stage 7: Vector operations

**Builds on** — imports the finished `Value` from `stage_06` via
`dlfs.stage_import` (it already provides `+`, `*`, `tanh`, `relu`,
`.backward()`); this stage ADDS a `Vec` container (elementwise ops, broadcasting,
`dot`, `sum`, `relu`) on top of it — no scalar autodiff is reimplemented.

**Context** — Real networks operate on vectors, not lone scalars. A thin `Vec`
container holds a list of `Value` scalars and adds the vector ops a neuron needs
— elementwise ops, dot product, and scalar broadcasting — *reusing* the scalar
chain rule from `stage_06`. No NumPy autodiff: every gradient still flows through
`Value`.

**Background** — A vector $\mathbf{x}=(x_1,\dots,x_n)$ is just $n$ scalars
travelling together. Elementwise binary ops apply the matching `Value` op per
index: $(\mathbf{a}+\mathbf{b})_i = a_i + b_i$ and
$(\mathbf{a}\odot\mathbf{b})_i = a_i\, b_i$. The local gradients are the
`stage_06` ones applied per component, so `backward()` handles them
automatically. The **dot product** $s=\mathbf{a}\cdot\mathbf{b}=\sum_{i=1}^{n}
a_i b_i$ is the key reduction; it is the core of every neuron's pre-activation
$w\cdot x + b$. Built from `*` and `+` of `Value`s, its gradients fall out of
the scalar engine:
$$\frac{\partial s}{\partial a_i} = b_i,\qquad \frac{\partial s}{\partial b_i} = a_i.$$
**Scalar broadcasting** combines a vector with a single number/`Value`:
$(\mathbf{a}+c)_i = a_i + c$, so $\partial/\partial c = \sum_i 1$ — the broadcast
scalar's gradient is the *sum* of the per-element grads. This is automatic
because the one `Value` $c$ is reused in $n$ children and `stage_06`'s `+=`
accumulation sums them. That summing-on-broadcast rule is the seed of full
tensor broadcasting in `stage_12`. The `sum()` reduction $\big(\sum_i a_i\big)$
has gradient $1$ to every element.

**Watch**
- [The spelled-out intro to neural networks and backpropagation (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds the exact `Value` engine you extend; watch the `backward()` accumulation part.
- [Vectors, what even are they? (Essence of Linear Algebra)](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown on vectors and the dot product geometry behind the reduction above.

**Exercise** — Implement `Vec` in `code.py`. `Value` is already imported from
`stage_06` for you (do not redefine it). You may use Python stdlib and NumPy
*for forward array creation only* (never for gradients).

- `class Vec`:
  - `Vec(data)` where `data` is an iterable of `float`/`int`/`Value`; store
    `self.data` as a Python `list` of `Value` (wrap non-`Value` entries).
    `len(v)`, `v[i]` (returns a `Value`), and iteration must work.
  - `__add__`, `__sub__`, `__mul__` (elementwise, Hadamard) accepting another
    `Vec` of equal length **or** a scalar (`int`/`float`/`Value`) that
    broadcasts to every element. Support reflected ops (`scalar + vec`).
    Mismatched non-scalar lengths raise `ValueError`.
  - `dot(self, other) -> Value`: the scalar $\sum_i a_i b_i$ built from `Value`
    `*`/`+` (so it is differentiable). Length mismatch raises `ValueError`.
  - `sum(self) -> Value`: returns $\sum_i a_i$ as a single `Value`.
  - `relu(self) -> Vec`: elementwise, using the imported `Value.relu()`:
    $\mathrm{relu}(x)=\max(0,x)$, gradient $1$ if $x>0$ else $0$.
  - No `Vec.backward`: a `Vec` has no single output. Reduce with `dot`/`sum` to
    a scalar `Value`, then call `.backward()` on that. Document this in the
    class docstring.
  - After backward through e.g. `a.dot(b)`, each `a[i].grad` and `b[i].grad`
    must be correct.
- Ops returning a vector return a new `Vec`; reductions (`dot`, `sum`) return a
  `Value`. Do not mutate inputs.
- Acceptance: gradients of `dot`, `sum`, elementwise ops, scalar broadcasting,
  and `relu` match central-difference numerical gradients within tol.

**Done when**
- `pytest stage_07_vector_operations/test.py` passes.
- Elementwise add/sub/mul, `dot`, `sum`, `relu`, and scalar broadcasting all forward-compute correctly.
- Analytical grads match central-difference gradcheck within `1e-5` for every element of every input.
- Broadcasting a single scalar `Value` across a `Vec` accumulates the summed gradient onto that scalar.
