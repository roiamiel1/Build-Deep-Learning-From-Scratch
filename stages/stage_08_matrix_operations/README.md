# Stage 8: Matrix operations

**Context** — In `stage_07` you built `Vec`, a vector of `Value` scalars whose key
reduction is the dot product. A neuron is one dot product; a layer is *many* dot products,
i.e. a matrix multiply. This stage introduces a thin `Mat` container (rows of `Value`s) and
the matrix ops a layer needs — `matmul`, `transpose`, `reshape`, `sum`, `mean` — all built
from the scalar `Value` engine so reverse-mode autodiff still flows through code you wrote.
This is where you internalize the matmul gradient and "dimension analysis" before the fast
NumPy-backed `Tensor` arrives in `stage_09`.

**Background** — A matrix product $C = A B$ with $A\in\mathbb{R}^{m\times k}$,
$B\in\mathbb{R}^{k\times n}$, $C\in\mathbb{R}^{m\times n}$ has entries
$C_{ij}=\sum_{p=1}^{k} A_{ip}B_{pj}$ — exactly a `Vec.dot` per output cell. Because each
$C_{ij}$ is assembled from `Value` `*`/`+`, the `stage_06` chain rule produces the two matmul
gradients automatically. Writing them in matrix form (with upstream $G=\partial L/\partial C$):
$$\frac{\partial L}{\partial A} = G\,B^{\top}, \qquad \frac{\partial L}{\partial B} = A^{\top}G.$$
Verify by **dimension analysis**: $G B^{\top}$ is $(m\times n)(n\times k)=m\times k$ (matches $A$),
$A^{\top}G$ is $(k\times m)(m\times n)=k\times n$ (matches $B$) — if the shapes don't line up, the
formula is wrong. **Transpose** $C=A^{\top}$ is pure rewiring (no arithmetic), so its backward is
$\partial L/\partial A = G^{\top}$. **Reshape** keeps the same `Value`s in new positions, so its
backward reshapes $G$ back to $A$'s shape (local gradient 1 everywhere). The reductions: for
$s=\sum_{ij}A_{ij}$, $\partial L/\partial A_{ij}=1$ (all-ones); for the **mean** over $N=mn$
elements, $\partial L/\partial A_{ij}=1/N$. This extends the `dot`/`sum` gradients and scalar-
broadcast summing from `stage_07`.

**Watch**
- [Backpropagation calculus (3Blue1Brown)](https://www.youtube.com/watch?v=tIeHLnjs5U8) — the chain rule flowing through layers as matrix products; why $B^{\top}$/$A^{\top}$ appear.
- [Matrix multiplication as composition (3Blue1Brown)](https://www.youtube.com/watch?v=XkY2DOUCWMU) — the geometric meaning behind $C_{ij}=\sum_p A_{ip}B_{pj}$.

**Cumulative framework** — `code.py` imports `Value` (`stage_06`) and `Vec` (`stage_07`) via
`dlfs.stage_import` (no redefinition of the scalar engine) and ADDS a new `Mat` class built on
top of them: `matmul`/`@`, `transpose`/`T`, `reshape`, `sum`, `mean`.

**Exercise** — Implement in `code.py`. The scalar engine is reused via `dlfs.stage_import`; you
may also use Python stdlib and NumPy *for forward array creation/shape bookkeeping only* (never
for gradients). No autodiff libraries.

- `class Mat`:
  - `Mat(data)` where `data` is a 2-D iterable of `float`/`int`/`Value`; store `self.data` as a
    `List[List[Value]]` (wrap non-`Value` entries; keep existing `Value`s as-is). Set `self.rows`,
    `self.cols`; expose `self.shape == (rows, cols)`. Unequal row lengths raise `ValueError`.
    `M[i]` returns a row (list of `Value`), so `M[i][j]` is an element; iteration yields rows.
  - `matmul(self, other) -> Mat` and `__matmul__` (the `@` operator): forward
    `C[i][j] = sum_p self[i][p] * other[p][j]`, built from `Value` `*`/`+`. Shape `(m,k)@(k,n)->(m,n)`.
    Inner-dim mismatch (`self.cols != other.rows`) raises `ValueError` reporting both shapes.
  - `transpose(self) -> Mat` and the `T` property: `C[j][i] = self[i][j]`, reusing the same `Value`s.
  - `reshape(self, rows, cols) -> Mat`: row-major flatten then regroup, reusing the same `Value`s;
    size mismatch raises `ValueError`.
  - `sum(self) -> Value`: $\sum_{ij}$ as one `Value`. `mean(self) -> Value`: `sum` scaled by `1/N`.
  - No `Mat.backward`: a `Mat` has no single output. Reduce with `sum`/`mean` to a scalar `Value`,
    then call `.backward()` on it. Document this in the class docstring.
- Ops returning a matrix return a NEW `Mat`; reductions return a `Value`. Do not mutate inputs.
- Acceptance: after `(A @ B).sum().backward()`, the accumulated grads equal $G B^{\top}$ and
  $A^{\top}G$; gradients of matmul, transpose, reshape, sum, mean match central-difference
  numerical gradients within `1e-5` for every element of every input.

**Done when**
- `pytest stage_08_matrix_operations/test.py` passes.
- Matmul/transpose/reshape/sum/mean forward-compute correctly; `dL/dA = G@B.T`, `dL/dB = A.T@G` verified.
- Analytical grads match central-difference gradcheck within `1e-5`; inner-dim mismatch raises `ValueError`.
