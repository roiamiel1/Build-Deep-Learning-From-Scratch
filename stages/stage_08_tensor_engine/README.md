# Stage 08: Tensor engine

This is THE core engine of the whole course. Every later stage — dense layers, optimizers, CNNs,
Transformers — imports the `Tensor` you build here. Nothing earlier defines a `Tensor`; you write it
fresh.

## 1. Why the old `Value` engine is too slow

Stages 01-07 built a **scalar** autodiff engine. A `Value` node holds **one number** and **one
gradient**. `Vec` (stage_06) and `Mat` (stage_07) are just containers: a `Mat` is a grid of `Value`
objects, and `Mat @ Mat` is a Python triple-loop that calls scalar `Value.__mul__` / `__add__` on
every pair. The autodiff is correct — but the cost is brutal.

Do the arithmetic. A single `100×100 @ 100×100` matmul computes $100 \times 100 \times 100 = 10^6$
scalar multiply-adds. The `Mat` version therefore builds **~one million `Value` objects** and **~one
million `_backward` Python closures**, each a separate heap allocation, walked **one at a time** by
the interpreter. Per scalar op Python must: check types, box/unbox floats, dispatch a method, build a
closure. One real training step does thousands of such matmuls. It is unusably slow.

The fix is to **stop making one node per number.** A `Tensor` is one node that holds a whole NumPy
**array** as its data and a same-shape array as its gradient. The forward math is one NumPy call —
which runs as a compiled C loop over a contiguous, fixed-`float64` buffer, applying one CPU
instruction to many entries at once (SIMD) and streaming cache-friendly memory. Same math, **~$10^6$×
fewer Python objects** on that matmul, and the heavy arithmetic leaves the interpreter entirely. The
*identical* array code later runs on a GPU (CuPy/PyTorch) unchanged — a per-scalar graph never can.

`Vec`/`Mat` were never wasted: they were the scaffolding that made you derive the gradient rules by
hand, in terms of scalars you already understood. Now they retire. The scalar `Value` survives only as
a 0-d bridge (`Tensor.from_value`); `Vec`/`Mat` are not imported by any later stage.

## 2. The math: backprop on arrays

A `Tensor` node holds an array $X$ and a same-shape gradient array $\partial L/\partial X$ — **one
gradient number per entry of $X$**. The chain rule does not change; it runs on every entry at once.

### Elementwise ops are easy — entries don't mix

For an elementwise op $Z = f(X)$, each output entry depends on **only its own** input entry. Example
$Z = X^2$: entry $z_{ij} = x_{ij}^2$ uses only $x_{ij}$, so its local derivative is the plain scalar
rule $2x_{ij}$. With upstream gradient $g_{ij} = \partial L/\partial z_{ij}$:

$$
\frac{\partial L}{\partial x_{ij}} = g_{ij}\cdot 2x_{ij}.
$$

This holds independently for every entry, so all of it is **one array line** —
`self.grad += out.grad * (2 * self.data)` — no loop. General form:
$\partial L/\partial X = g \odot f'(X)$, "multiply the upstream gradient by the local derivative." Every
elementwise op in this stage (`+`, `*`, `**`, `relu`, `tanh`, `exp`, `log`) is this one pattern.

### Matrix multiply: the derivation of $\partial L/\partial A = G B^{\top}$

Matmul is the one op where entries **do** mix, so it needs its own rule. Derive it from scratch — this
is the heart of the stage.

**Setup.** $Z = A B$ with shapes $A:(m,n)$, $B:(n,p)$, so $Z:(m,p)$. From the next layer we receive
$G = \partial L/\partial Z$, shape $(m,p)$ — one upstream gradient per entry of $Z$. We want
$\partial L/\partial A$, which must come out shape $(m,n)$ (a gradient is always the shape of the thing
it differentiates).

**One output entry.** By the definition of matrix multiply,

$$
z_{ij} = \sum_{k} a_{ik}\, b_{kj}\quad(\text{row } i \text{ of } A \text{ dotted with column } j \text{ of } B).
$$

**Pick one weight $a_{ik}$ and ask how it moves the loss.** It can only reach $L$ through the outputs
$z$, so chain-rule over every output:

$$
\frac{\partial L}{\partial a_{ik}} = \sum_{i'\,j} \underbrace{\frac{\partial L}{\partial z_{i'j}}}_{g_{i'j}}\cdot \frac{\partial z_{i'j}}{\partial a_{ik}}.
$$

**Which outputs does $a_{ik}$ actually touch?** In $z_{i'j} = \sum_k a_{i'k} b_{kj}$, the symbol
$a_{ik}$ appears only when the row matches, $i' = i$. There it multiplies $b_{kj}$. So

$$
\frac{\partial z_{i'j}}{\partial a_{ik}} = \begin{cases} b_{kj} & i' = i\\[2pt] 0 & i' \neq i.\end{cases}
$$

That is: $a_{ik}$ feeds the **whole row $i$** of $Z$ (all columns $j$), each scaled by $b_{kj}$.

**Drop the zero terms** ($i'\neq i$ vanish):

$$
\frac{\partial L}{\partial a_{ik}} = \sum_{j} g_{ij}\, b_{kj}.
$$

**Recognize the sum as a matrix product.** $\sum_j g_{ij} b_{kj}$ is row $i$ of $G$ dotted with row $k$
of $B$ — i.e. row $i$ of $G$ times **column $k$ of $B^{\top}$** — which is exactly the $(i,k)$ entry of
$G B^{\top}$. Therefore

$$
\boxed{\ \frac{\partial L}{\partial A} = G\,B^{\top}\ }
$$

**Shape check.** $G:(m,p)$ times $B^{\top}:(p,n)$ gives $(m,n)$ = shape of $A$. ✓

**Same argument for $B$.** A weight $b_{kj}$ feeds the whole **column $j$** of $Z$ (all rows $i$), with
$\partial z_{ij}/\partial b_{kj} = a_{ik}$, so $\partial L/\partial b_{kj} = \sum_i a_{ik}\,g_{ij} =
\sum_i (A^{\top})_{ki}\, g_{ij}$, which is

$$
\boxed{\ \frac{\partial L}{\partial B} = A^{\top} G\ }\qquad\text{shape } (n,m)\times(m,p)=(n,p)=\text{shape of }B.\ \checkmark
$$

**The shape-matching shortcut.** Once you know the answer is $G$, $A$, $B$ combined with transposes,
only **one** arrangement produces the correct output shape — so the shapes alone pin the formula. The
operand that was on the **left** ($A$) keeps the upstream $G$ on its left ($A^{\top}G$); the operand on
the **right** ($B$) keeps $G$ on its right ($G B^{\top}$). Useful memory hook, but you should be able to
re-derive it from the sum above.

## 3. Why matrix multiply is THE operation that matters

This is not one op among many — it is the workhorse of every neural net, which is why it earns a
hand-derived gradient:

- **A dense/linear layer IS a matmul.** A layer computing outputs from inputs is $Y = X W$ (plus
  bias): inputs $X$ times a weight matrix $W$. That is the bulk of the compute in an MLP.
- **Attention is matmuls.** A Transformer's core ($QK^{\top}$, then times $V$) is two matmuls.
- **Convolutions reduce to matmul** (im2col) on most hardware.
- **It is where the FLOPs and the parameters live.** Elementwise ops (`relu`, `add`) are cheap glue;
  the matmuls are where nearly all the multiply-adds — and all the GPU speedup — happen.

So getting $\partial L/\partial A = G B^{\top}$ and $\partial L/\partial B = A^{\top}G$ right, fast, and
vectorized is what makes everything downstream trainable at a usable speed.

## 4. Your mission

In `code.py`, implement **one class `Tensor`** — an N-dimensional reverse-mode autodiff node backed by
NumPy. Every later stage imports it.

Every differentiable op has the **same five-line shape**; learn it once. Example for multiply:

```python
def __mul__(self, other):
    other = self._coerce(other)                # 1. wrap raw numbers/arrays as Tensors
    out = self._make_tensor(self.data * other.data, # 2. forward: NumPy does the array math
                 _prev=(self, other), _op="*") # 3. record parents + op name for the graph
    def _backward():                           # 4. local gradient → each parent
        Tensor._accumulate(self,  out.grad * other.data)   # dL/dself = g ⊙ other
        Tensor._accumulate(other, out.grad * self.data)    # dL/dother = g ⊙ self
    out._backward = _backward                  # 5. attach; backward() calls it later
    return out
```

Only line 2 (forward formula) and line 4 (local gradient) change per op. `_accumulate` routes the
gradient to a parent — for an equal-shaped operand it is just `parent.grad += incoming`; it only
sums-to-scalar when the operand is a 0-d constant.

**Implement:**

- **Fields**: `data` (np.ndarray float64), `grad` (zeros, same shape), `_prev` (tuple of `Tensor`),
  `_op` (str), `_backward` (callable, no-op for leaves). Names mirror stage_05's `Value`.
- **Construction**: `Tensor(data)` accepts scalar / list / ndarray; store `np.asarray(data, dtype=np.float64)`.
- **Operators** (each returns a new `Tensor`, sets `_prev`/`_op`, defines `_backward` via `_accumulate`):
  `__add__`, `__mul__`, their `__r*__` reflections, `__neg__`, `__sub__`, `__rsub__`; `__pow__(self, c)`
  for a numeric constant `c`; `__truediv__`/`__rtruediv__` built from `*` and `**` (no new backward).
  A `_coerce` helper wraps raw operands so `t + 2.0` and `2.0 * t` work.
- **Elementwise methods**: `relu`, `tanh`, `exp`, `log` — each its own `_backward` using the rules in §2
  ($g\odot\mathbf 1[x>0]$, $g\odot(1-z^2)$, $g\odot z$, $g\oslash x$).
- **Reshape**: `reshape(*shape)` — pure rearrangement, no entry created/combined, so its `_backward` is
  just the **inverse reshape**: forward `self.data.reshape(shape)`; backward `self.grad += out.grad.reshape(self.data.shape)`.
  Accept varargs (`t.reshape(2, 3)`) or a single tuple (`t.reshape((2, 3))`), and a `-1` placeholder
  NumPy infers (`t.reshape(-1)` flattens). Later stages reshape between conv feature maps and dense layers.
- **Matmul**: `__matmul__` (the `@` operator) pushing `G @ B.T` to the left operand and `A.T @ G` to the
  right — the rule you derived in §2. That rule is exact for 2-D operands; also handle the 1-D forms the
  neuron / dense layer use (`(n,)@(n,)`→scalar, `(n,)@(n,m)`→`(m,)`, batched `(b,n)@(n,)`→`(b,)`). For a
  1-D operand the bare formula is wrong — `.T` is a no-op on 1-D and the true gradient is an outer
  product — so **promote each 1-D operand to 2-D** (left→`(1,n)` row, right→`(n,1)` column), apply the
  `G@B.T` / `A.T@G` rule, then squeeze the inserted axis back so each grad matches its operand's shape.
- **Autodiff**: `backward()` — (1) topo-sort via DFS over `_prev`, (2) seed `self.grad = np.ones_like(self.data)`,
  (3) walk reversed topo order calling each `_backward`. Gradients **accumulate** with `+=` so a tensor
  reused on several paths (e.g. `y = x*x + x`) sums its contributions. `zero_grad()` resets `grad`.

**Scope this stage:** operands are **equal-shaped** (a Python/0-d numeric constant is fine; two
`Tensor`s must share shape) plus the `@` matmul above. General broadcasting-gradient reduction is
stage_11; `sum`/`mean` reductions are stage_12.

**Allowed tools:** Python stdlib + NumPy only. NumPy is for forward array math and storage — you write
every gradient yourself; never call a NumPy autograd/`grad` helper.

**Done when:**

- `pytest stage_08_tensor_engine/test.py` passes.
- Central-difference gradcheck on `+`, `*`, `**`, `-`, `/`, `relu`, `tanh`, `exp`, `log`, `@` matches
  analytic `grad` within `1e-6` (elementwise gradcheck on 0-d tensors — no `.sum()` until stage_12;
  matmul gradcheck seeds the output grad with ones; tolerance relaxed near ReLU kinks).
- A reused tensor accumulates gradient from every path; `backward()` seeds with `ones_like` and fills
  `.grad` for every leaf; `data`/`grad` stay `float64` ndarrays of identical shape.

## 5. Watch — the matrix-derivative math

- [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown: what the gradient flowing backward through a layer actually means.
- [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown: the chain rule worked out in the layer / matrix form your `_backward` methods implement.
- [Stanford CS231n, Lecture 6: Backpropagation](https://www.youtube.com/watch?v=dB-u77Y5a6A) — derives the **vectorized / matrix** gradients ($G B^{\top}$, $A^{\top}G$) exactly as in §2, with the shape-matching trick.
