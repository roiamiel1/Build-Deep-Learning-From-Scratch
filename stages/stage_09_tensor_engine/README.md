# Stage 09: Tensor engine

**Context** — This is THE core engine of the whole course. So far (stages 06-08) you built a
scalar autodiff engine (`Value` from stage_06) and wired thousands of scalars into a neuron, layer,
and MLP (stage_08). Scalar graphs are correct but absurdly slow. Here you collapse the same
reverse-mode autodiff idea onto a single class — `Tensor` — that stores an N-dimensional NumPy array
and tracks one gradient array per node. Every later stage (layers, optimizers, CNNs, Transformers)
imports this `Tensor`.

**Background** — A `Tensor` wraps `data` (a `np.ndarray`) and carries `grad` (same shape, the
accumulated $\partial L / \partial \text{self}$). Each non-leaf tensor remembers the `_prev` tensors it
was built from, the `_op` name, and a local `_backward` closure that knows how to push gradient to
those parents. This is identical in spirit to `Value`, but the chain rule now runs on whole arrays at
once. `.backward()` builds a topological order of the graph (the algorithm from stage_05), seeds the
output gradient with ones, and walks nodes in reverse, calling each `_backward`. Gradients
**accumulate** with `+=` because a tensor used in several places receives gradient from each.

For an op $z = f(x, y)$ with upstream gradient $g = \partial L/\partial z$, each parent receives the
vector-Jacobian product. The elementwise ops you implement here:

$$z = x + y:\quad \frac{\partial L}{\partial x} = g,\quad \frac{\partial L}{\partial y} = g$$
$$z = x \odot y:\quad \frac{\partial L}{\partial x} = g \odot y,\quad \frac{\partial L}{\partial y} = g \odot x$$
$$z = x^{c}\ (c\ \text{const}):\quad \frac{\partial L}{\partial x} = g \odot c\, x^{c-1}$$
$$z = \mathrm{ReLU}(x):\quad \frac{\partial L}{\partial x} = g \odot \mathbf{1}[x > 0]$$

Defer general broadcasting gradient reduction to stage_12; keep this stage to **equal-shaped**
operands (a scalar Python constant is fine, but two `Tensor`s must share shape).

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds the exact engine you are now lifting to arrays; watch the `_backward`/topo-sort part.
- [What is automatic differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY) — 3Blue1Brown-style intuition for reverse-mode AD on a graph.

**Cumulative chain** — This is the one new *core* the curriculum adds: no earlier stage defines a
`Tensor`, so it is written fresh (not subclassed). Why the abstraction changes here: the scalar
`Value`/`Vec`/`Mat` graphs from stages 06-08 are one node *per number* — correct but unusably slow —
so we collapse the identical reverse-mode rule onto one NumPy array (and one gradient array) per node.
`code.py` still imports the scalar engine via `Stage6_Value = stage_import("stage_06", "Value")` and
bridges it in via `Tensor.from_value(v)` (lifts a scalar `Value` to a 0-d `Tensor` leaf), making the unification literal
rather than copy-pasted. Every later stage imports THIS `Tensor`.

**Exercise** — In `code.py`, implement a single class `Tensor`:
- **Fields**: `data` (np.ndarray, float64), `grad` (np.ndarray of zeros, same shape), `_prev`
  (tuple of `Tensor`), `_op` (str, e.g. `""`, `"+"`, `"*"`, `"**"`, `"relu"`), and a private
  `_backward` callable (defaults to a no-op). (Field names mirror stage_06's `Value`.)
- **Construction**: `Tensor(data)` accepts a scalar, list, or ndarray; store as `np.asarray(..., dtype=np.float64)`.
- **Operator overloading** (each returns a new `Tensor`, sets its `parents`/`operation`, and defines
  `_backward` that does `parent.grad += ...`):
  - `__add__`, `__mul__` (and `__radd__`, `__rmul__`), `__neg__`, `__sub__`, `__rsub__`.
  - `__pow__(self, c)` for a numeric constant `c` only.
  - `__truediv__`, `__rtruediv__` implemented via `*` and `**` (no new backward needed).
  - Wrap raw Python/NumPy operands with a helper so `t + 2.0` and `2.0 * t` work.
  - Route `+`/`*` backward through a tiny `_accumulate` helper that sums the
    gradient to a scalar when the operand is a 0-d numeric constant (the only
    broadcast case allowed now); equal-shaped `Tensor`s just `+=`. Full
    broadcasting-gradient reduction is stage_12.
- **Methods**: `relu(self)`; `backward(self)` which (1) topo-sorts via DFS over `_prev`, (2) sets
  `self.grad = np.ones_like(self.data)`, (3) iterates reversed topo order calling `_backward`.
- **Repr**: `Tensor(data=..., grad=...)`.
- Allowed tools: Python stdlib + NumPy only. NumPy is for forward array math and storing
  arrays — never call any autodiff/`grad` helper.

**Done when**
- `pytest stage_09_tensor_engine/test.py` passes.
- Central-difference gradcheck on `+`, `*`, `**`, `-`, `/`, and `relu` matches analytic `grad`
  within `1e-6` (gradcheck runs on 0-d tensors since there is no `.sum()` yet — that is stage_13;
  tol relaxed near ReLU kinks).
- A reused tensor (e.g. `y = x*x + x`) accumulates gradient from every path.
- `backward()` runs once, seeds the output with `ones_like`, and fills `.grad` for every leaf.
- `data`/`grad` are always `float64` ndarrays of identical shape; `zero_grad()` resets `grad`.
