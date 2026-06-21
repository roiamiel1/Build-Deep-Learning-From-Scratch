# Stage 3: Local derivatives

**Context** — In stage_02 you built a *computational graph*: every expression became a DAG of `Value` nodes, each node remembering its op (`_op`) and its input nodes. That graph stored only forward *values*. This stage attaches the missing half of backprop: for every op, the **local derivative** of the node's output with respect to each of its inputs. These local derivatives are the per-edge factors the chain rule will later multiply together (stage_04 manual backprop, stage_06 the `Value` engine). No global backward pass yet — just one honest derivative per edge.

**Background** — A node $z = f(a, b)$ has a *local derivative* with respect to each input, evaluated at the current forward values. By the chain rule, an upstream gradient $\bar z = \partial L / \partial z$ flows to each input as $\bar a = \bar z \cdot \partial z / \partial a$. So if we can produce $\partial z / \partial a$ for every op, backprop is just "multiply and accumulate." Derive the three core rules by hand:
$$z = a + b \;\Rightarrow\; \frac{\partial z}{\partial a} = 1,\quad \frac{\partial z}{\partial b} = 1$$
$$z = a \cdot b \;\Rightarrow\; \frac{\partial z}{\partial a} = b,\quad \frac{\partial z}{\partial b} = a$$
$$z = a / b \;\Rightarrow\; \frac{\partial z}{\partial a} = \frac{1}{b},\quad \frac{\partial z}{\partial b} = -\frac{a}{b^{2}}$$
Note division is *not* symmetric, and every local derivative is evaluated at the stored forward values of `a` and `b` — that is why stage_02 kept the values around. Also handle the unary negation $z = -a \Rightarrow \partial z/\partial a = 1\cdot(-1) = -1$ and subtraction $z = a - b \Rightarrow (1, -1)$, since they reduce to add/mul. Local derivatives are purely *local*: they ignore the rest of the graph, which is exactly what makes the chain rule modular.

**Watch**
- [The Chain Rule (Backpropagation calculus)](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown: how local derivatives chain edge-by-edge through a graph.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy: the same per-op local-derivative idea you are encoding here.

**Cumulative** — Imports `Value` from `stage_02` via `dlfs.stage_import` (no new `Value` class); ADDS the per-op local-derivative functions and the `local_derivatives(node)` dispatcher that reads `node._op` on top of it.

**Exercise** — In `code.py`, build on the imported `stage_02` `Value` graph. Implement local-derivative functions, one per op, each returning the partials **with respect to every input**, evaluated at the inputs' forward values:
- `d_add(a, b) -> tuple[float, float]` returns `(1.0, 1.0)`.
- `d_sub(a, b) -> tuple[float, float]` returns `(1.0, -1.0)`.
- `d_mul(a, b) -> tuple[float, float]` returns `(b, a)`.
- `d_div(a, b) -> tuple[float, float]` returns `(1/b, -a/b**2)`; raise `ZeroDivisionError` if `b == 0`.
- `d_neg(a) -> tuple[float]` returns `(-1.0,)`.
- `local_derivatives(node) -> tuple[float, ...]` — the dispatcher: read `node._op` (the stage_02 `Value` op label), pull each input's stored forward value (`.data`), call the matching `d_*`, and return the tuple of local derivatives in operand order. Constant/leaf nodes (op `""`) return `()`.
- Inputs are Python floats (or the forward values held by `Value`s); outputs are plain `float`s in a tuple, one per input, in input order.
- Allowed: Python stdlib only. No NumPy needed; no autodiff libraries.
- Acceptance: each `d_*` matches its hand-derived rule, `local_derivatives` dispatches correctly for `+ - * / neg` and returns `()` for leaves, and every analytical partial agrees with a central-difference estimate.

**Done when**
- [ ] `pytest stage_03_local_derivatives/test.py` passes.
- [ ] All five `d_*` functions return the hand-derived partials in input order.
- [ ] `local_derivatives` dispatches on `node._op` and returns `()` for leaf nodes.
- [ ] `d_div(a, 0)` raises `ZeroDivisionError`.
- [ ] Every analytical partial matches central differences within `tol = 1e-6`.
