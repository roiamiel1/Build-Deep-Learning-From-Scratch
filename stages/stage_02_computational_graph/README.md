# Stage 02: Computational graph

**Context** — In `stage_01` you built `Value`, a thin wrapper around a single scalar that supports `+` and `*`. Right now a `Value` knows only its own number — once an operation produces a result, the inputs that created it are forgotten. This stage makes every operation *record* where its result came from, so a chain of operations forms a directed acyclic graph (DAG). That recorded graph is the skeleton that automatic differentiation (stage 04+) will walk backward over.

**Background** — A scalar expression like $f = (a + b) \cdot c$ is really a tree of operations: a `+` node feeding a `*` node. To differentiate it automatically we must remember that structure. We extend `Value` so each node stores three things: `_prev`, the **set of input `Value`s** that produced it; `_op`, a **string label** for the operation (`'+'`, `'*'`, `''` for leaves); and `_backward`, a **closure hook** that is a no-op for now (stage_03 installs the real per-op gradient rule on it). Leaves (numbers you create directly) have an empty `_prev` and empty `_op`. Every operator returns a *new* `Value` whose `_prev` points back at its operands, so following `_prev` from any node enumerates the subgraph that built it. We use a **set** for `_prev` so a reused operand (`a * a`) is a single parent and the DAG walk terminates; we do **not** store operand order on the node, because the gradient closures added in stage_03 capture each operand directly — so `a - b` and `a / b` will know which side is which without the node remembering order. This is purely bookkeeping: forward arithmetic is unchanged from stage 01, and `grad` still defaults to `0.0`. The reverse pass that will use these edges relies on the chain rule, e.g. for $L$ depending on $u = a + b$,
$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial u}\cdot\frac{\partial u}{\partial a}, \qquad \frac{\partial u}{\partial a} = 1,$$
and for $v = a \cdot b$, $\frac{\partial v}{\partial a} = b$ and $\frac{\partial v}{\partial b} = a$. We do **not** compute gradients yet — we only build the graph those local derivatives will later flow through. The one structural rule that matters: building the graph must be acyclic. Reusing a node (e.g. `a * a`) is fine and must not duplicate it in `_prev` (hence a *set*).

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this `Value` graph; watch the first ~25 min where `_prev`/`_op` are introduced and the graph is visualized.
- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown's mental model of computation flowing through a graph; motivates why we record structure now.

**Cumulative** — This stage imports `Value` via `stage_import("stage_01", "Value")` and SUBCLASSES it (not a rewrite). It ADDS: an `__init__` override that records graph provenance (`_prev` set, `_op`, and a no-op `_backward` hook) on top of stage_01's `data`/`grad`, graph-aware `__add__`/`__mul__` overrides (so the whole DAG is built from this stage's `Value`), a graph-aware `__repr__`, and a module-level `trace(root)` to walk the DAG.

**Exercise** — Extend `Value` from `stage_01` in `code.py` so it records its computational graph. Add nothing that computes gradients yet.
- `__radd__`, `__rmul__` are INHERITED from stage_01; do not re-add them.
- `__init__(self, data, _children=(), _op='')`: OVERRIDE stage_01's `__init__`. Call `Stage1_Value.__init__(self, data)` to set `data`/`grad`, then store `self._prev = set(_children)`, `self._op = _op`, and `self._backward = lambda: None` (a no-op hook stage_03 will replace per op). stage_01's `__init__` takes only `data`, so this override is what introduces the graph fields.
- `__add__(self, other)`: accept a `Value` or a Python number; if `other` is a number, wrap it as `Value(other)`. Return `Value(self.data + other.data, (self, other), '+')` — so `_prev == {self, other}` and `_op == '+'`; construct THIS subclass so the whole DAG is uniform.
- `__mul__(self, other)`: same number-coercion rule. Return `Value(self.data * other.data, (self, other), '*')`.
- `__repr__(self)`: return `f"Value(data={self.data}, op={self._op!r})"` (surfaces the `_op` label).
- Provide a free function `trace(root)` that walks `_prev` from `root` and returns `(nodes, edges)`: `nodes` a set of all `Value`s reachable from `root`, `edges` a set of `(parent, child)` tuples. It must not revisit nodes (use a visited set) so it terminates even though the graph is a DAG.
- Forward values must equal stage 01's results exactly; only graph metadata is new.
- Allowed tools: Python stdlib only. No NumPy needed here; no autodiff libraries.

**Done when**
- [ ] `pytest stage_02_computational_graph/test.py` passes.
- [ ] Leaves have `_prev == set()` and `_op == ''`; `+`/`*` results have the right two parents and correct `_op`.
- [ ] Every node has a callable `_backward`; a leaf's is a no-op that changes no grad.
- [ ] `a * a` produces a node with a single parent in `_prev` (set dedup) and `trace` terminates with no duplicates.
- [ ] `trace(root)` returns every node and every parent→child edge exactly once.
