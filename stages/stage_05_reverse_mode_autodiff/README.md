# Stage 5: Reverse-mode autodiff

**Context** — In stage_04 you hand-applied the chain rule through a *fixed* expression graph, manually choosing the order in which to push gradients. That doesn't scale: a real network has thousands of nodes and you can't hand-order them. This stage automates ordering and accumulation so a single `backward()` call computes every gradient. This is the algorithmic heart of every deep-learning framework, and the direct precursor to the `Value` scalar engine in stage_06.

**Background** — A computation graph is a DAG: each `Node` is a scalar value produced by an op (`+`, `*`) over parent nodes. Forward, we evaluate leaves first. To differentiate, we want $\partial L/\partial v$ for *every* node $v$, where $L$ is the output. The multivariable chain rule says a node's gradient is the sum over all consumers $u$ of $v$:
$$\frac{\partial L}{\partial v} = \sum_{u \,:\, v \to u} \frac{\partial L}{\partial u}\,\frac{\partial u}{\partial v}.$$
This requires that *every* consumer $u$ has its gradient finished before we process $v$ — i.e. we must visit nodes in **reverse topological order**. So: (1) topologically sort the graph (DFS post-order, then reverse); (2) **seed** the output with $\partial L/\partial L = 1$; (3) walk in reverse, and at each node call its local backward, which uses the chain-rule terms from stage_04 to **add** (`+=`, never `=`, because a node can feed multiple consumers) the contribution onto each parent's `grad`. For `c = a + b`: $\partial c/\partial a = \partial c/\partial b = 1$. For `c = a * b`: $\partial c/\partial a = b$, $\partial c/\partial b = a$. **Why reverse mode:** one backward pass gives the gradient of one scalar output w.r.t. *all* inputs — exactly the shape of deep learning (scalar loss, millions of params). Forward mode would cost one pass *per input*.

**Watch**
- [The spelled-out intro to neural networks and backpropagation (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this: topo sort, seed grad=1, reverse walk with `+=`.
- [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown on why gradients accumulate over all downstream paths.

**Cumulative build** — This stage `stage_import`s `Value` from `stage_02` (the computation-graph node) and **subclasses** it, adding only the reverse pass: a per-op `_backward` closure (encoding stage_03's local rules), `topo_sort(root)`, and `backward()`. It does not restart the class.

**Exercise** — Extend stage_02's `Value` (imported via `dlfs.stage_import`) into a minimal reverse-mode autodiff engine over scalar nodes. (You may use only Python stdlib; no NumPy needed, no autodiff libraries.)
- `Value.__init__`: call `super().__init__(...)` (inherits `data`, `grad`, `_prev`, `_op` from stage_02) and add `_backward` (a closure, default no-op).
- override `__add__` and `__mul__`: build a result `Value` whose `_backward` closure adds the local-times-incoming gradient (`+=`) onto each parent's `grad`, using the chain rule from stage_04. `__radd__`/`__rmul__` are inherited and route through these, so scalar promotion still works.
- `topo_sort(root) -> list[Value]`: return all nodes reachable from `root` in topological order (parents before children; a node appears only after all its dependencies). Use DFS post-order + reverse, with a `visited` set so shared subgraphs are counted once.
- `backward(self)`: build `topo_sort(self)`, zeroing is assumed already done (do NOT auto-zero other nodes — caller's responsibility), set `self.grad = 1.0`, then iterate the topo order **in reverse** calling each node's `_backward()`.
- Acceptance: gradients match central-difference numerical gradients within tol; shared subexpressions (e.g. `d = a*a` or `e = x + x`) accumulate correctly via `+=`; diamond graphs sum both paths.

**Done when**
- [ ] `pytest stage_05_reverse_mode_autodiff/test.py` passes.
- [ ] `topo_sort` returns each reachable node exactly once, dependencies first.
- [ ] One `backward()` on the output fills `.grad` for every node; analytical grads match central-difference gradcheck within `1e-4`.
- [ ] Reused inputs (`x + x`, `a * a`) and diamond graphs give correctly summed gradients.
