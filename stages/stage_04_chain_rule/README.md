# Stage 04: The backward pass (chain rule, automated)

**Context** — stage_03 gave every result node a `_backward` closure that pushes the local derivative one edge. A single closure only knows its own op. This stage adds the **global** reverse pass — `backward()` — that runs all the closures in the right order so gradient flows from the output all the way to every leaf. After `loss.backward()`, each node's `.grad` holds `d(loss)/d(node)`. That is reverse-mode automatic differentiation, complete for scalars: the exact `Value.backward` from micrograd, reused unchanged by every later stage.

**Background** — The chain rule says the derivative of a composition multiplies local derivatives along each path, and **sums** over paths when an input feeds several branches:
$$\frac{\partial L}{\partial x} = \sum_{\text{paths } x \to L}\ \prod_{\text{edges on path}} (\text{local derivative}).$$
We compute this **backward** for two reasons. First, the output is the scalar we want gradients *of*, so seeding $\partial L/\partial L = 1$ and pushing toward the inputs reuses shared sub-results instead of recomputing them. Second, going backward each node receives exactly one number — its gradient of the output w.r.t. itself — and its `_backward` closure passes a scaled copy to each input.

The order matters. A node's `_backward` may only run **after** every node that consumes it has already contributed to its `.grad` — otherwise it would push an incomplete gradient. A **topological sort** gives that order: list nodes so every parent comes before the children built from it (DFS post-order over `_prev`); then walk it in **reverse**. Because the stage_03 closures accumulate with `+=`, a node reached by several paths correctly sums its contributions. The whole pass is three lines: `topo = topo_sort(self); self.grad = 1.0; for v in reversed(topo): v._backward()`.

The canonical check: `a=Value(2); b=Value(3); c=a*b; d=c+a; d.backward()` gives `a.grad == 4` (a reaches `d` through `c`, contributing `b=3`, and directly through `+a`, contributing `1`; `3+1=4`) and `b.grad == 2`.

**Watch**
- [What is backpropagation really doing? (3Blue1Brown)](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — visual intuition for chaining local derivatives backward.
- [The spelled-out intro to backpropagation (Andrej Karpathy)](https://www.youtube.com/watch?v=VMj-3S1tku0) — he writes this exact topo-sort + reverse `_backward` loop.

**Cumulative build** — Imports `Value` from `stage_03` via `dlfs.stage_import` and SUBCLASSES it. It inherits `data`/`grad`/`_prev`/`_op`/`_backward` and the gradient-installing `__add__`/`__mul__`/`__pow__` (and the derived ops). It ADDS only the module-level `topo_sort(root)` and the `backward()` method. After this stage `Value` is a complete scalar autodiff engine; stage_05 only adds more primitive ops on top.

**Exercise** — In `code.py`, subclass the `stage_03` `Value`:
- `topo_sort(root)` — return every node reachable from `root` in topological order (dependencies first), each node exactly once, terminating on reused nodes. Use a `visited` set and DFS **post-order** over `_prev` (append a node only after recursing into all its parents).
- `Value.backward(self)` — `topo = topo_sort(self)`, then `self.grad = 1.0`, then `for v in reversed(topo): v._backward()`. Do **not** zero grads first.
- Do not re-implement any op — they are inherited from stage_03. This stage is only the ordering + the reverse walk.
- Allowed: Python stdlib only. NumPy is allowed in the *test* gradcheck helper, never to compute a derivative in `code.py`.

**Done when**
- `pytest stage_04_chain_rule/test.py` passes.
- `topo_sort` lists parents before children, each node once, and terminates on `a * a`.
- `backward()` seeds the output grad to `1.0` and fills `.grad` for every node.
- The canonical case `d = a*b + a` gives `a.grad == 4`, `b.grad == 2` (accumulation over two paths).
- Gradcheck of composite expressions (using `+ * ** /` and constants) matches central differences within `tol = 1e-5`.
