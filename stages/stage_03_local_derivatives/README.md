# Stage 03: Local derivatives as `_backward` closures

**Context** — In stage_02 you built a *computational graph*: every result `Value` remembers its op (`_op`), its parents (`_prev`), and reserves a no-op `_backward` hook. This stage fills that hook in. For every op you install a small **closure** on the result node that, given the output's gradient, pushes the *local* derivative onto each input's `grad`. This is the per-edge half of backprop. There is still **no** global pass that runs the closures in order — that is stage_04's `backward()`. Here you only make each edge know how to push gradient one step.

**Background** — A node $z = f(a, b)$ has a *local derivative* with respect to each input, evaluated at the current forward values. By the chain rule, the output's gradient $g = \partial L / \partial z$ flows to each input as $\partial L/\partial a = g \cdot \partial z/\partial a$. So if every op's result node carries a closure that does exactly that push, a later reverse walk just calls the closures in the right order. Derive the rules by hand:
$$z = a + b \;\Rightarrow\; \frac{\partial z}{\partial a} = 1,\quad \frac{\partial z}{\partial b} = 1$$
$$z = a \cdot b \;\Rightarrow\; \frac{\partial z}{\partial a} = b,\quad \frac{\partial z}{\partial b} = a$$
$$z = a^{c}\ (c\ \text{const}) \;\Rightarrow\; \frac{\partial z}{\partial a} = c\,a^{c-1}$$
The closure accumulates with `+=`, never `=`: a node reused in several places (e.g. `a * a`, or `a` feeding two branches) must *sum* the gradient it receives from every consumer. That `+=` is the whole reason the reverse pass is correct for graphs, not just chains.

The asymmetric ops need **no operand-order bookkeeping** on the node. `a - b` is `a + (-b)` and `a / b` is `a * b**-1` — they compose out of `+`, `*`, `**`, which are *inherited from stage_01*. Because each closure **captures its own operands** (`self` and `other`) directly, `a / b` knows which side is the numerator without the node storing order. This is why stage_02's `_prev` could be an unordered set. (Division's $\partial z/\partial a = 1/b$, $\partial z/\partial b = -a/b^2$ falls out automatically from the `*` and `**` closures composed.)

**Watch**
- [The Chain Rule (Backpropagation calculus)](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown: how local derivatives chain edge-by-edge through a graph.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy: he installs exactly these `_backward` closures op by op.

**Cumulative** — Imports `Value` from `stage_02` via `dlfs.stage_import` and SUBCLASSES it. It overrides `__add__`, `__mul__`, `__pow__` so each builds its result node (as in stage_02) **and** sets `out._backward` to the local-derivative closure. The derived ops (`__neg__`/`__sub__`/`__truediv__`, inherited from stage_01) compose out of those three, so they get correct gradients for free. No global `backward()` yet.

**Exercise** — In `code.py`, subclass the `stage_02` `Value` and install `_backward` closures:
- `__add__(self, other)`: coerce a number operand to `Value`; build `out = Value(self.data + other.data, (self, other), '+')`; set `out._backward` to a closure doing `self.grad += out.grad; other.grad += out.grad`.
- `__mul__(self, other)`: build the `'*'` node; closure does `self.grad += other.data * out.grad; other.grad += self.data * out.grad`.
- `__pow__(self, c)`: assert `c` is an int/float constant; build the `f'**{c}'` node; closure does `self.grad += (c * self.data ** (c - 1)) * out.grad`.
- Do **not** re-implement `__neg__`/`__sub__`/`__rsub__`/`__truediv__`/`__rtruediv__` — they are inherited and compose out of the three above, so their gradients flow through your closures.
- `__repr__`: `Value(data=..., grad=...)`.
- The closures must use `+=` (accumulate), never `=`.
- Allowed: Python stdlib only. No NumPy, no autodiff libraries.

**Done when**
- [ ] `pytest stage_03_local_derivatives/test.py` passes.
- [ ] Seeding a result node's `.grad` and calling `node._backward()` once pushes the correct local derivative onto each operand (`+` → `(g, g)`; `*` → `(b·g, a·g)`; `**c` → `c·a^{c-1}·g`).
- [ ] `a * a`'s closure accumulates `2a` onto the shared operand (`+=`, not `=`).
- [ ] A leaf's inherited `_backward` is a no-op that changes no grad.
- [ ] `a ** Value(...)` is rejected (powers must be constant).
