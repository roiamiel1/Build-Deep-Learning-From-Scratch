# Stage 05: Scalar backprop engine

**Context** — This is the payoff of stages 1-4. Until now you built `Value` with forward arithmetic (stage_01), recorded the computational graph (stage_02), installed per-op `_backward` closures (stage_03), and added the global `backward()` reverse pass (stage_04). At the end of stage_04 you already have a complete scalar autodiff engine. This stage **extends** it with the remaining primitive ops a neural net needs (`tanh`, `exp`, `relu`) so the same `.backward()` differentiates them too. This `Value` is the seed of the whole curriculum; the `Tensor` engine in later stages is its N-dimensional generalization.

Stages 01-05 together reimplement Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) `Value` class from scratch — the same engine, built up one concept per stage instead of all at once. After this stage your `Value` is feature-equivalent to `micrograd/engine.py`.

**Background** — A `Value` wraps a scalar `data` and a `grad` (initialized to 0). Every operation that creates a `Value` records its inputs (`_prev`) and a local closure `_backward` that knows how to push gradient from the output back to those inputs. Reverse-mode autodiff is: set the output's grad to 1, walk the graph in **reverse topological order** (stage_04), and at each node call its `_backward`. The chain rule says, for output $L$ and an intermediate $v$ with children $c_i$, $\frac{\partial L}{\partial v} = \sum_i \frac{\partial L}{\partial c_i}\frac{\partial c_i}{\partial v}$. The **sum** is critical: a node reused in several places must *accumulate* (`+=`) gradient from every consumer — this is why we never overwrite `grad`. The `+`, `*`, and `**` rules come inherited from stages 03/04; the NEW local rules you add here: for $t=\tanh(x)$, $\frac{dt}{dx}=1-t^2$; for $e=\exp(x)$, $\frac{de}{dx}=e$; for $r=\mathrm{ReLU}(x)$, $\frac{dr}{dx}=\mathbb{1}[x>0]$. The canonical test (already passing after stage_04): `a=Value(2); b=Value(3); c=a*b; d=c+a; d.backward()` gives `a.grad == 4` (one path through `c` gives `b=3`, the direct path `d=c+a` gives `1`, summed = 4) and `b.grad == 2`.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this `Value` engine and `.backward()` from scratch; the reference for this stage.
- [Backpropagation, intuitively | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown on what backprop is actually doing to the graph.

**Cumulative chain** — This stage `stage_import`s `Value` from **stage_04** and SUBCLASSES it. The constructor, the differentiable `__add__`/`__mul__`/`__pow__` (with their `_backward` closures), the reflected `__radd__`/`__rmul__`, `backward()` (topo sort + reverse accumulation), and the derived `__neg__`/`__sub__`/`__rsub__`/`__truediv__` are all INHERITED. This stage ADDS only the remaining primitive ops with their local rules (`tanh`, `exp`, `relu`) plus a grad-aware `__repr__`.

**Exercise** — In `code.py`, complete the `Value` subclass. You may use ONLY Python stdlib (NumPy/Matplotlib allowed but unnecessary here; NO autodiff libs).
- ADD these new ops, each returning a `Value` whose `_backward` accumulates into inputs with `+=`:
  - `tanh(self)`, `exp(self)`, `relu(self)`.
- Thin `__add__`/`__mul__`/`__pow__` overrides: delegate to `super()` (stage_04's math, which already installs the gradient closures) and only re-bless the result as a stage-06 `Value` so the new unary ops chain on intermediates — do NOT re-derive gradients.
- `__repr__`: e.g. `Value(data=2.0, grad=4.0)`.
- Acceptance: every new `_backward` uses `+=` (never `=`); reused nodes (like `a` in the canonical example) get summed gradients.

**Done when**
- `pytest stage_05_backprop_engine/test.py` passes.
- The canonical `d.backward()` example yields `a.grad == 4.0`, `b.grad == 2.0`.
- Analytical grads from `.backward()` match central-difference numerical grads within `tol=1e-5` for every op (`+`, `*`, `pow` inherited; `tanh`, `exp`, `relu` new) and for composite expressions with reused nodes.
