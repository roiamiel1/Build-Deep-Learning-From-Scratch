# Stage 6: Scalar backprop engine

**Context** — This is the payoff of stages 1-5. Until now you computed derivatives numerically (stage_01), composed them by the chain rule (stage_02), built expressions as DAGs (stage_03), hand-coded reverse passes (stage_04), and topologically ordered nodes (stage_05). Now you package all of it into one self-contained class, `Value`, whose `.backward()` runs reverse-mode autodiff automatically. This `Value` is the seed of the whole curriculum; the `Tensor` engine in later stages is its N-dimensional generalization.

**Background** — A `Value` wraps a scalar `data` and a `grad` (initialized to 0). Every operation that creates a `Value` records its inputs (`_prev`) and a local closure `_backward` that knows how to push gradient from the output back to those inputs. Reverse-mode autodiff is: set the output's grad to 1, walk the graph in **reverse topological order** (stage_05), and at each node call its `_backward`. The chain rule (stage_02) says, for output $L$ and an intermediate $v$ with children $c_i$, $\frac{\partial L}{\partial v} = \sum_i \frac{\partial L}{\partial c_i}\frac{\partial c_i}{\partial v}$. The **sum** is critical: a node reused in several places must *accumulate* (`+=`) gradient from every consumer — this is why we never overwrite `grad`. The `+` rule ($\frac{\partial}{\partial a}(a+b)=1$) and `*` rule ($\frac{\partial}{\partial a}(a\cdot b)=b$) come inherited from stage_05; the NEW local rules you add here: for $t=\tanh(x)$, $\frac{dt}{dx}=1-t^2$; for $e=\exp(x)$, $\frac{de}{dx}=e$; for $r=\mathrm{ReLU}(x)$, $\frac{dr}{dx}=\mathbb{1}[x>0]$; for $p=x^{n}$ ($n$ constant), $\frac{dp}{dx}=n\,x^{n-1}$. The canonical test: `a=Value(2); b=Value(3); c=a*b; d=c+a; d.backward()` gives `a.grad == 4` (one path through `c` gives `b=3`, the direct path `d=c+a` gives `1`, summed = 4) and `b.grad == 2`.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this `Value` engine and `.backward()` from scratch; the reference for this stage.
- [Backpropagation, intuitively | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown on what backprop is actually doing to the graph.

**Cumulative chain** — This stage `stage_import`s `Value` from **stage_05** and SUBCLASSES it. The constructor, the differentiable `__add__`/`__mul__`, the reflected `__radd__`/`__rmul__`, and `backward()` (topo sort + reverse accumulation) are all INHERITED — so are the derived `__neg__`/`__sub__`/`__rsub__`/`__truediv__` composed from `+`/`*`/`**`. This stage ADDS only the remaining primitive ops with their local rules (`__pow__`, `tanh`, `exp`, `relu`) plus a grad-aware `__repr__`.

**Exercise** — In `code.py`, complete the `Value` subclass. You may use ONLY Python stdlib (NumPy/Matplotlib allowed but unnecessary here; NO autodiff libs).
- ADD these new ops, each returning a `Value` whose `_backward` accumulates into inputs with `+=`:
  - `__pow__(self, n)` where `n` is a Python `int`/`float` constant (not a `Value`); rule $n x^{n-1}$. (stage_01's `__pow__` is non-differentiable; complete it here.)
  - `tanh(self)`, `exp(self)`, `relu(self)`.
- Thin `__add__`/`__mul__` overrides: delegate to `super()` (stage_05's math) and only re-bless the result as a stage-06 `Value` so unary ops chain on intermediates — do NOT re-derive gradients.
- `__repr__`: e.g. `Value(data=2.0, grad=4.0)` (stage_02's repr shows data/op only).
- Acceptance: every new `_backward` uses `+=` (never `=`); `__pow__` rejects a `Value` exponent; reused nodes (like `a` in the canonical example) get summed gradients.

**Done when**
- `pytest stage_06_backprop_engine/test.py` passes.
- The canonical `d.backward()` example yields `a.grad == 4.0`, `b.grad == 2.0`.
- Analytical grads from `.backward()` match central-difference numerical grads within `tol=1e-5` for every op (`+`, `*`, `pow`, `tanh`, `exp`, `relu`) and for composite expressions with reused nodes.
