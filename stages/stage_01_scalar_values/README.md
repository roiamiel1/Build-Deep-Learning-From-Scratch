# Stage 01: Scalar values & arithmetic

**Cumulative chain** — This is the ORIGIN stage: it imports nothing from earlier stages (there are none). It exports the base `Value`, which every later stage extends via `dlfs.stage_import` — stage 02 subclasses it to add the computational graph (`_prev`/`_op`), stage 06 subclasses again to add `.backward()`.

**Context** — This is the first stage of the whole curriculum and the seed of the autodiff engine you will build. You wrap a single floating-point number in a `Value` object and teach it to do arithmetic (`+`, `-`, `*`, `/`) through Python operator overloading. There are **no gradients yet** — that machinery (`.backward()` filling `grad`) arrives in stage 06, and there is **no computational graph yet** — that arrives in stage 02. Here you only build the object and the forward math, plus the conceptual foundation of variables, functions, and derivatives.

**Background** — A neural network is just one enormous differentiable function built by composing tiny operations. To differentiate it automatically, we first need a data type we control at every step, instead of bare Python floats. A `Value` stores one number in `self.data`. Overloading `__add__`, `__mul__`, etc. lets `a + b` and `a * b` return *new* `Value`s, so expressions like `d = a * b + c` compose into the forward computation. The next stage (02) will subclass this `Value` to also record the operands and operation of each result — the skeleton of the computational graph — but here the result is just the number. The derivative is the foundation: for a function $f$, the derivative measures sensitivity of the output to a tiny change in an input,
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.$$
We approximate it numerically with the symmetric **central difference**, which has error $O(h^2)$ instead of $O(h)$:
$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}.$$
You will *not* derive analytic gradients in this stage — you only verify, numerically, that derivatives of your `Value` expressions exist and behave (e.g. $\frac{d}{da}(a b) = b$). Everything later (the computational graph in stage 02, the `Value` autodiff engine in stage 06) builds on the class and forward ops you write here.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this `Value` object from scratch; watch the first ~25 min (the class, `data`, and operator overloading) and stop before gradients.
- [Derivatives, the limit definition (Essence of Calculus, ch. 2)](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown on what a derivative actually is, the intuition you will need from stage 02 onward.

**Exercise** — Implement the base `Value` class in `code.py` (this is the origin: no `dlfs.stage_import`).
- `Value(data)`: store the number in `self.data` (coerce to `float`). Also initialize `self.grad = 0.0` (used in later stages, not here).
- `__repr__`: return `Value(data=<x>)`.
- Implement, each returning a **new** `Value` holding the resulting number:
  - `__add__(self, other)` → `self.data + other.data`
  - `__mul__(self, other)` → `self.data * other.data`
  - `__neg__(self)` → `-self.data` (implement via `self * -1`)
  - `__sub__(self, other)` → implement as `self + (-other)`
  - `__truediv__(self, other)` → implement as `self * other ** -1`
  - `__pow__(self, exponent)` → `self.data ** exponent`, where `exponent` is a Python `int`/`float` (not a `Value`)
- Support mixing with plain numbers (e.g. `2 * a`, `a + 1`, `a - 3`): wrap a non-`Value` operand in `Value`, and define the reflected ops `__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`.
- Allowed tools: Python stdlib only. **No** NumPy, no autodiff libraries. Do not add a `.backward()`, do not record a graph (`_prev`/`_op`), and do not compute any gradient — those arrive in stages 02 and 06.
- Acceptance: arithmetic on `Value`s matches the same arithmetic on raw floats; operations with ints/floats on either side work.

**Done when**
- [ ] `pytest stage_01_scalar_values/test.py` passes.
- [ ] `Value` supports `+ - * / ** neg` and mixed int/float operands on both sides.
- [ ] The central-difference test confirms numerical derivatives of `Value` expressions exist and match the expected slope within tolerance.
