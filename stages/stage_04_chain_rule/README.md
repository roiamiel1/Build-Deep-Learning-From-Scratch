# Stage 4: Chain rule

**Context** — In stage 03 you computed the *local* derivative of a single operation in isolation (for `c = a * b`, the locals are `dc/da = b` and `dc/db = a`). Real expressions are many ops chained together. This stage teaches you to propagate a derivative through a *multi-op* expression by multiplying locals along the path — by hand, with no graph object and no autodiff engine yet. This is the manual rehearsal for the reverse pass you will automate in stage 06's `Value`.

**Background** — The chain rule says the derivative of a composition is the product of the derivatives of its links. If $y = f(u)$ and $u = g(x)$, then

$$\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}.$$

For a deeper chain $x \to u \to v \to y$ you multiply every local link:

$$\frac{dy}{dx} = \frac{dy}{dv}\cdot\frac{dv}{du}\cdot\frac{du}{dx}.$$

When an input feeds **two** branches that later recombine (e.g. $y = a\cdot b$ where $a = x+1$ and $b = x^2$), the total derivative is the **sum over paths** — the multivariate chain rule:

$$\frac{dy}{dx} = \frac{\partial y}{\partial a}\frac{da}{dx} + \frac{\partial y}{\partial b}\frac{db}{dx}.$$

Two reasons we go backward rather than forward. First, the *output* is what we ultimately want gradients of (a scalar loss, later), so seeding $\frac{dy}{dy}=1$ and pushing that signal toward the inputs reuses shared sub-results. Second, going backward means every node receives one number — its derivative of the output w.r.t. itself, sometimes called the **adjoint** $\bar{n} = \frac{dy}{dn}$ — and passes a scaled copy to each of its inputs.

The practical recipe (a manual **backward pass**): seed the output adjoint $\bar{y}=1$, then walk the ops in reverse; for each op, multiply the downstream adjoint by that op's **local derivative** (from stage 03 — `mul_local` gives $(b,a)$ for $a\cdot b$, `add_local` gives $(1,1)$, etc.) and **accumulate** (add) into each input's adjoint when paths merge. Stage 03 gave you the locals one op at a time; this stage *composes* them into a full gradient. In stage 05 you'll order the nodes with a topological sort so accumulation is always correct, and in stage 06 the `Value` class will run this exact bookkeeping for you inside `.backward()`.

**Watch**
- [What is backpropagation really doing? (3Blue1Brown)](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — visual intuition for chaining local derivatives backward.
- [The spelled-out intro to backpropagation (Andrej Karpathy)](https://www.youtube.com/watch?v=VMj-3S1tku0) — manual forward+backward on a small expression, exactly this stage.

**Cumulative build** — This stage reuses, via `dlfs.stage_import`, `Value` from stage_02 (bound as `Stage2_Value`) and the `d_add`/`d_mul` locals from stage_03 (bound as `Stage3_d_add`/`Stage3_d_mul`, no new `Value`), and ADDS `chain()`, `accumulate()`, and a manual `backward_pass()` that compose those locals by hand.

**Exercise** — Implement, in `code.py`, manual chain-rule propagation over fixed expressions using only Python floats (NumPy allowed only for the gradcheck helper, never to compute a derivative for you).

- `chain(locals_list)` — given a list of local derivatives `[d1, d2, ..., dn]` along a single straight path, return their product $\prod_i d_i$ (the chain rule for a linear chain). Empty list returns `1.0`.
- `accumulate(path_derivs)` — given a list of per-path total derivatives that all reach the same input, return their sum (multivariate chain rule for merging branches).
- `f_chain(x)` — compute and return `(y, dy_dx)` for the straight chain $u = 3x + 1$, $v = u^2$, $y = \tanh(v)$. Compute `y` by a forward pass (store the intermediates `u`, `v`); compute `dy_dx` by a manual backward pass that feeds the three local derivatives into `chain`: $\frac{du}{dx}=3$, $\frac{dv}{du}=2u$, $\frac{dy}{dv}=1-\tanh^2(v)$. Note the local for $y=\tanh(v)$ is written in terms of the *output*, so reuse the `y` you already computed. Do **not** use a finite difference.
- `f_branch(x)` — compute and return `(y, dy_dx)` for the branching expression $a = x + 1$, $b = x^2$, $y = a\cdot b$, where `x` feeds both branches. The path through `a` contributes $\frac{\partial y}{\partial a}\frac{da}{dx} = b\cdot 1$; the path through `b` contributes $\frac{\partial y}{\partial b}\frac{db}{dx} = a\cdot 2x$. `dy_dx` must come from the sum-over-paths rule using `accumulate` on those two contributions.
- `backward_pass(x)` — return a dict mapping each node name to its adjoint (derivative of the output w.r.t. that node) for `f_branch`'s expression: keys `"y"`, `"a"`, `"b"`, `"x"`. Seed `"y" -> 1.0`; set `"a" = dy/da`, `"b" = dy/db`; then `"x"` must be the `accumulate` of the two contributions arriving from `a` and `b`. The returned `"x"` must equal `f_branch(x)[1]`.

Inputs are Python floats; outputs are floats (or a float-valued dict for `backward_pass`). No graph/`Value` class yet — keep it explicit and procedural. Use `chain` and `accumulate` as the building blocks so the composition is visible.

**Done when**
- `pytest stage_04_chain_rule/test.py` passes.
- `chain` and `accumulate` match hand computation; empty-`chain` returns `1.0`.
- `f_chain` and `f_branch` analytical `dy_dx` match central-difference gradcheck within `tol=1e-5` across several `x`.
- `backward_pass` returns `dy/dx` equal to `f_branch`'s `dy_dx`, and `dy/dy == 1.0`.
