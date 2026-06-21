# Stage 32: Framework refactor (mytorch)

**Imports** `Tensor` (s09), `Dense` (s11), `cross_entropy_loss`/`mse_loss` (s13), `SGD` (s14), `Adam` (s18), `DataLoader`/`Dataset` (s20) via `dlfs.stage_import`; **adds** the `Module`/`Parameter` object layer, `Sequential`, `Linear` (wraps `Dense`), `ReLU`, and the `CrossEntropyLoss`/`MSELoss` loss `Module`s — the consolidated `mytorch` public API.

**Context** — You have built every moving part of a deep-learning framework piece by piece across stages 1–31: the autodiff `Tensor` (`stage_09`), `Dense`/`Linear` and `MLP` (`stages 11–12`), losses (`stage_13`), optimizers (`stages 14, 17, 18`), and the `DataLoader` (`stage_20`). This stage does **no new math** — it *refactors* those scattered pieces into one coherent, importable mini-framework, **`mytorch`**, with a `torch.nn`-style object API: a `Module` base class, `Parameter`, container `Sequential`, layer/loss `Module`s, and a clean public namespace. The result is the package you will use unchanged in the remaining capstone stages.

**Background** — The only genuinely new abstraction is the **`Module`**: an object that *owns* `Parameter`s (leaf `Tensor`s that require grad) and possibly child `Module`s, and exposes a uniform interface — `parameters()` (recursively gather all leaf params), `zero_grad()` (clear every `p.grad`), `train()`/`eval()` (set a `training` flag, used by dropout/batchnorm `Module`s from stages 22–23), and `__call__` → `forward`. A `Parameter` is just a `Tensor` tagged `requires_grad=True`; recursive collection is what lets `optimizer = Adam(model.parameters())` work for any nesting depth. Each concrete `Module` wraps prior-stage code: `Linear` wraps the `Dense` forward $Z = XW + b$; `ReLU` wraps `Tensor.relu()`; `CrossEntropyLoss` wraps `cross_entropy_loss`. **No gradient is hand-derived here** — every forward is built from `Tensor` ops so `Tensor.backward()` (`stage_09`) produces the gradients. For reference, the two it ultimately drives are the linear-layer gradients and the fused softmax-CE gradient:
$$\frac{\partial \mathcal{L}}{\partial W} = X^\top \frac{\partial \mathcal{L}}{\partial Z},\qquad \frac{\partial \mathcal{L}}{\partial b} = \mathbf{1}^\top \frac{\partial \mathcal{L}}{\partial Z},\qquad \frac{\partial \mathcal{L}}{\partial Z_{\text{ce}}} = \frac{\operatorname{softmax}(Z) - Y}{B}.$$
The framework's job is purely *organizational*: collect parameters, route forward calls, and re-export a tidy namespace so `from mytorch import Tensor, Module, ...` reads like real PyTorch.

**Watch**
- [Building makemore — the spelled-out intro (nn.Module-style refactor)](https://www.youtube.com/watch?v=TCH_1BHY58I) — Karpathy refactors hand-wired nets into reusable `Linear`/`Module` blocks; this stage mirrors that.
- [PyTorch nn.Module explained](https://www.youtube.com/watch?v=Z_ikDlimN6A) — what `parameters()`, `train()/eval()`, and submodule registration actually do.

**Exercise** — Implement the `mytorch` framework surface in `code.py`. Reuse prior-stage code (load `Tensor` from `stage_09`, `Dense` from `stage_11`, `cross_entropy_loss`/`mse_loss` from `stage_13`, `SGD` from `stage_14`, `Adam` from `stage_18`, `DataLoader`/`Dataset` from `stage_20`) via `dlfs.stage_import`. Allowed tools: NumPy (forward array math only), Python stdlib, and your own prior stages. **No** PyTorch/TensorFlow/JAX/autograd libraries.

- `class Parameter(Tensor)`: a leaf `Tensor` with `requires_grad=True`. `__init__(self, data)` wraps `data` and sets `self.requires_grad = True`.
- `class Module`:
  - `__init__`: set `self.training = True`. Auto-register submodules/params via `__setattr__` (or via `_modules`/`_params` dicts).
  - `parameters(self) -> list[Tensor]`: recursively gather every `Parameter` from `self` and child `Module`s (no duplicates, stable order).
  - `zero_grad(self)`: set `p.grad = np.zeros_like(p.data)` for every parameter.
  - `train(self)` / `eval(self)`: set `self.training` (recursively on children) to `True`/`False`; return `self`.
  - `forward(self, *a, **k)`: `raise NotImplementedError`. `__call__` forwards to `forward`.
- `class Linear(Module)` — `Linear(in_features, out_features, bias=True, seed=None)`: build `Dense` (stage 11) internally; register `W` (and `b`) as `Parameter`s; `forward(x)` returns $XW+b$, shape `(B, out_features)`.
- `class ReLU(Module)`: `forward(x)` returns `x.relu()` (no parameters).
- `class Sequential(Module)` — `Sequential(*modules)`: store children in order; `forward(x)` chains them; `parameters()` returns all children's params in order.
- `class CrossEntropyLoss(Module)`: `forward(logits, targets)` returns scalar `Tensor` via `cross_entropy_loss` (stage 13).
- Re-export `SGD`, `Adam` (stage 18) and `DataLoader`, `Dataset` (stage 20) so they import from `mytorch`.
- An optimizer constructed from `model.parameters()` must update the *same* tensor objects the model holds (in-place on `p.data`).

**Done when**
- `pytest stage_32_framework_refactor/test.py` passes.
- `Linear` gradcheck: `dL/dW`, `dL/db` from `Tensor.backward()` match central differences within tol.
- `model.parameters()` recursively returns every `Parameter` of a nested `Sequential` exactly once, in order.
- `zero_grad()` clears all grads; `train()/eval()` toggle `training` on the module and all children.
- A tiny `Sequential(Linear, ReLU, Linear)` trains with `Adam` + `CrossEntropyLoss` and drives a 2-class toy loss down.
