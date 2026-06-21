# Stage 24: Convolution mathematics

**Context** — Before coding a fast `Conv2d` layer in the next stage, you derive the math on
paper and in shape helpers. This is the convolution analogue of `stage_08`, where you learned
the matmul gradient by "dimension analysis": here you reduce 2-D convolution to a single matrix
multiply via **im2col**, so all the `stage_08` matmul-gradient machinery applies directly. No
autodiff, no layer — just output-shape formulas, the forward as cross-correlation, the three
backward rules (input/kernel/bias), and the im2col reshape that ties them together.

**Background** — A conv layer maps an input of shape $(N, C_{in}, H, W)$ through a kernel
$K$ of shape $(C_{out}, C_{in}, k_H, k_W)$ and bias $b$ of shape $(C_{out},)$ to an output of
shape $(N, C_{out}, H_{out}, W_{out})$. What deep-learning libraries call "convolution" is really
**cross-correlation** (no kernel flip):
$$Y[n,o,i,j] = b[o] + \sum_{c}\sum_{u=0}^{k_H-1}\sum_{v=0}^{k_W-1} X[n,c,\,s\,i - p + d\,u,\;s\,j - p + d\,v]\,K[o,c,u,v],$$
with stride $s$, symmetric pad $p$, dilation $d$. The **output size** along each spatial axis is
$$H_{out} = \left\lfloor \frac{H + 2p - d(k_H-1) - 1}{s}\right\rfloor + 1,$$
and likewise for $W_{out}$ (the dilated kernel spans $d(k-1)+1$ pixels). **im2col** unrolls every
receptive field into a column: $X \to X_{col}$ of shape $(N\cdot H_{out}\cdot W_{out},\; C_{in}k_Hk_W)$,
and the kernel flattens to $W_{row}$ of shape $(C_{out},\; C_{in}k_Hk_W)$. Then the whole forward is
one matmul $Y_{col} = X_{col} W_{row}^{\top} + b$, exactly the `stage_08` pattern. So the backward
borrows `stage_08`'s $dL/dA = G B^{\top}$, $dL/dB = A^{\top}G$. With $G = dL/dY_{col}$:
$$\frac{\partial L}{\partial W_{row}} = G^{\top} X_{col},\qquad
\frac{\partial L}{\partial X_{col}} = G\, W_{row},\qquad
\frac{\partial L}{\partial b}=\sum_{n,i,j} G[n,i,j,:].$$
The only convolution-specific step is the inverse of im2col, **col2im**: scatter-add $dL/dX_{col}$
back to the overlapping input positions (overlaps **sum**). $dL/dW_{row}$ reshapes back to $K$'s
4-D shape. This is the bridge to `stage_08`: convolution *is* matmul once you im2col.

**Watch**
- [But what is a convolution? (3Blue1Brown)](https://www.youtube.com/watch?v=KuXjwB4LzSA) — the sliding-window sum, and why "convolution" in ML is cross-correlation.
- [Convolutional Neural Networks (Stanford CS231n, lecture 5)](https://www.youtube.com/watch?v=bNb2fEVKeEo) — output-size formula, stride/pad, and the im2col trick.

**Exercise** — This stage is math + shape helpers only. Implement the pure functions in `code.py`.
You may use Python stdlib and NumPy for array/shape math only. No autodiff library, no layer class.

- `conv_output_size(in_size, kernel, stride=1, pad=0, dilation=1) -> int`: return
  $\lfloor (in\_size + 2\,pad - dilation(kernel-1) - 1)/stride\rfloor + 1$. Raise `ValueError`
  if the result is `< 1` or any argument is invalid (`stride < 1`, `kernel < 1`, `pad < 0`,
  `dilation < 1`).
- `conv2d_output_shape(x_shape, w_shape, stride, pad, dilation) -> tuple`: given
  `x_shape=(N,C_in,H,W)` and `w_shape=(C_out,C_in,kH,kW)`, return `(N,C_out,H_out,W_out)`.
  Raise `ValueError` if the two `C_in` disagree.
- `im2col_shape(x_shape, w_shape, stride, pad, dilation) -> tuple`: return the 2-D shape
  `(N*H_out*W_out, C_in*kH*kW)` of the unrolled matrix.
- `conv2d_forward(x, w, b, stride=1, pad=0, dilation=1) -> np.ndarray`: reference forward via
  im2col + matmul (cross-correlation). Inputs are plain NumPy arrays. This is the numerical
  oracle the tests differentiate.
- `conv2d_backward(dy, x, w, b, stride, pad, dilation) -> (dx, dw, db)`: analytical gradients
  using the three formulas above and **col2im** (scatter-add for overlaps). Shapes must match
  `x`, `w`, `b` respectively.
- Acceptance: shape helpers match the formula on a table of cases; `conv2d_forward` matches a
  naive triple-loop cross-correlation; `dx, dw, db` match central-difference numerical gradients
  within `1e-4`; overlapping receptive fields accumulate (col2im sums, never overwrites).

**Done when**
- `pytest stage_24_conv2d_math/test.py` passes.
- `conv_output_size` / `conv2d_output_shape` / `im2col_shape` match the formulas on all cases.
- `conv2d_forward` equals the naive cross-correlation reference.
- `dx`, `dw`, `db` from `conv2d_backward` match central-difference gradcheck within `1e-4`.
