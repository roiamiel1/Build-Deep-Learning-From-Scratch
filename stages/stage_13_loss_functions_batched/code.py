"""Stage 13: Loss functions over a BATCH.

stage_12 built softmax / log_softmax / cross_entropy for a single example -- a (C,)
logit vector and one label. Real training pushes a whole *batch* of examples through
the network at once, so this stage lifts those three functions to operate on a
(B, C) matrix of logits: B examples stacked as rows, C class scores per row.

The math per row is identical to stage_12; the only new idea is doing it
*independently per row* and then averaging the per-example losses over the batch. So
every reduction that was over "the C classes" becomes a reduction over ``axis=1``
(the class axis), and the final cross-entropy averages over the B rows with ``.mean``.

mse_loss / mae_loss already worked elementwise over any shape, so they are reused from
stage_12 unchanged; only the classification losses need a batched rewrite.

All losses use Tensor ops so gradients flow through ``Tensor.backward()``; NumPy is
allowed for forward array construction only, never for grads.
"""

from __future__ import annotations

import numpy as np

from dlfs import stage_import

# stage_12 already added the sum/mean reductions to the Tensor and built the
# single-example losses. Reuse its Tensor (so .sum(axis=...) / .mean() are present)
# and re-export the regression losses unchanged.
Tensor = stage_import("stage_12", "Tensor")
mse_loss = stage_import("stage_12", "mse_loss")
mae_loss = stage_import("stage_12", "mae_loss")


def log_softmax(logits) -> "Tensor":
    """Row-wise log-softmax of a BATCH of logits.

    INPUT:  logits -- a (B, C) Tensor: B examples (rows), C class scores each.
    OUTPUT: a (B, C) Tensor: each row holds that example's log-probabilities log(p_c).

    This is the stage_12 log_softmax applied to every row independently. The stable
    identity, now with a PER-ROW max m_b = max_c z_{b,c}:

        log_softmax(z)_{b,c} = (z_{b,c} - m_b) - log( sum_k exp(z_{b,k} - m_b) )

    The only change from the single-example version is that the max and the
    log-sum-exp are taken along the class axis (``axis=1``) and kept as a (B, 1)
    column so they broadcast back over the C classes of each row.

    STEPS:
      1. m = the per-row max logit, as a (B, 1) constant (off the data; no grad).
      2. shift = logits - m.
      3. logsumexp = log( sum over axis=1 of exp(shift) ), keepdims -> (B, 1).
      4. return shift - logsumexp.
    """
    m = np.max(logits.data, axis=1, keepdims=True)
    return (logits - m) - (logits - m).exp().sum(axis=1, keepdims=True).log()


def softmax(logits) -> "Tensor":
    """Row-wise softmax of a BATCH of logits.

    INPUT:  logits -- a (B, C) Tensor.
    OUTPUT: a (B, C) Tensor: each row is a probability distribution (entries in (0, 1),
            the row sums to 1).

    As in stage_12, softmax is just exp of log_softmax, so reuse the batched
    log_softmax and inherit its per-row stability.
    """
    logits -= np.max(logits.data, axis=1, keepdims=True)
    logits_exp = logits.exp()
    return logits_exp / logits_exp.sum(axis=1, keepdims=True)

def cross_entropy_loss(logits, targets) -> "Tensor":
    """Mean softmax cross-entropy over a BATCH.

    INPUTS:
      logits  -- a (B, C) Tensor: the raw class scores for B examples.
      targets -- the correct class of each of the B examples, in one of two forms:
                 * shape (B,) integers: targets[i] is the class index for row i, or
                 * shape (B, C) array: row i is a probability distribution over the
                   classes (a single 1.0 for a hard label, fractions for a soft one).

    OUTPUT: a scalar (0-d) Tensor -- the per-example cross-entropy AVERAGED over the
    batch. For each row it is the negative log-probability the model gave that row's
    correct class; the batch loss is the mean of those B numbers.

    This is the stage_12 cross-entropy done per row, then averaged. With per-row
    log-probs lp = log_softmax(logits) and the targets written as a (B, C) matrix y
    (each row a one-hot / soft label):

        L = -(1/B) * sum_b sum_c  y_{b,c} * lp_{b,c}

    i.e. multiply y by lp, sum over the class axis (``axis=1``) to get one loss per
    row, then ``.mean()`` over the batch. Build it from log_softmax so backward()
    produces the clean (p - y)/B gradient on its own -- never hand-write it.

    Turning (B,) integer targets into the (B, C) matrix y, and how you pick out each
    row's true-class log-prob, is yours to design (it generalizes the length-C vector
    you built in stage_12 to one row per example).
    """
    # TODO: lp = log_softmax(logits); build the (B, C) target matrix y from `targets`;
    #       return -(lp * y).sum(axis=1).mean(); let backward() supply the gradient.
    if len(targets.shape) == 1:
        # this is an index of the right class in each batch entry
        indices = targets
        targets = np.zeros((targets.shape[0], logits.shape[1],))
        for batch_entry in range(targets.shape[0]):
            targets[batch_entry][indices[batch_entry]] = 1.0

    lp = log_softmax(logits)
    return -(lp * targets).sum(axis=1, keepdims=True).mean()
