"""Stage 15: First training loop.

Wires the framework built so far -- ``Tensor`` (stage_08/11/12), ``MLP``
(stage_11), ``mse_loss`` (stage_12), ``SGD`` (stage_14) -- into the canonical
learning loop:

    forward -> loss -> backward() -> step() -> zero_grad()

plus the off-graph ``accuracy`` metric used to watch it learn.  The exercise
is ``accuracy`` and ``train``; the ``plot_history`` helper is provided, and
the toy datasets (``make_moons`` / ``make_spiral``) live in this stage's
``test.py`` as fixtures.  Every gradient comes from ``Tensor.backward()`` --
nothing here derives a gradient by hand.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

# Framework pieces built in earlier stages, re-exported under canonical names.
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage11_MLP = stage_import("stage_11", "MLP")
Stage12_mse_loss = stage_import("stage_12", "mse_loss")
Stage14_SGD = stage_import("stage_14", "SGD")

Tensor = Stage12_Tensor
MLP = Stage11_MLP
mse_loss = Stage12_mse_loss
SGD = Stage14_SGD


# --------------------------------------------------------------------------- #
# Exercise 1: the accuracy metric (off-graph, read-only)
# --------------------------------------------------------------------------- #
def accuracy(pred: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]) -> float:
    """Binary accuracy: the fraction of examples where sign(pred) == sign(y).

    A metric, NOT a loss: it is read-only and off-graph.  Work on the raw
    ``.data`` arrays (never build ``Tensor`` ops here -- nothing should ever
    backprop through a metric).

    Accepts a ``Tensor`` or a plain ndarray for either argument, and ``(N, 1)``
    / ``(N,)`` shapes interchangeably: compare the flattened values.  Raise
    ``ValueError`` if the two carry different numbers of elements.  A
    prediction of exactly ``0.0`` has no sign, matches neither class, and
    counts as wrong.

    Returns a plain Python ``float`` in ``[0.0, 1.0]``.
    """
    if isinstance(pred, Tensor):
        pred = pred.data.flatten()
    
    if isinstance(y, Tensor):
        y = y.data

    pred = pred.flatten()
    y = y.flatten()

    result_arr = (np.sign(pred) == np.sign(y)).reshape(-1)

    return float(result_arr.sum() / len(result_arr))


# --------------------------------------------------------------------------- #
# Exercise 2: the training loop
# --------------------------------------------------------------------------- #
def train(
    model: "Stage11_MLP",
    X: "Tensor",
    y: "Tensor",
    *,
    lr: float = 0.1,
    epochs: int = 200,
    optimizer: Optional["Stage14_SGD"] = None,
) -> Dict[str, List[float]]:
    """Full-batch training: run the canonical loop from the README once per
    epoch, using the imported ``mse_loss`` and this stage's ``accuracy``.

    Contract (the tests pin every clause):
      * ``X`` and ``y`` must be ``Tensor`` instances -- raise ``TypeError``
        otherwise.  This guards the silent ``(N, 1) - (N,) -> (N, N)``
        broadcast bug: raw ndarrays don't get to skip the shape checks below.
      * ``X`` must be 2-D ``(N, n_in)``; ``y`` must be ``(N, 1)`` or ``(N,)``
        -- normalize it to a ``(N, 1)`` column OFF-graph (a fresh leaf
        ``Tensor``, not a graph op), and raise ``ValueError`` for any other
        shape or when the row counts disagree.
      * ``optimizer`` defaults to plain ``SGD`` over ``model.parameters()``
        with ``lr``; when one is passed, use it as-is and ignore ``lr``.
      * Per epoch, record the scalar loss (a plain ``float``) and the accuracy,
        both from that same forward pass (metrics log the model the step was
        computed on, i.e. before the update).
      * Return ``{"loss": [...], "accuracy": [...]}`` -- one float per epoch.
        After ``train`` returns, every parameter's ``.grad`` is all-zeros, so
        the caller can immediately ``backward()`` something else.
    """
    assert isinstance(X, Tensor)
    assert isinstance(y, Tensor)

    if y.data.ndim < X.data.ndim:
        y = y.reshape(X.shape[0], 1)

    if optimizer is None:
        optimizer = SGD(model.parameters(), lr=lr)

    loss_history = []
    accuracy_history = []

    for _ in range(epochs):
        pred = model(X)
        loss = mse_loss(pred, y)
        loss_history.append(float(loss.data))
        loss.grad = 1.0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        accuracy_history.append(accuracy(pred, y))
    
    return {"loss": loss_history, "accuracy": accuracy_history}



# --------------------------------------------------------------------------- #
# Plot helper (provided)
# --------------------------------------------------------------------------- #
def plot_history(history: Dict[str, List[float]], path: Optional[str] = None):
    """Plot the loss and accuracy curves from a ``train`` history dict.

    Saves to ``path`` if given, otherwise shows the figure.  Returns the Figure.
    """
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_axis = range(1, len(history["loss"]) + 1)

    ax_loss.plot(epochs_axis, history["loss"])
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training loss")

    ax_acc.plot(epochs_axis, history["accuracy"])
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_title("Training accuracy")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    return fig
