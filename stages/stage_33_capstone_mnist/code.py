"""Stage 33: Capstone -- MNIST.

Integration stage: read real MNIST off disk and train a stage_32 ``mytorch``
MLP. No new gradient math, no new layer/loss/optimizer -- only the stage_32 API.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Whole framework re-exported from stage_32 (Tensor/Module/layers/loss/optim/data).
from dlfs import stage_import

(
    Stage32_Tensor,
    Stage32_Module,
    Stage32_Sequential,
    Stage32_Linear,
    Stage32_ReLU,
    Stage32_CrossEntropyLoss,
    Stage32_Adam,
    Stage32_DataLoader,
    Stage32_Dataset,
) = stage_import(
    "stage_32",
    "Tensor",
    "Module",
    "Sequential",
    "Linear",
    "ReLU",
    "CrossEntropyLoss",
    "Adam",
    "DataLoader",
    "Dataset",
)

# Canonical name for the stage_32 Tensor (tests read CODE.Tensor).
Tensor = Stage32_Tensor

# Lazily-built, cached stage_32 CrossEntropyLoss (avoids forcing Module at import).
_ce_loss_module = None


def cross_entropy_loss(logits, targets):
    """Scalar softmax cross-entropy via the stage_32 ``CrossEntropyLoss`` module."""
    global _ce_loss_module
    if _ce_loss_module is None:
        _ce_loss_module = Stage32_CrossEntropyLoss()
    return _ce_loss_module(logits, targets)


def accuracy(logits, targets) -> float:
    """Fraction of rows where argmax(logits) == target. logits (B,C), targets (B,)."""
    # TODO: implement classification accuracy (argmax over logits vs integer labels)
    raise NotImplementedError("accuracy")


def load_mnist_idx(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read MNIST images/labels from big-endian IDX (plain or .gz). Returns
    (X (N,28,28) float64 0..255, y (N,) int); ValueError on bad magic/count."""
    # TODO: parse the IDX headers (gzip-open when .gz), read payloads, validate magics/counts
    raise NotImplementedError("load_mnist_idx")


def preprocess(X, *, flatten: bool, normalize: bool = True) -> np.ndarray:
    """Scale pixels to [0,1] and reshape: flatten -> (N,784) else (N,1,28,28)."""
    # TODO: optionally /255, then reshape per the flatten flag
    raise NotImplementedError("preprocess")


def make_loaders(
    X,
    y,
    *,
    batch_size: int,
    flatten: bool,
    val_frac: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple["Stage32_DataLoader", Optional["Stage32_DataLoader"]]:
    """Preprocess (X, y) and wrap into stage_32 DataLoaders. Returns
    (train_loader, val_loader); train shuffles, val_loader is None if val_frac==0."""
    # TODO: preprocess, split off val_frac via an rng permutation, build the two DataLoaders
    raise NotImplementedError("make_loaders")


def build_mlp(seed: Optional[int] = None) -> "Stage32_Sequential":
    """The capstone classifier: Sequential(Linear(784,128), ReLU, Linear(128,10))."""
    # TODO: build the 784->128->10 Sequential from stage_32 Linear/ReLU
    raise NotImplementedError("build_mlp")


def evaluate(model, loader) -> Dict[str, float]:
    """Size-weighted mean loss + accuracy over ``loader`` (eval mode, no backward).
    Returns {"loss", "acc"}."""
    # TODO: switch to eval mode, accumulate size-weighted loss/accuracy over the loader
    raise NotImplementedError("evaluate")


def train_mnist(
    model,
    train_loader,
    *,
    epochs: int,
    lr: float = 1e-3,
    val_loader=None,
    optimizer=None,
    verbose: bool = False,
) -> Dict[str, list]:
    """Train ``model`` with Adam over ``epochs`` passes of ``train_loader``. Returns
    history {"train_loss", "val_loss", "val_acc", "steps"} (val_* empty if no val_loader)."""
    # TODO: per-epoch forward/backward/step loop; record size-weighted train loss + optional val metrics
    raise NotImplementedError("train_mnist")


def make_synthetic_digits(
    n_per_class: int = 64,
    *,
    n_classes: int = 10,
    noise: float = 0.4,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny MNIST-shaped fallback (per-class stroke templates + noise) so the pipeline
    runs without the real dataset. Returns (X (N,28,28) ~0..255, y (N,) int), shuffled."""
    # TODO: draw n_classes distinct 28x28 templates, add noise, rescale to 0..255, shuffle
    raise NotImplementedError("make_synthetic_digits")


def _default_mnist_paths(data_dir: str) -> Tuple[str, str, str, str]:
    """Return (train_images, train_labels, test_images, test_labels) paths under
    ``data_dir`` (prefers .gz; paths need not exist)."""
    # TODO: resolve the four canonical IDX basenames (.gz if present) under data_dir
    raise NotImplementedError("_default_mnist_paths")


def run_capstone(
    data_dir: Optional[str] = None,
    *,
    model: str = "mlp",
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    subset: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, object]:
    """End-to-end: load MNIST (or synthetic fallback), train, evaluate on test. Returns
    {"test_acc", "test_loss", "history", "model", "source"} (source mnist/synthetic)."""
    # TODO: resolve data (real IDX or synthetic), build loaders + MLP, train, evaluate, report
    raise NotImplementedError("run_capstone")
