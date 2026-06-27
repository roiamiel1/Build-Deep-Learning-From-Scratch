"""Stage 34: Capstone -- tiny Transformer (character-level language model).

INTEGRATION ONLY: assemble a GPT-style char-LM from prior stages (mytorch
framework, TransformerBlock + LayerNorm, causal mask, cross-entropy). No
gradients are hand-derived -- the Tensor engine produces them. SKELETON ONLY.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Prior-stage building blocks (integration only).
from dlfs import stage_import

# Tensor (09); mytorch Module/Parameter/Linear/Adam (32); TransformerBlock/LayerNorm (30).
Stage11_Tensor = stage_import("stage_11", "Tensor")
Stage31_Module, Stage31_Parameter, Stage31_Linear, Stage31_Adam = stage_import(
    "stage_31", "Module", "Parameter", "Linear", "Adam"
)
Stage29_TransformerBlock, Stage29_LayerNorm = stage_import(
    "stage_29", "TransformerBlock", "LayerNorm"
)
# causal_mask (28); cross_entropy_loss (13).
Stage27_causal_mask = stage_import("stage_27", "causal_mask")
Stage12_cross_entropy_loss = stage_import("stage_12", "cross_entropy_loss")


class CharTokenizer:
    """Character-level tokenizer (chars <-> int ids) from sorted unique chars of ``text``."""

    def __init__(self, text: str) -> None:
        # TODO: build sorted vocab ``chars`` and the ``stoi``/``itos`` maps.
        raise NotImplementedError("CharTokenizer.__init__")

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters (V)."""
        # TODO: implement vocab_size.
        raise NotImplementedError("CharTokenizer.vocab_size")

    def encode(self, s: str) -> np.ndarray:
        """Map a string to a 1-D int-id array, shape ``(len(s),)``."""
        # TODO: implement encode.
        raise NotImplementedError("CharTokenizer.encode")

    def decode(self, ids) -> str:
        """Map a 1-D iterable of int ids back to a string."""
        # TODO: implement decode.
        raise NotImplementedError("CharTokenizer.decode")


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample B random (input, next-token target) windows of length L from ``data`` (T,).
    Returns X (B, L) and Y (B, L), where Y is X shifted by one."""
    # TODO: sample start indices, stack windows X and their +1 shift Y.
    raise NotImplementedError("get_batch")


class TokenEmbedding(Stage31_Module):
    """Learnable embedding lookup: Parameter ``E`` (V, D); forward gathers ``E[ids]``;
    gather backward scatter-adds upstream grad into E (np.add.at semantics)."""

    def __init__(
        self, vocab_size: int, d_model: int, seed: Optional[int] = None
    ) -> None:
        # TODO: init Parameter ``E`` (V, D) with small normal; store config.
        raise NotImplementedError("TokenEmbedding.__init__")

    def forward(self, ids: np.ndarray) -> "Stage11_Tensor":
        """Gather ``E[ids]`` -> Tensor (B, L, D) as an autodiff node; backward
        scatter-adds into E.grad."""
        # TODO: build/return the gather Tensor with scatter-add backward.
        raise NotImplementedError("TokenEmbedding.forward")


def positional_embedding(
    block_size: int, d_model: int, seed: Optional[int] = None
) -> "Stage31_Parameter":
    """Create a learned positional table Parameter P, shape (block_size, D)."""
    # TODO: implement positional_embedding (small normal init).
    raise NotImplementedError("positional_embedding")


class TransformerLM(Stage31_Module):
    """GPT-style char-LM: token+positional embed -> pre-norm TransformerBlock stack
    (causal mask) -> final LayerNorm -> linear head, producing (B, L, V) logits."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        block_size: int,
        *,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: build tok_emb, pos_emb, n_layers blocks, final_ln, head (per-submodule seeds)
        raise NotImplementedError("TransformerLM.__init__")

    def forward(self, ids: np.ndarray) -> "Stage11_Tensor":
        """Compute next-token logits (B, L, V) from ids (B, L), L <= block_size."""
        # TODO: implement the pre-norm wiring; return (B, L, V) logits.
        raise NotImplementedError("TransformerLM.forward")

    def __call__(self, ids: np.ndarray) -> "Stage11_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        raise NotImplementedError("TransformerLM.__call__")

    def parameters(self) -> List["Stage11_Tensor"]:
        """Every learnable Parameter, once, in stable order (tok_emb, pos_emb, blocks, final_ln, head)."""
        # TODO: gather params from all submodules + pos_emb.
        raise NotImplementedError("TransformerLM.parameters")


def lm_loss(logits: "Stage11_Tensor", targets: np.ndarray) -> "Stage11_Tensor":
    """Mean next-token cross-entropy over all B*L positions; returns a scalar Tensor.
    A fresh model returns ~ ``ln(V)``."""
    # TODO: flatten positions into the batch axis and call cross_entropy_loss.
    raise NotImplementedError("lm_loss")


def train_lm(
    model: "TransformerLM",
    data: np.ndarray,
    *,
    block_size: int,
    batch_size: int,
    steps: int,
    lr: float,
    seed: Optional[int] = None,
) -> List[float]:
    """Train the LM with Adam; return per-step loss history (float(loss.data))."""
    # TODO: implement the Adam training loop; append loss per step.
    raise NotImplementedError("train_lm")


def sample(
    model: "TransformerLM",
    tokenizer: "CharTokenizer",
    *,
    prompt: str = "",
    max_new_tokens: int = 200,
    block_size: int = 64,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Autoregressively generate text: prompt + ``max_new_tokens`` sampled chars.
    Each step uses the last position's logits / temperature -> stable softmax -> draw."""
    # TODO: implement the autoregressive loop; decode ids back to a string.
    raise NotImplementedError("sample")


def plot_loss(history: List[float], path: Optional[str] = None) -> None:
    """Plot the training-loss curve (Matplotlib); save to ``path`` if given."""
    # TODO: plot history vs step; save/show.
    raise NotImplementedError("plot_loss")
