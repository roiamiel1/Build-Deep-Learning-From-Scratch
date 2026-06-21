"""Stage 31: Vision Transformer (ViT).

Image front end + classifier head around the stage_30 Transformer encoder:
patchify -> patch embed -> prepend class token -> + positional embeddings ->
TransformerBlock stack -> LayerNorm -> linear head on the class-token row.
Pure NumPy with hand-written forward/backward + forward cache (no stage_09 autograd).

Shapes: x (N,C,H,W) -> cols (N*Np, C*P*P), Np=(H/P)*(W/P) -> patches (N,Np,D) ->
sequence (N,Np+1,D) after cls token -> logits (N, n_classes) from row 0.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# im2col/col2im (25): patch unfold + adjoint. TransformerBlock/LayerNorm (30): encoder + norm.
from dlfs import stage_import

Stage25_im2col, Stage25_col2im = stage_import("stage_25", "im2col", "col2im")
Stage30_TransformerBlock, Stage30_LayerNorm = stage_import(
    "stage_30", "TransformerBlock", "LayerNorm"
)


def num_patches(image_size: int, patch_size: int) -> int:
    """Number of square patches: (image_size // patch_size) ** 2. ValueError if patch_size does not divide image_size."""
    # TODO: validate, then return (image_size // patch_size) ** 2.
    raise NotImplementedError("num_patches")


class PatchEmbed:
    """Linear patch embedding: (N, C, H, W) -> (N, Np, D) tokens.

    Splits images into non-overlapping P x P patches via im2col (stride P, pad 0)
    and projects each flattened C*P*P patch with a learned linear map E (+ bias b).
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        d_model: int,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: init E ~ N(0, 1/sqrt(C*P*P)) (C*P*P, d_model), b zeros; zero grads; cache=None.
        raise NotImplementedError("PatchEmbed.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(N, C, H, W) -> (N, Np, d_model); patches ordered as im2col rows. Caches for backward."""
        # TODO: im2col -> cols @ E + b -> reshape (N, Np, D); cache and return.
        raise NotImplementedError("PatchEmbed.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward."""
        # TODO: return self.forward(x)
        raise NotImplementedError("PatchEmbed.__call__")

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop (N, Np, D) -> dL/dx (N, C, H, W); set E_grad/b_grad (via col2im)."""
        # TODO: compute E_grad/b_grad from cache, then col2im(dcols) -> dx.
        raise NotImplementedError("PatchEmbed.backward")

    def parameters(self) -> List[np.ndarray]:
        """Return [E, b]."""
        # TODO: return [self.E, self.b].
        raise NotImplementedError("PatchEmbed.parameters")

    def zero_grad(self) -> None:
        """Reset E_grad and b_grad to zeros."""
        # TODO: zero the gradient arrays.
        raise NotImplementedError("PatchEmbed.zero_grad")

    def __repr__(self) -> str:
        # TODO: short repr with C, P, D.
        raise NotImplementedError("PatchEmbed.__repr__")


class ViT:
    """Vision Transformer image classifier.

    Forward (square images): patch_embed -> prepend cls_token -> + pos_embed ->
    TransformerBlock stack -> final LayerNorm -> read row 0 -> linear head.
    depth == 0 (no blocks) is allowed for isolating embedding/head gradients.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        n_classes: int,
        *,
        norm: str = "pre",
        seed: Optional[int] = None,
    ) -> None:
        # TODO: build PatchEmbed, cls_token (1,1,D), pos_embed (1,Np+1,D), `depth`
        #       TransformerBlocks, final LayerNorm, head; zero grads; cache=None.
        raise NotImplementedError("ViT.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Classify (N, C, H, W) images -> logits (N, n_classes) from the class-token row. Caches for backward."""
        # TODO: implement the ViT forward; cache & return.
        raise NotImplementedError("ViT.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward."""
        # TODO: return self.forward(x)
        raise NotImplementedError("ViT.__call__")

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        """Backprop logits grad (N, n_classes) -> dL/dx (N, C, H, W); set head/pos/cls/sub-module grads."""
        # TODO: implement the reverse pass through head, norm, blocks, patch_embed.
        raise NotImplementedError("ViT.backward")

    def parameters(self) -> List[np.ndarray]:
        """Return all learnable params: patch_embed, cls_token, pos_embed, blocks, final_norm, W_head, b_head."""
        # TODO: concatenate the parameter lists in the documented order.
        raise NotImplementedError("ViT.parameters")

    def zero_grad(self) -> None:
        """Reset the gradient of every learnable array (own + sub-modules)."""
        # TODO: zero sub-module grads and own cls/pos/head grads.
        raise NotImplementedError("ViT.zero_grad")

    def __repr__(self) -> str:
        # TODO: short repr with image_size, patch_size, d_model, depth, n_classes.
        raise NotImplementedError("ViT.__repr__")
