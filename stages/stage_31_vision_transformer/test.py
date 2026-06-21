"""Tests for Stage 31: Vision Transformer.

Checks the image-specific front end and head of a ViT (the parts this stage
builds), plus the assembled forward pipeline:

  * ``num_patches`` shape formula + validation,
  * ``PatchEmbed.forward`` matches an independent flatten-then-matmul reference,
  * ``PatchEmbed`` gradcheck (E, b, input image) via central differences:
        df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps),
  * ``ViT.forward`` output shape / token-sequence length / parameter shapes,
  * head + final-LayerNorm + class-token-select path gradcheck with ``depth=0``,
  * positional embeddings are actually used (patch-permutation invariance only
    holds when ``pos_embed`` is zeroed).

The pieces from stage_25 (im2col/col2im) and stage_30 (TransformerBlock,
LayerNorm) are loaded by this stage's code. If a dependency or the skeleton is
not implemented yet, the affected test skips cleanly instead of erroring.

Run with:  pytest stage_31_vision_transformer/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import PatchEmbed, ViT, num_patches
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_31 ViT / its stage_25 / stage_30 deps not importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(31)
EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-4


# --- helpers -----------------------------------------------------------------
def try_call(fn, *a, **k):
    """Run ``fn``; skip the test if the skeleton is still unimplemented."""
    try:
        return fn(*a, **k)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"not implemented yet: {exc}")


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued f at numpy point x (any shape)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = float(f(x))
        x[idx] = orig - eps
        fm = float(f(x))
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def weighted_sum(arr):
    """Fixed, non-symmetric scalar reduction so every output element matters."""
    arr = np.asarray(arr, dtype=float)
    w = np.linspace(0.1, 1.0, arr.size).reshape(arr.shape)
    return float(np.sum(arr * w)), w


def ref_patch_embed(x, E, b, P):
    """Independent reference: flatten each PxP patch (channel-major) then matmul.

    Mirrors im2col's row ordering: row index runs over (n, patch_row, patch_col),
    and each row is channel-major over (C, P, P).
    """
    N, C, H, W = x.shape
    nh, nw = H // P, W // P
    rows = []
    for n in range(N):
        for ph in range(nh):
            for pw in range(nw):
                patch = x[n, :, ph * P:(ph + 1) * P, pw * P:(pw + 1) * P]
                rows.append(patch.reshape(-1))  # channel-major flatten
    cols = np.stack(rows, axis=0)               # (N*nh*nw, C*P*P)
    out = cols @ E + b                          # (N*Np, D)
    return out.reshape(N, nh * nw, E.shape[1])


# ---------------------------------------------------------------------------
# num_patches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "img,p,expected",
    [(8, 4, 4), (8, 2, 16), (32, 8, 16), (12, 4, 9), (4, 4, 1)],
)
def test_num_patches(img, p, expected):
    got = try_call(num_patches, img, p)
    assert got == expected, f"num_patches({img},{p}) = {got}, want {expected}"


@pytest.mark.parametrize("img,p", [(8, 3), (10, 4), (7, 2)])
def test_num_patches_indivisible_raises(img, p):
    try:
        with pytest.raises(ValueError):
            num_patches(img, p)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"not implemented yet: {exc}")


# ---------------------------------------------------------------------------
# PatchEmbed: shapes, forward correctness
# ---------------------------------------------------------------------------
def test_patch_embed_param_shapes():
    pe = try_call(PatchEmbed, in_channels=3, patch_size=4, d_model=8, seed=1)
    E, b = try_call(pe.parameters)
    assert E.shape == (3 * 4 * 4, 8), f"E shape {E.shape} != (48, 8)"
    assert b.shape == (8,), f"b shape {b.shape} != (8,)"


def test_patch_embed_forward_shape():
    N, C, H, W, P, D = 2, 3, 8, 8, 4, 6
    pe = try_call(PatchEmbed, in_channels=C, patch_size=P, d_model=D, seed=2)
    x = RNG.standard_normal((N, C, H, W))
    out = try_call(pe.forward, x)
    Np = (H // P) * (W // P)
    assert out.shape == (N, Np, D), f"forward shape {out.shape} != ({N},{Np},{D})"


def test_patch_embed_matches_reference():
    N, C, H, W, P, D = 2, 2, 8, 6, 2, 5
    pe = try_call(PatchEmbed, in_channels=C, patch_size=P, d_model=D, seed=3)
    E, b = (np.asarray(p) for p in try_call(pe.parameters))
    x = RNG.standard_normal((N, C, H, W))
    out = np.asarray(try_call(pe.forward, x))
    ref = ref_patch_embed(x, E, b, P)
    assert out.shape == ref.shape, f"shape {out.shape} != ref {ref.shape}"
    assert maxdiff(out, ref) < 1e-10, (
        f"PatchEmbed forward disagrees with flatten+matmul reference: "
        f"max abs diff {maxdiff(out, ref):.3e}"
    )


# ---------------------------------------------------------------------------
# PatchEmbed: gradcheck (E, b, input image)
# ---------------------------------------------------------------------------
def test_patch_embed_gradcheck():
    N, C, H, W, P, D = 2, 2, 6, 6, 3, 4
    pe = try_call(PatchEmbed, in_channels=C, patch_size=P, d_model=D, seed=4)
    x = RNG.standard_normal((N, C, H, W))

    # analytic grads via forward/backward
    try_call(pe.zero_grad)
    out = np.asarray(try_call(pe.forward, x))
    _, w = weighted_sum(out)                 # dL/dout = w  (since L = sum(w*out))
    dx = np.asarray(try_call(pe.backward, w))
    E0, b0 = (np.asarray(p).copy() for p in try_call(pe.parameters))
    gE = np.asarray(pe.E_grad).copy()
    gb = np.asarray(pe.b_grad).copy()

    assert dx.shape == x.shape, f"dx shape {dx.shape} != {x.shape}"
    assert gE.shape == E0.shape, f"E_grad shape {gE.shape} != {E0.shape}"
    assert gb.shape == b0.shape, f"b_grad shape {gb.shape} != {b0.shape}"

    def loss_from_out(o):
        return float(np.sum(np.asarray(o) * w))

    # wrt E
    def fE(Evar):
        pe.E = Evar
        return loss_from_out(pe.forward(x))

    nE = central_diff(fE, E0.copy())
    pe.E = E0  # restore

    # wrt b
    def fb(bvar):
        pe.b = bvar
        return loss_from_out(pe.forward(x))

    nb = central_diff(fb, b0.copy())
    pe.b = b0  # restore

    # wrt input image
    def fx(xvar):
        return loss_from_out(pe.forward(xvar))

    nx = central_diff(fx, x.copy())

    assert maxdiff(gE, nE) < ATOL, f"E gradcheck max diff {maxdiff(gE, nE):.3e}"
    assert maxdiff(gb, nb) < ATOL, f"b gradcheck max diff {maxdiff(gb, nb):.3e}"
    assert maxdiff(dx, nx) < ATOL, f"dx gradcheck max diff {maxdiff(dx, nx):.3e}"


# ---------------------------------------------------------------------------
# ViT: construction, shapes
# ---------------------------------------------------------------------------
def make_vit(depth=2, **kw):
    cfg = dict(
        image_size=8, patch_size=4, in_channels=3, d_model=8, n_heads=2,
        d_ff=16, depth=depth, n_classes=5, norm="pre", seed=0,
    )
    cfg.update(kw)
    return try_call(ViT, **cfg)


def test_vit_token_and_param_shapes():
    v = make_vit(depth=1, image_size=8, patch_size=4, d_model=8, n_classes=5)
    Np = (8 // 4) ** 2
    assert np.asarray(v.cls_token).shape == (1, 1, 8), "cls_token must be (1,1,D)"
    assert np.asarray(v.pos_embed).shape == (1, Np + 1, 8), (
        "pos_embed must be (1, Np+1, D) -- the class token adds one position"
    )
    assert np.asarray(v.W_head).shape == (8, 5), "W_head must be (d_model, n_classes)"
    assert np.asarray(v.b_head).shape == (5,), "b_head must be (n_classes,)"


def test_vit_forward_shape():
    N = 3
    v = make_vit(depth=2, image_size=8, patch_size=4, in_channels=3,
                 d_model=8, n_classes=5)
    x = RNG.standard_normal((N, 3, 8, 8))
    logits = np.asarray(try_call(v.forward, x))
    assert logits.shape == (N, 5), f"logits shape {logits.shape} != ({N}, 5)"


def test_vit_depth_zero_runs():
    """depth=0 (no blocks) is a valid configuration used to isolate the head."""
    N = 2
    v = make_vit(depth=0, image_size=8, patch_size=4, in_channels=2,
                 d_model=6, n_classes=4)
    x = RNG.standard_normal((N, 2, 8, 8))
    logits = np.asarray(try_call(v.forward, x))
    assert logits.shape == (N, 4), f"depth=0 logits shape {logits.shape}"


# ---------------------------------------------------------------------------
# ViT head + final-norm + cls-select gradcheck (depth=0 isolates this stage)
# ---------------------------------------------------------------------------
def test_vit_head_gradcheck_depth0():
    N, C, P, D, K = 2, 2, 4, 6, 4
    v = make_vit(depth=0, image_size=8, patch_size=P, in_channels=C,
                 d_model=D, n_classes=K, seed=7)
    x = RNG.standard_normal((N, C, 8, 8))

    try_call(v.zero_grad)
    logits = np.asarray(try_call(v.forward, x))
    _, w = weighted_sum(logits)              # dL/dlogits = w
    dx = np.asarray(try_call(v.backward, w))

    Wh0 = np.asarray(v.W_head).copy()
    bh0 = np.asarray(v.b_head).copy()
    gWh = np.asarray(v.W_head_grad).copy()
    gbh = np.asarray(v.b_head_grad).copy()
    assert dx.shape == x.shape, f"dx shape {dx.shape} != {x.shape}"

    def loss(o):
        return float(np.sum(np.asarray(o) * w))

    def fWh(Wvar):
        v.W_head = Wvar
        return loss(v.forward(x))

    nWh = central_diff(fWh, Wh0.copy())
    v.W_head = Wh0

    def fbh(bvar):
        v.b_head = bvar
        return loss(v.forward(x))

    nbh = central_diff(fbh, bh0.copy())
    v.b_head = bh0

    assert maxdiff(gWh, nWh) < ATOL, f"W_head gradcheck max diff {maxdiff(gWh, nWh):.3e}"
    assert maxdiff(gbh, nbh) < ATOL, f"b_head gradcheck max diff {maxdiff(gbh, nbh):.3e}"


def test_vit_input_gradcheck_depth0():
    """With depth=0 the whole forward is differentiable end-to-end in pure NumPy
    (patch embed -> add pos -> final norm -> head); gradcheck the image."""
    N, C, P, D, K = 2, 1, 4, 6, 3
    v = make_vit(depth=0, image_size=8, patch_size=P, in_channels=C,
                 d_model=D, n_classes=K, seed=8)
    x = RNG.standard_normal((N, C, 8, 8))

    try_call(v.zero_grad)
    logits = np.asarray(try_call(v.forward, x))
    _, w = weighted_sum(logits)
    dx = np.asarray(try_call(v.backward, w))

    def fx(xvar):
        return float(np.sum(np.asarray(v.forward(xvar)) * w))

    nx = central_diff(fx, x.copy())
    assert maxdiff(dx, nx) < ATOL, (
        f"input gradcheck (depth=0) max diff {maxdiff(dx, nx):.3e}"
    )


# ---------------------------------------------------------------------------
# positional embeddings are actually used
# ---------------------------------------------------------------------------
def test_pos_embed_is_added_to_sequence():
    """A nonzero pos_embed must change the output -- it is *added* to the
    sequence (including the class-token row), so zeroing it vs. perturbing it
    gives different logits even at depth=0 (no attention mixing required)."""
    N, C, P, D, K = 2, 2, 4, 6, 4
    v = make_vit(depth=0, image_size=8, patch_size=P, in_channels=C,
                 d_model=D, n_classes=K, seed=9)
    x = RNG.standard_normal((N, C, 8, 8))

    v.pos_embed = np.zeros_like(np.asarray(v.pos_embed))
    l0 = np.asarray(try_call(v.forward, x))

    v.pos_embed = RNG.standard_normal(np.asarray(v.pos_embed).shape) * 0.5
    l1 = np.asarray(v.forward(x))

    assert maxdiff(l0, l1) > 1e-6, (
        "changing pos_embed must change the logits -- the positional embedding "
        "is otherwise not being added into the sequence"
    )


def test_pos_embed_grad_gradcheck_depth0():
    """Gradcheck the learnable positional embedding at depth=0."""
    N, C, P, D, K = 2, 1, 4, 6, 3
    v = make_vit(depth=0, image_size=8, patch_size=P, in_channels=C,
                 d_model=D, n_classes=K, seed=10)
    x = RNG.standard_normal((N, C, 8, 8))

    try_call(v.zero_grad)
    logits = np.asarray(try_call(v.forward, x))
    _, w = weighted_sum(logits)
    try_call(v.backward, w)
    g_pos = np.asarray(v.pos_embed_grad).copy()
    pos0 = np.asarray(v.pos_embed).copy()
    assert g_pos.shape == pos0.shape, f"pos_embed_grad shape {g_pos.shape}"

    def f(pvar):
        v.pos_embed = pvar
        return float(np.sum(np.asarray(v.forward(x)) * w))

    n_pos = central_diff(f, pos0.copy())
    v.pos_embed = pos0
    assert maxdiff(g_pos, n_pos) < ATOL, (
        f"pos_embed gradcheck max diff {maxdiff(g_pos, n_pos):.3e}"
    )


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
def test_seed_reproducible():
    a = make_vit(depth=1, seed=123)
    b = make_vit(depth=1, seed=123)
    for pa, pb in zip(try_call(a.parameters), try_call(b.parameters)):
        assert np.allclose(np.asarray(pa), np.asarray(pb)), (
            "same seed must produce identical parameters"
        )
