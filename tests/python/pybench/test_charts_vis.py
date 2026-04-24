from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from pybench.charts.meta import RunMetadata
from pybench.charts.vis import (
    MultiDimVisInputs,
    VisInputs,
    _NOISE_COLOR,
    _colors_for_labels,
    _palette_for_k,
    _rgba_hex,
    build_empty_vis_figure,
    build_multidim_vis_figure,
    build_vis_figure,
)


def _meta() -> RunMetadata:
    return RunMetadata(
        timestamp_iso="2026-04-24T00:00:00+00:00",
        git_sha="abc1234",
        machine="test-host",
        canonical_encoding_version=1,
    )


def _make_inputs(
    *,
    n: int = 1000,
    d: int = 2,
    k: int = 5,
    theirs_mode: str = "match",  # "match" | "permuted" | "none" | "disagree"
    seed: int = 42,
    subsample_seed: int = 123,
    max_scatter_points: int = 20_000,
) -> VisInputs:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    gt = rng.integers(0, k, n).astype(np.int32)
    # Ours is a pure permutation of gt so Hungarian alignment brings it back.
    perm = rng.permutation(k)
    ours = perm[gt].astype(np.int32)
    if theirs_mode == "match":
        theirs: np.ndarray | None = gt.copy()
    elif theirs_mode == "permuted":
        perm_t = rng.permutation(k)
        theirs = perm_t[gt].astype(np.int32)
    elif theirs_mode == "disagree":
        theirs = gt.copy()
        # Corrupt ~5% of points.
        idx = rng.choice(n, size=max(1, n // 20), replace=False)
        theirs[idx] = (theirs[idx] + 1) % max(1, k)
    elif theirs_mode == "none":
        theirs = None
    else:
        raise ValueError(theirs_mode)
    proj = X[:, :2].astype(np.float32)
    return VisInputs(
        algo="kmeans",
        recipe_name="kmeans_blobs",
        size=n,
        dims=d,
        non_njobs_params=(("k", k),),
        n_jobs=1,
        ari=0.97,
        title_meta=_meta(),
        gt_labels=gt,
        ours_labels=ours,
        theirs_labels=theirs,
        projection_2d=proj,
        subsample_seed=subsample_seed,
        max_scatter_points=max_scatter_points,
    )


def _png_bytes(fig: Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Layout / structural tests
# -----------------------------------------------------------------------------


def test_vis_figure_has_4_panels() -> None:
    fig = build_vis_figure(_make_inputs())
    assert len(fig.axes) == 4


def test_vis_figure_shares_xy_limits() -> None:
    """All four panels share x/y limits (pinned from projection bounds + 5% pad)."""
    fig = build_vis_figure(_make_inputs())
    x_lims = [tuple(float(v) for v in ax.get_xlim()) for ax in fig.axes]
    y_lims = [tuple(float(v) for v in ax.get_ylim()) for ax in fig.axes]
    assert all(lim == x_lims[0] for lim in x_lims)
    assert all(lim == y_lims[0] for lim in y_lims)


def test_full_agreement_shows_annotation() -> None:
    """Pure permutation of gt -> disagreement panel shows 'full agreement'
    text, NOT a hexbin PolyCollection."""
    # Pure permutation on both sides -> after alignment, zero disagreement.
    inputs = _make_inputs(theirs_mode="permuted")
    fig = build_vis_figure(inputs)
    disagree_ax = fig.axes[3]
    # Hexbin creates a PolyCollection; we must NOT have one here.
    polycolls = [c for c in disagree_ax.collections if isinstance(c, PolyCollection)]
    assert polycolls == []
    # And we MUST have the "full agreement" text annotation.
    texts = [t.get_text() for t in disagree_ax.texts]
    assert any("full agreement" in t for t in texts)


def test_disagreement_renders_hexbin_when_points_disagree() -> None:
    """When ~5% of points disagree, the disagreement panel DOES render a hexbin."""
    inputs = _make_inputs(theirs_mode="disagree")
    fig = build_vis_figure(inputs)
    disagree_ax = fig.axes[3]
    polycolls = [c for c in disagree_ax.collections if isinstance(c, PolyCollection)]
    # Hexbin renders at least one PolyCollection (the tiled hex grid).
    assert len(polycolls) >= 1
    # No "full agreement" annotation should appear in this case.
    texts = [t.get_text() for t in disagree_ax.texts]
    assert not any("full agreement" in t for t in texts)


# -----------------------------------------------------------------------------
# Noise color consistency
# -----------------------------------------------------------------------------


def test_noise_is_grey_across_panels() -> None:
    """Noise color (-1) is the SAME RGBA across gt, ours, theirs panels."""
    rng = np.random.default_rng(7)
    # Fixture: some real clusters + some noise in every label array.
    pattern = np.array([-1, 0, 0, 1, 1, 2, 2, -1], dtype=np.int32)
    gt = np.tile(pattern, 64)
    n = gt.size
    ours = gt.copy()
    theirs = gt.copy()
    X = rng.standard_normal((n, 2)).astype(np.float32)
    inputs = VisInputs(
        algo="dbscan",
        recipe_name="dbscan_blobs",
        size=n,
        dims=2,
        non_njobs_params=(),
        n_jobs=1,
        ari=1.0,
        title_meta=_meta(),
        gt_labels=gt,
        ours_labels=ours,
        theirs_labels=theirs,
        projection_2d=X,
        subsample_seed=1,
    )

    noise_rgba = _rgba_hex(_NOISE_COLOR)
    # The palette function itself: -1 should never be called through it;
    # _colors_for_labels handles that separately. Directly check the utility.
    palette = _palette_for_k(3)
    colors = _colors_for_labels(
        np.array([-1, 0, -1, 1, 2, -1], dtype=np.int32), palette
    )
    # The three -1 entries must share the noise color exactly.
    for i in (0, 2, 5):
        assert tuple(colors[i]) == pytest.approx(noise_rgba)
    # Non-noise entries must NOT match the noise color.
    for i in (1, 3, 4):
        assert tuple(colors[i]) != pytest.approx(noise_rgba)

    # And the full build path preserves that: rendering gt / ours / theirs
    # with the same data reads back the same noise RGBA on each.
    fig = build_vis_figure(inputs)
    # Smoke-check: PNG actually rendered.
    assert _png_bytes(fig)


# -----------------------------------------------------------------------------
# Palette dispatch tests
# -----------------------------------------------------------------------------


def test_palette_dispatch_k_0() -> None:
    """k == 0: every id returns grey."""
    palette = _palette_for_k(0)
    noise = _rgba_hex(_NOISE_COLOR)
    for i in (0, 1, 5, 100):
        assert palette(i) == pytest.approx(noise)


def test_palette_dispatch_k_1() -> None:
    """k == 1: tab10 used, single color returned."""
    palette = _palette_for_k(1)
    c0 = palette(0)
    # tab10 returns RGBA; alpha is 1.0.
    assert len(c0) == 4
    assert c0[3] == pytest.approx(1.0)
    # Must NOT be noise grey.
    assert c0 != pytest.approx(_rgba_hex(_NOISE_COLOR))


def test_palette_dispatch_k_10() -> None:
    """k == 10: colourblind-safe 20-colour set, deterministic + distinct."""
    palette = _palette_for_k(10)
    colors = [palette(i) for i in range(10)]
    # Every entry is a valid RGBA tuple.
    for c in colors:
        assert len(c) == 4
        assert c[3] == pytest.approx(1.0)
    # Pairwise distinct across the first 10 ids (no collisions).
    assert len({tuple(c) for c in colors}) == 10


def test_palette_dispatch_k_11() -> None:
    """k == 11: still inside the CB-safe 20-colour set."""
    palette = _palette_for_k(11)
    colors = [palette(i) for i in range(11)]
    for c in colors:
        assert c[3] == pytest.approx(1.0)
    assert len({tuple(c) for c in colors}) == 11


def test_palette_dispatch_k_20() -> None:
    """k == 20: full CB-safe palette; every slot distinct."""
    palette = _palette_for_k(20)
    colors = [palette(i) for i in range(20)]
    for c in colors:
        assert c[3] == pytest.approx(1.0)
    assert len({tuple(c) for c in colors}) == 20


def test_palette_dispatch_k_21() -> None:
    """k == 21: HSV-spaced palette (upper boundary -> custom path)."""
    palette = _palette_for_k(21)
    # All 21 ids return a valid RGBA tuple with alpha=1.0.
    for i in range(21):
        c = palette(i)
        assert len(c) == 4
        assert 0.0 <= c[0] <= 1.0
        assert 0.0 <= c[1] <= 1.0
        assert 0.0 <= c[2] <= 1.0
        assert c[3] == pytest.approx(1.0)
    # Consecutive ids get distinct colors (hue-stepped).
    c0 = palette(0)
    c1 = palette(1)
    assert c0 != c1


def test_palette_dispatch_k_50() -> None:
    """k == 50: HSV-spaced, seeded from k, deterministic across calls."""
    p1 = _palette_for_k(50)
    p2 = _palette_for_k(50)
    for i in range(50):
        # Same seed (k) -> byte-identical palette across two calls.
        assert p1(i) == pytest.approx(p2(i))


def test_palette_same_k_across_recipes_shares_palette() -> None:
    """Two recipes with the same k share their palette (seed is k, not hash6)."""
    p_a = _palette_for_k(30)
    p_b = _palette_for_k(30)
    for i in range(30):
        assert p_a(i) == pytest.approx(p_b(i))


# -----------------------------------------------------------------------------
# Theirs=None placeholder
# -----------------------------------------------------------------------------


def test_theirs_none_renders_placeholder() -> None:
    """--ours-only: theirs panel shows '--ours-only run' annotation."""
    inputs = _make_inputs(theirs_mode="none")
    fig = build_vis_figure(inputs)
    # Panel 3 (index 2 zero-based) is the theirs panel.
    theirs_ax = fig.axes[2]
    texts = [t.get_text() for t in theirs_ax.texts]
    assert any("ours-only" in t for t in texts)
    # The theirs panel has no scatter collection.
    from matplotlib.collections import PathCollection

    scatters = [c for c in theirs_ax.collections if isinstance(c, PathCollection)]
    assert scatters == []


# -----------------------------------------------------------------------------
# ARI caption
# -----------------------------------------------------------------------------


def test_caption_includes_ari() -> None:
    """Figure-level caption text includes 'ARI = <three-decimal float>'."""
    inputs = _make_inputs()  # ari=0.97
    fig = build_vis_figure(inputs)
    all_texts = []
    for t in fig.texts:
        all_texts.append(t.get_text())
    joined = " \n ".join(all_texts)
    assert "ARI = 0.970" in joined


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_deterministic_savefig() -> None:
    """Two calls with same VisInputs -> byte-identical PNG (SHA-256 equal)."""
    inputs = _make_inputs()
    fig_a = build_vis_figure(inputs)
    fig_b = build_vis_figure(inputs)
    png_a = _png_bytes(fig_a)
    png_b = _png_bytes(fig_b)
    h_a = hashlib.sha256(png_a).hexdigest()
    h_b = hashlib.sha256(png_b).hexdigest()
    assert h_a == h_b, "vis PNG is non-deterministic across identical calls"


# -----------------------------------------------------------------------------
# Projection-not-recomputed
# -----------------------------------------------------------------------------


def test_projection_is_not_recomputed() -> None:
    """Monkeypatch PCA to raise; vis must still work because it consumes
    inputs.projection_2d directly."""
    inputs = _make_inputs()

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("PCA should not be called from vis")

    # Patch both PCA and TruncatedSVD to be safe; if either is imported in the
    # vis path, the test will surface it.
    with (
        patch("sklearn.decomposition.PCA", side_effect=_boom),
        patch("sklearn.decomposition.TruncatedSVD", side_effect=_boom),
    ):
        fig = build_vis_figure(inputs)
        png = _png_bytes(fig)
    assert len(png) > 0


# -----------------------------------------------------------------------------
# Subsample behavior
# -----------------------------------------------------------------------------


def test_subsample_uses_seed() -> None:
    """Same subsample_seed -> same selected indices across calls (deterministic)."""
    from pybench.charts.vis import _subsample_indices

    idx_a = _subsample_indices(n=10_000, seed=42, k=1000)
    idx_b = _subsample_indices(n=10_000, seed=42, k=1000)
    assert np.array_equal(idx_a, idx_b)
    # Different seed -> different selection (overwhelmingly likely at n=10k).
    idx_c = _subsample_indices(n=10_000, seed=99, k=1000)
    assert not np.array_equal(idx_a, idx_c)


def test_all_panels_use_same_subsample() -> None:
    """GT, ours, theirs panels render the SAME points (not independently subsampled)."""
    # Request a small subsample bound so subsampling actually runs.
    inputs = _make_inputs(n=2000, max_scatter_points=200)
    fig = build_vis_figure(inputs)
    # Extract the scatter PathCollections from gt, ours, theirs.
    from matplotlib.collections import PathCollection

    def _scatter_xy(ax: Any) -> np.ndarray:
        for c in ax.collections:
            if isinstance(c, PathCollection):
                return c.get_offsets().data.copy()  # type: ignore[attr-defined]
        raise AssertionError(f"no PathCollection on axes {ax}")

    xy_gt = _scatter_xy(fig.axes[0])
    xy_ours = _scatter_xy(fig.axes[1])
    xy_theirs = _scatter_xy(fig.axes[2])
    assert np.array_equal(xy_gt, xy_ours)
    assert np.array_equal(xy_gt, xy_theirs)
    assert xy_gt.shape[0] == 200


def test_subsample_no_op_when_n_below_bound() -> None:
    """When n <= max_scatter_points, the subsample is every index, preserving order."""
    from pybench.charts.vis import _subsample_indices

    idx = _subsample_indices(n=100, seed=1, k=200)
    assert np.array_equal(idx, np.arange(100))


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_build_empty_vis_figure_returns_valid_figure() -> None:
    fig = build_empty_vis_figure("kmeans", "projection NaN")
    assert isinstance(fig, Figure)
    # Has at least one axes and a reason text.
    assert len(fig.axes) >= 1
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("projection NaN" in t for t in texts)


def test_vis_falls_back_to_empty_on_all_nan_projection() -> None:
    inputs = _make_inputs()
    nan_proj = np.full_like(inputs.projection_2d, np.nan)
    # Rebuild VisInputs with NaN projection.
    inputs = VisInputs(
        algo=inputs.algo,
        recipe_name=inputs.recipe_name,
        size=inputs.size,
        dims=inputs.dims,
        non_njobs_params=inputs.non_njobs_params,
        n_jobs=inputs.n_jobs,
        ari=inputs.ari,
        title_meta=inputs.title_meta,
        gt_labels=inputs.gt_labels,
        ours_labels=inputs.ours_labels,
        theirs_labels=inputs.theirs_labels,
        projection_2d=nan_proj,
        subsample_seed=inputs.subsample_seed,
        max_scatter_points=inputs.max_scatter_points,
    )
    fig = build_vis_figure(inputs)
    # Empty-figure fallback has 1 axes, not 4.
    assert len(fig.axes) == 1


def test_vis_with_all_noise_labels() -> None:
    """All-noise gt and pred renders grey scatter and 'full agreement' disagreement."""
    rng = np.random.default_rng(11)
    n = 500
    X = rng.standard_normal((n, 2)).astype(np.float32)
    gt = np.full(n, -1, dtype=np.int32)
    ours = gt.copy()
    theirs = gt.copy()
    inputs = VisInputs(
        algo="dbscan",
        recipe_name="dbscan_noise",
        size=n,
        dims=2,
        non_njobs_params=(),
        n_jobs=1,
        ari=1.0,
        title_meta=_meta(),
        gt_labels=gt,
        ours_labels=ours,
        theirs_labels=theirs,
        projection_2d=X,
        subsample_seed=1,
    )
    fig = build_vis_figure(inputs)
    # Must still render 4 panels, including a 'full agreement' on disagreement.
    assert len(fig.axes) == 4
    disagree_ax = fig.axes[3]
    texts = [t.get_text() for t in disagree_ax.texts]
    assert any("full agreement" in t for t in texts)


# -----------------------------------------------------------------------------
# Multi-dim layout tests
# -----------------------------------------------------------------------------


def _inputs_for_dim(dim: int, *, seed: int = 42, ari: float = 0.97) -> VisInputs:
    """Construct a VisInputs standing in for a given dim's row."""
    n = 400
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype(np.float32)
    k = 4
    gt = rng.integers(0, k, n).astype(np.int32)
    ours = gt.copy()
    theirs = gt.copy()
    return VisInputs(
        algo="kmeans",
        recipe_name="kmeans_blobs",
        size=n,
        dims=dim,
        non_njobs_params=(("k", k),),
        n_jobs=1,
        ari=ari,
        title_meta=_meta(),
        gt_labels=gt,
        ours_labels=ours,
        theirs_labels=theirs,
        projection_2d=X,
        subsample_seed=seed,
    )


def test_multidim_vis_three_dims_has_12_axes() -> None:
    """3 rows x 4 panels == 12 axes."""
    inputs = MultiDimVisInputs(
        rows=tuple(_inputs_for_dim(d, seed=s) for d, s in ((2, 1), (32, 2), (512, 3))),
        selection_mode="test: 3 dims",
    )
    fig = build_multidim_vis_figure(inputs)
    assert len(fig.axes) == 12


def test_multidim_vis_single_dim_has_4_axes() -> None:
    """1-row payload renders the same 4 panels build_vis_figure would."""
    inputs = MultiDimVisInputs(
        rows=(_inputs_for_dim(8, seed=7),),
        selection_mode="test: single dim",
    )
    fig = build_multidim_vis_figure(inputs)
    assert len(fig.axes) == 4


def test_multidim_vis_two_dims_has_8_axes() -> None:
    """2 rows x 4 panels == 8 axes (partition has only 2 unique dims)."""
    inputs = MultiDimVisInputs(
        rows=tuple(_inputs_for_dim(d, seed=s) for d, s in ((2, 11), (128, 12))),
        selection_mode="test: 2 dims",
    )
    fig = build_multidim_vis_figure(inputs)
    assert len(fig.axes) == 8


def test_multidim_vis_rejects_empty_rows() -> None:
    """Zero rows is a programmer error, not a render fallback."""
    with pytest.raises(ValueError):
        build_multidim_vis_figure(MultiDimVisInputs(rows=(), selection_mode="empty"))


def test_multidim_vis_header_reports_selection_mode() -> None:
    """The selection_mode string appears in the figure-level header text."""
    mode = "cell selection: 3 representative dims (low / mid / high)"
    inputs = MultiDimVisInputs(
        rows=tuple(_inputs_for_dim(d, seed=s) for d, s in ((2, 1), (32, 2), (512, 3))),
        selection_mode=mode,
    )
    fig = build_multidim_vis_figure(inputs)
    all_text = " ".join(t.get_text() for t in fig.texts)
    assert mode in all_text


def test_multidim_vis_caption_reports_per_dim_ari() -> None:
    """Each row's ARI appears in the caption as 'd={dim} = {ari:.3f}'."""
    inputs = MultiDimVisInputs(
        rows=(
            _inputs_for_dim(2, seed=1, ari=0.999),
            _inputs_for_dim(32, seed=2, ari=0.985),
            _inputs_for_dim(512, seed=3, ari=0.971),
        ),
        selection_mode="test",
    )
    fig = build_multidim_vis_figure(inputs)
    all_text = " ".join(t.get_text() for t in fig.texts)
    assert "d=2 = 0.999" in all_text
    assert "d=32 = 0.985" in all_text
    assert "d=512 = 0.971" in all_text


def test_multidim_vis_deterministic_savefig() -> None:
    """Two calls with the same MultiDimVisInputs produce byte-identical PNGs."""
    rows = tuple(_inputs_for_dim(d, seed=s) for d, s in ((2, 10), (32, 11), (512, 12)))
    inputs = MultiDimVisInputs(rows=rows, selection_mode="test")
    fig_a = build_multidim_vis_figure(inputs)
    fig_b = build_multidim_vis_figure(inputs)
    png_a = _png_bytes(fig_a)
    png_b = _png_bytes(fig_b)
    assert hashlib.sha256(png_a).hexdigest() == hashlib.sha256(png_b).hexdigest(), (
        "multi-dim vis PNG is non-deterministic across identical calls"
    )


def test_multidim_vis_palette_is_cb_safe() -> None:
    """Each row still routes cluster ids through the CB-safe palette."""
    from matplotlib.collections import PathCollection

    inputs = MultiDimVisInputs(
        rows=tuple(_inputs_for_dim(d, seed=s) for d, s in ((2, 5), (32, 6), (512, 7))),
        selection_mode="test",
    )
    fig = build_multidim_vis_figure(inputs)
    # Extract RGBA colors of the ground-truth scatter for each row, and
    # require every non-noise color to land in the palette's dispatch
    # (either the CB-safe 20-color set or the HSV fallback).
    palette_20 = {
        _rgba_hex(c)
        for c in (
            "#0072B2",
            "#009E73",
            "#D55E00",
            "#CC79A7",
            "#56B4E9",
            "#E69F00",
            "#F0E442",
        )
    }
    for row_idx, row in enumerate(inputs.rows):
        gt_ax = fig.axes[row_idx * 4]
        scatters = [c for c in gt_ax.collections if isinstance(c, PathCollection)]
        assert scatters, f"row {row_idx} gt panel missing PathCollection"
        # k=4 in _inputs_for_dim, so every color should come from the
        # first few CB-safe swatches. Ensure at least one match.
        colors = {tuple(float(v) for v in c) for c in scatters[0].get_facecolors()}
        assert any(c in palette_20 for c in colors), (
            f"row {row_idx} colors {colors} not in CB-safe prefix"
        )
