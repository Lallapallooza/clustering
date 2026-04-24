"""Pure-function 4-panel visualization of cluster label comparisons.

Given a fixture, ground-truth labels, our predicted labels, a competitor's
predicted labels, and a pre-computed 2-D projection of the fixture, build a
Matplotlib Figure with four side-by-side panels:

1. Ground truth -- scatter colored by true cluster IDs.
2. Ours        -- same points, colored by our predicted IDs after Hungarian
                  alignment against the ground truth.
3. Theirs      -- same points, colored by the competitor's predicted IDs
                  after alignment. Falls back to a placeholder panel when
                  the run was ``--ours-only``.
4. Disagreement-- hexbin density over the FULL point set (not subsampled)
                  showing where aligned-ours or aligned-theirs differs from
                  the ground truth. A "full agreement" annotation replaces
                  the hexbin when zero points disagree.

The Figure is produced by the Matplotlib OO API (``Figure`` +
``FigureCanvasAgg``); no ``matplotlib.pyplot`` is imported. Two calls with the
same :class:`VisInputs` produce byte-identical PNGs via ``savefig(BytesIO)``.
The projection is NEVER recomputed here; ``inputs.projection_2d`` is consumed
as-is so the output stays stable across dependency-version drift on the
projection estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pybench.charts.label_align import align_to_ground_truth
from pybench.charts.meta import RunMetadata

# Chrome colors mirror pybench/charts/figure.py's palette so the vis and
# absolute charts read as one design language. The two modules are kept
# independent; we duplicate these few constants rather than coupling via an
# import.
_TEXT = "#111419"
_MUTED = "#4E5866"
_AXIS = "#2A313A"
_GRID = "#E4E8ED"

# Noise marker color. Consistent grey across every panel so the eye learns
# "grey = noise" even when the palette is different per k.
_NOISE_COLOR = "#BFC4C9"

# Hexbin grid resolution tuned for the fixed 3.6"-per-panel layout: at this
# width, gridsize=60 keeps a localized ~1% disagreement region visible as a
# hot spot; larger values wash out the signal, smaller values smear across
# cluster cores. Retune if the panel width changes.
_HEXBIN_GRIDSIZE = 60

# Visual subsample bound: scatter plots of >20k points overwhelm the eye and
# the PNG. The subsample is deterministic, seeded from the partition hash.
_DEFAULT_MAX_SCATTER_POINTS = 20_000


@dataclass(frozen=True, slots=True)
class VisInputs:
    """Full input payload for :func:`build_vis_figure`.

    All arrays must be pre-validated by the caller: dtypes and shapes are
    trusted here. The partition-level ``subsample_seed`` is hashed from the
    chart ``hash6`` in the CLI layer so two invocations on the same
    partition produce byte-identical figures.
    """

    algo: str
    recipe_name: str
    size: int
    dims: int
    non_njobs_params: tuple[tuple[str, Any], ...]
    n_jobs: int
    ari: float
    title_meta: RunMetadata
    gt_labels: np.ndarray
    ours_labels: np.ndarray
    theirs_labels: np.ndarray | None
    projection_2d: np.ndarray
    subsample_seed: int
    max_scatter_points: int = _DEFAULT_MAX_SCATTER_POINTS
    # Optional cross-reference to the companion bench chart PNG. The CLI
    # passes the partition's ``chart_filename`` so the caption reads as a
    # single pointer. Kept optional with a sentinel default so direct
    # construction (e.g. in tests) doesn't need to know the hash6.
    bench_filename: str | None = None


# -----------------------------------------------------------------------------
# Palette dispatch
# -----------------------------------------------------------------------------


_GLASBEY_CB_SAFE_20: tuple[str, ...] = (
    # Glasbey-style colourblind-safe 20-colour palette, distinct under
    # deuteranopia/protanopia when viewed as cluster swatches against white.
    # Starts with the Okabe-Ito core (the 7 "safe" hues) then fills with
    # luminance-spread extensions so consecutive ids stay perceptually
    # distinct through the tail.
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#332288",  # indigo
    "#117733",  # forest
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # magenta
    "#DDCC77",  # sand
    "#88CCEE",  # pale cyan
    "#661100",  # maroon
    "#6699CC",  # mid-blue
    "#AA4466",  # rose
    "#4477AA",  # steel blue
    "#228833",  # dark green
)


def _palette_for_k(k: int) -> Callable[[int], tuple[float, float, float, float]]:
    """Return a function mapping a cluster id in ``[0, k)`` to an RGBA color.

    Dispatch rules:
    - ``k == 0``: everything grey (all-noise fixture).
    - ``0 < k <= 20``: Glasbey-style colourblind-safe 20-colour set. The
      leading prefix is the Okabe-Ito core, so the first 7 ids match the
      bench chart's series palette.
    - ``k > 20``: deterministic HSV-spaced palette seeded from ``k`` with
      luminance pushed darker so cluster colours remain distinct under
      colourblindness. Two recipes with the same ``k`` share their palette.

    The returned callable takes an integer id and returns an RGBA tuple.
    Noise (``-1``) is never routed through this function; callers special-case
    it to ``_NOISE_COLOR``.
    """
    if k <= 0:
        return lambda _i: _rgba_hex(_NOISE_COLOR)
    if k <= len(_GLASBEY_CB_SAFE_20):
        palette = _GLASBEY_CB_SAFE_20
        return lambda i: _rgba_hex(palette[int(i) % len(palette)])

    # Deterministic HSV palette. Seeding from ``k`` (NOT hash6) means
    # different recipes with the same cluster count share the same palette,
    # which is useful for visually comparing sibling runs. Value is capped
    # below 0.75 so the darker luminance keeps CB contrast on white.
    rng = np.random.default_rng(seed=int(k))
    # Evenly-spaced hues, randomly-offset so consecutive ids get visually
    # distinct colors.
    hues = (np.arange(k) / k + rng.random()) % 1.0
    # Saturation high; value in [0.55, 0.72] so cluster fills read as a
    # saturated mid-luminance family under deuteranopia/protanopia.
    sat = 0.70 + 0.20 * rng.random(k)
    val = 0.55 + 0.17 * rng.random(k)
    hsv = np.stack([hues, sat, val], axis=1)
    rgb = _hsv_to_rgb(hsv)
    rgba = np.concatenate([rgb, np.ones((k, 1))], axis=1)

    def _lookup(i: int) -> tuple[float, float, float, float]:
        idx = int(i) % k
        return tuple(float(v) for v in rgba[idx])  # type: ignore[return-value]

    return _lookup


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB conversion. ``hsv`` shape: (n, 3) in [0, 1]."""
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]
    i = np.floor(h * 6.0).astype(np.int64)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    rgb = np.zeros((hsv.shape[0], 3))
    idx0 = i == 0
    idx1 = i == 1
    idx2 = i == 2
    idx3 = i == 3
    idx4 = i == 4
    idx5 = i == 5
    rgb[idx0] = np.stack([v[idx0], t[idx0], p[idx0]], axis=1)
    rgb[idx1] = np.stack([q[idx1], v[idx1], p[idx1]], axis=1)
    rgb[idx2] = np.stack([p[idx2], v[idx2], t[idx2]], axis=1)
    rgb[idx3] = np.stack([p[idx3], q[idx3], v[idx3]], axis=1)
    rgb[idx4] = np.stack([t[idx4], p[idx4], v[idx4]], axis=1)
    rgb[idx5] = np.stack([v[idx5], p[idx5], q[idx5]], axis=1)
    return rgb


def _rgba_hex(hex_color: str) -> tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, 1.0)


# -----------------------------------------------------------------------------
# Layout + chrome
# -----------------------------------------------------------------------------


def _style_panel(ax: Any, *, title: str) -> None:
    """Apply shared panel chrome (title, tick colors, spine thinning)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        which="major",
        color=_AXIS,
        labelcolor=_TEXT,
        labelsize=8.5,
        length=2.5,
        width=0.6,
    )
    ax.grid(
        which="major",
        axis="both",
        color=_GRID,
        linestyle="-",
        linewidth=0.5,
        alpha=1.0,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.set_title(
        title,
        fontsize=11.0,
        fontweight="bold",
        color=_TEXT,
        pad=5,
        loc="left",
    )


def _subsample_indices(n: int, seed: int, k: int) -> np.ndarray:
    """Deterministic subsample of ``[0, n)`` of size ``min(n, k)``."""
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed=int(seed))
    return rng.choice(n, size=int(k), replace=False)


def _colors_for_labels(
    labels: np.ndarray, palette: Callable[[int], tuple[float, float, float, float]]
) -> np.ndarray:
    """Build an (n, 4) RGBA array from integer labels. ``-1`` -> ``_NOISE_COLOR``."""
    noise = _rgba_hex(_NOISE_COLOR)
    rgba = np.empty((labels.shape[0], 4), dtype=np.float64)
    for i, label in enumerate(labels):
        lbl = int(label)
        if lbl < 0:
            rgba[i] = noise
        else:
            rgba[i] = palette(lbl)
    return rgba


def _xy_limits(
    projection: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute shared xlim/ylim from the projection with 5% padding on each side.

    Returns ``((xlo, xhi), (ylo, yhi))``. Falls back to ``(-1, 1)`` on axes
    where ``projection`` is degenerate (single point, all-NaN, etc.).
    """
    finite = np.isfinite(projection).all(axis=1)
    if not finite.any():
        return (-1.0, 1.0), (-1.0, 1.0)
    pts = projection[finite]
    x_lo, y_lo = pts.min(axis=0)
    x_hi, y_hi = pts.max(axis=0)
    x_range = float(x_hi - x_lo)
    y_range = float(y_hi - y_lo)
    if x_range == 0.0:
        x_lo -= 1.0
        x_hi += 1.0
        x_range = 2.0
    if y_range == 0.0:
        y_lo -= 1.0
        y_hi += 1.0
        y_range = 2.0
    x_pad = 0.05 * x_range
    y_pad = 0.05 * y_range
    return (
        (float(x_lo) - x_pad, float(x_hi) + x_pad),
        (float(y_lo) - y_pad, float(y_hi) + y_pad),
    )


def _effective_k(*label_arrays: np.ndarray) -> int:
    """Count unique non-noise ids across the supplied arrays."""
    seen: set[int] = set()
    for arr in label_arrays:
        if arr is None or arr.size == 0:
            continue
        uniq = np.unique(arr)
        for v in uniq:
            iv = int(v)
            if iv != -1:
                seen.add(iv)
    return len(seen)


def _format_non_njobs_params(pairs: tuple[tuple[str, Any], ...]) -> str:
    if not pairs:
        return ""
    return "  |  ".join(f"{k}={v}" for k, v in pairs)


def _draw_caption(
    fig: Figure,
    *,
    algo: str,
    recipe_name: str,
    size: int,
    dims: int,
    n_jobs: int,
    ari: float,
    non_njobs_params: tuple[tuple[str, Any], ...],
    title_meta: RunMetadata,
    bench_filename: str | None = None,
) -> None:
    """Figure-level title + caption block.

    When ``algo`` and ``recipe_name`` are equal (the common case) the title
    emits the name once. The parameter row sits below the title without
    ``ARI``; ``ARI`` moves into the caption bar so it stops colliding with
    ``k={n_clusters}`` on wide parameter rows. The upper-right
    "ground truth / ours / theirs / disagreement" stamp is dropped because
    it duplicates the per-panel titles that already label each column.
    """
    if algo == recipe_name:
        title_text = f"{algo}  |  n={size}  d={dims}  n_jobs={n_jobs}"
    else:
        title_text = f"{algo}  |  {recipe_name}  |  n={size}  d={dims}  n_jobs={n_jobs}"
    fig.text(
        0.02,
        0.965,
        title_text,
        fontsize=12.5,
        fontweight="bold",
        color=_TEXT,
        ha="left",
        va="top",
    )
    param_summary = _format_non_njobs_params(non_njobs_params)
    if param_summary:
        fig.text(
            0.02,
            0.925,
            param_summary,
            fontsize=10.5,
            color=_MUTED,
            ha="left",
            va="top",
        )
    # Cell selection hint: tells the reader which (size, dims, n_jobs) this
    # vis represents and that the default pick is largest-n x smallest-d.
    # Placed to the right of the param row so it does not collide with the
    # panel titles on a tight layout.
    fig.text(
        0.98,
        0.925,
        f"cell: n={size}, d={dims}, n_jobs={n_jobs}"
        "  (default: largest size x smallest dim)",
        fontsize=9.5,
        color=_MUTED,
        ha="right",
        va="top",
    )
    caption = (
        f"ARI = {ari:.3f}    |    git  {title_meta.git_sha}    |    "
        f"{title_meta.machine}    |    {title_meta.timestamp_iso}"
    )
    fig.text(
        0.5,
        0.055,
        caption,
        fontsize=9.0,
        color=_MUTED,
        ha="center",
        va="bottom",
    )
    if bench_filename:
        fig.text(
            0.5,
            0.025,
            f"bench: {bench_filename}",
            fontsize=8.5,
            color=_MUTED,
            ha="center",
            va="bottom",
        )


def _draw_multidim_header(
    fig: Figure,
    *,
    algo: str,
    recipe_name: str,
    non_njobs_params: tuple[tuple[str, Any], ...],
    selection_mode: str,
    header_top: float,
) -> None:
    """Header for a multi-dim vis figure: title + params + selection mode.

    ``header_top`` is the normalized y in figure coords where the title sits;
    lines are offset downward from it. Keeping the offsets fixed means taller
    figures (more rows) do not stretch the header block vertically.
    """
    if algo == recipe_name:
        title_text = f"{algo}  |  {_format_non_njobs_params(non_njobs_params)}".rstrip(
            "  |  "
        )
    else:
        title_text = (
            f"{algo}  |  {recipe_name}  |  {_format_non_njobs_params(non_njobs_params)}"
        ).rstrip("  |  ")
    fig.text(
        0.02,
        header_top,
        title_text,
        fontsize=12.5,
        fontweight="bold",
        color=_TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.98,
        header_top,
        selection_mode,
        fontsize=9.5,
        color=_MUTED,
        ha="right",
        va="top",
    )


def _draw_multidim_caption(
    fig: Figure,
    *,
    per_dim_ari: tuple[tuple[int, float], ...],
    title_meta: RunMetadata,
    bench_filename: str | None,
    caption_bottom: float,
) -> None:
    """Caption block for a multi-dim figure: per-dim ARI + provenance.

    ``caption_bottom`` anchors the primary caption row in figure coords; the
    optional ``bench:`` cross-reference sits just below it.
    """
    ari_parts = ", ".join(f"d={dim} = {ari:.3f}" for dim, ari in per_dim_ari)
    caption = (
        f"ARI per dim: {ari_parts}    |    git  {title_meta.git_sha}    |    "
        f"{title_meta.machine}    |    {title_meta.timestamp_iso}"
    )
    fig.text(
        0.5,
        caption_bottom,
        caption,
        fontsize=9.0,
        color=_MUTED,
        ha="center",
        va="bottom",
    )
    if bench_filename:
        fig.text(
            0.5,
            caption_bottom - 0.015,
            f"bench: {bench_filename}",
            fontsize=8.5,
            color=_MUTED,
            ha="center",
            va="bottom",
        )


# -----------------------------------------------------------------------------
# Panel renderers
# -----------------------------------------------------------------------------


def _render_scatter_panel(
    ax: Any,
    *,
    title: str,
    projection: np.ndarray,
    labels: np.ndarray,
    subsample: np.ndarray,
    palette: Callable[[int], tuple[float, float, float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_axis_labels: bool = False,
) -> None:
    _style_panel(ax, title=title)
    pts = projection[subsample]
    lbls = labels[subsample]
    colors = _colors_for_labels(lbls, palette)
    # s=6 for legibility at n=20k; too small and points vanish on light
    # backgrounds, too large and overlapping cores hide structure.
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=6,
        c=colors,
        marker="o",
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if show_axis_labels:
        ax.set_xlabel(
            "projection dim 1",
            fontsize=9.5,
            color=_TEXT,
            labelpad=4,
        )
        ax.set_ylabel(
            "projection dim 2",
            fontsize=9.5,
            color=_TEXT,
            labelpad=4,
        )


def _render_theirs_placeholder(ax: Any) -> None:
    _style_panel(ax, title="Theirs")
    ax.text(
        0.5,
        0.5,
        "--ours-only run",
        transform=ax.transAxes,
        fontsize=11.0,
        color=_MUTED,
        ha="center",
        va="center",
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _render_disagreement_panel(
    ax: Any,
    *,
    projection: np.ndarray,
    disagree_mask: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    _style_panel(ax, title="Disagreement")
    if not disagree_mask.any():
        ax.text(
            0.5,
            0.5,
            "full agreement",
            transform=ax.transAxes,
            fontsize=11.0,
            color=_MUTED,
            ha="center",
            va="center",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return
    disagree_pts = projection[disagree_mask]
    # viridis reads well on white with print-safe contrast. mincnt=1 so the
    # empty background stays white instead of rendering a lowest-bin tile.
    ax.hexbin(
        disagree_pts[:, 0],
        disagree_pts[:, 1],
        gridsize=_HEXBIN_GRIDSIZE,
        cmap="viridis",
        mincnt=1,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
    )
    # Overlay the disagreement count + percentage so the reader can tell a
    # dense hotspot from a diffuse smear without reading off a colorbar.
    disagree_n = int(disagree_mask.sum())
    total = int(disagree_mask.size)
    pct = disagree_n / total if total else 0.0
    ax.text(
        0.02,
        0.98,
        f"{disagree_n} disagreements ({pct:.3%})",
        transform=ax.transAxes,
        fontsize=9.5,
        color=_TEXT,
        ha="left",
        va="top",
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------


def _render_vis_row(
    *,
    ax_gt: Any,
    ax_ours: Any,
    ax_theirs: Any,
    ax_disagree: Any,
    inputs: VisInputs,
    show_axis_labels: bool,
    panel_titles: tuple[str, str, str, str],
) -> None:
    """Render one row of 4 vis panels into the supplied axes.

    Factors out the per-cell rendering that ``build_vis_figure`` and
    ``build_multidim_vis_figure`` both need. Hungarian alignment, the shared
    palette, limits, subsample indices, and the disagreement mask are all
    derived from ``inputs`` alone so each row is independently valid.
    """
    projection = inputs.projection_2d
    aligned_ours, _ = align_to_ground_truth(inputs.gt_labels, inputs.ours_labels)
    if inputs.theirs_labels is not None:
        aligned_theirs, _ = align_to_ground_truth(
            inputs.gt_labels, inputs.theirs_labels
        )
    else:
        aligned_theirs = None

    if aligned_theirs is not None:
        k = _effective_k(inputs.gt_labels, aligned_ours, aligned_theirs)
    else:
        k = _effective_k(inputs.gt_labels, aligned_ours)
    palette = _palette_for_k(k)

    xlim, ylim = _xy_limits(projection)
    subsample = _subsample_indices(
        projection.shape[0], inputs.subsample_seed, inputs.max_scatter_points
    )
    disagree = _disagreement_mask(inputs.gt_labels, aligned_ours, aligned_theirs)

    _render_scatter_panel(
        ax_gt,
        title=panel_titles[0],
        projection=projection,
        labels=inputs.gt_labels,
        subsample=subsample,
        palette=palette,
        xlim=xlim,
        ylim=ylim,
        show_axis_labels=show_axis_labels,
    )
    _render_scatter_panel(
        ax_ours,
        title=panel_titles[1],
        projection=projection,
        labels=aligned_ours,
        subsample=subsample,
        palette=palette,
        xlim=xlim,
        ylim=ylim,
    )
    if aligned_theirs is not None:
        _render_scatter_panel(
            ax_theirs,
            title=panel_titles[2],
            projection=projection,
            labels=aligned_theirs,
            subsample=subsample,
            palette=palette,
            xlim=xlim,
            ylim=ylim,
        )
    else:
        _render_theirs_placeholder(ax_theirs)
    _render_disagreement_panel(
        ax_disagree,
        projection=projection,
        disagree_mask=disagree,
        xlim=xlim,
        ylim=ylim,
    )


def build_vis_figure(inputs: VisInputs) -> Figure:
    """Pure function: build and return a Matplotlib Figure. No I/O."""
    projection = inputs.projection_2d
    if projection.size == 0 or not np.isfinite(projection).all():
        return build_empty_vis_figure(
            inputs.algo, "projection is empty or contains non-finite values"
        )

    # 1x4 horizontal layout: ground truth, ours, theirs, disagreement. Figure
    # dimensions scale with the panel count so each panel keeps a consistent
    # aspect ratio. Height bumped from 4.4 to 5.0 so the three-line header
    # (title / params / cell-selection hint) clears the panel titles; bottom
    # lifted to 0.15 so the two caption rows (ARI+git / bench xref) don't
    # clip the panels.
    fig = Figure(figsize=(14.4, 5.0), facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        left=0.04,
        right=0.98,
        top=0.82,
        bottom=0.16,
        wspace=0.12,
    )
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_ours = fig.add_subplot(gs[0, 1])
    ax_theirs = fig.add_subplot(gs[0, 2])
    ax_disagree = fig.add_subplot(gs[0, 3])

    _render_vis_row(
        ax_gt=ax_gt,
        ax_ours=ax_ours,
        ax_theirs=ax_theirs,
        ax_disagree=ax_disagree,
        inputs=inputs,
        show_axis_labels=True,
        panel_titles=("Ground truth", "Ours", "Theirs", "Disagreement"),
    )

    _draw_caption(
        fig,
        algo=inputs.algo,
        recipe_name=inputs.recipe_name,
        size=inputs.size,
        dims=inputs.dims,
        n_jobs=inputs.n_jobs,
        ari=inputs.ari,
        non_njobs_params=inputs.non_njobs_params,
        title_meta=inputs.title_meta,
        bench_filename=inputs.bench_filename,
    )
    return fig


@dataclass(frozen=True, slots=True)
class MultiDimVisInputs:
    """Bundle of :class:`VisInputs` for multi-dim vis rendering.

    Each element of ``rows`` is the full payload for one dim's row of four
    panels. The per-row :class:`VisInputs` already carries its own ``dims``,
    ``size``, ``n_jobs``, and ``ari``; ``selection_mode`` is a human-readable
    string (e.g. ``"3 representative dims (low / mid / high)"``) displayed in
    the figure header so readers know how the dims shown were picked.

    All rows must agree on ``algo``, ``recipe_name``, ``non_njobs_params``,
    ``title_meta``, and ``bench_filename`` -- those drive the figure-level
    header/caption that the rows share. Per-dim arrays (labels, projection)
    are independent.
    """

    rows: tuple[VisInputs, ...]
    selection_mode: str


# Layout constants for the multi-dim figure. Height scales linearly with the
# row count; header/caption bands stay fixed in absolute inches so they do
# not stretch as rows are added.
_MULTIDIM_ROW_HEIGHT_IN = 4.4
_MULTIDIM_HEADER_IN = 0.9
_MULTIDIM_CAPTION_IN = 0.7
_MULTIDIM_WIDTH_IN = 14.4
_MULTIDIM_MAX_ROWS = 10


def build_multidim_vis_figure(inputs: MultiDimVisInputs) -> Figure:
    """Render a multi-row vis figure with one row of 4 panels per dim.

    Each row gets its own :class:`VisInputs` via ``inputs.rows``. The figure
    height scales with the row count so every panel keeps the same aspect
    ratio; a single-row payload produces a figure visually equivalent to
    :func:`build_vis_figure` but with a dim-aware header and per-dim ARI in
    the caption. An empty ``rows`` tuple is rejected with ``ValueError``.

    When any row's projection is empty or contains non-finite values, the
    figure falls back to :func:`build_empty_vis_figure` for the whole figure.
    """
    if not inputs.rows:
        raise ValueError("MultiDimVisInputs.rows must be non-empty")
    for row in inputs.rows:
        proj = row.projection_2d
        if proj.size == 0 or not np.isfinite(proj).all():
            return build_empty_vis_figure(
                row.algo,
                f"projection for d={row.dims} is empty or contains non-finite values",
            )

    n_rows = len(inputs.rows)
    fig_h = (
        n_rows * _MULTIDIM_ROW_HEIGHT_IN + _MULTIDIM_HEADER_IN + _MULTIDIM_CAPTION_IN
    )
    fig = Figure(figsize=(_MULTIDIM_WIDTH_IN, fig_h), facecolor="white")
    FigureCanvasAgg(fig)

    # Normalize header/caption bands to figure coords; panel grid occupies
    # the middle band. Using absolute-inch offsets keeps the header and
    # caption at the same visual weight regardless of row count.
    header_top_abs = _MULTIDIM_HEADER_IN
    caption_h_abs = _MULTIDIM_CAPTION_IN
    top_frac = 1.0 - (0.35 * _MULTIDIM_HEADER_IN) / fig_h
    # The gridspec's top reserves enough room for row-level titles + header;
    # bottom reserves the caption band.
    panel_top = 1.0 - header_top_abs / fig_h
    panel_bottom = caption_h_abs / fig_h

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=4,
        left=0.06,
        right=0.98,
        top=panel_top,
        bottom=panel_bottom,
        wspace=0.12,
        hspace=0.35,
    )

    first = inputs.rows[0]
    # Shared header: algo + params + selection mode. Positioned at the top of
    # the figure. ``top_frac`` is a sliver above ``panel_top`` so the header
    # text does not overlap the first row's titles.
    _draw_multidim_header(
        fig,
        algo=first.algo,
        recipe_name=first.recipe_name,
        non_njobs_params=first.non_njobs_params,
        selection_mode=inputs.selection_mode,
        header_top=top_frac,
    )

    for row_idx, row in enumerate(inputs.rows):
        ax_gt = fig.add_subplot(gs[row_idx, 0])
        ax_ours = fig.add_subplot(gs[row_idx, 1])
        ax_theirs = fig.add_subplot(gs[row_idx, 2])
        ax_disagree = fig.add_subplot(gs[row_idx, 3])
        is_last_row = row_idx == n_rows - 1
        # Row titles prefix the dim so the reader can orient quickly; the
        # disagreement panel carries the per-dim n + n_jobs so the reader
        # sees the exact cell that was rendered.
        panel_titles = (
            f"d={row.dims}  Ground truth",
            "Ours",
            "Theirs",
            f"Disagreement  (n={row.size}, n_jobs={row.n_jobs})",
        )
        _render_vis_row(
            ax_gt=ax_gt,
            ax_ours=ax_ours,
            ax_theirs=ax_theirs,
            ax_disagree=ax_disagree,
            inputs=row,
            show_axis_labels=is_last_row,
            panel_titles=panel_titles,
        )

    per_dim_ari = tuple((row.dims, float(row.ari)) for row in inputs.rows)
    _draw_multidim_caption(
        fig,
        per_dim_ari=per_dim_ari,
        title_meta=first.title_meta,
        bench_filename=first.bench_filename,
        caption_bottom=0.35 * _MULTIDIM_CAPTION_IN / fig_h + 0.005,
    )
    return fig


def build_empty_vis_figure(algo: str, reason: str) -> Figure:
    """Fallback for degenerate cases (e.g. projection NaN)."""
    fig = Figure(figsize=(6.0, 4.0), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.set_title(f"{algo}: vis unavailable", color=_TEXT)
    ax.text(
        0.5,
        0.5,
        reason,
        fontsize=10.0,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=_MUTED,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _disagreement_mask(
    gt: np.ndarray,
    aligned_ours: np.ndarray,
    aligned_theirs: np.ndarray | None,
) -> np.ndarray:
    """Compute the disagreement mask over the full point set.

    A point is flagged as disagreeing when ours OR theirs differs from gt.
    In an ours-only run (theirs is None), the mask is just ours != gt.

    Disagreement is strict: a point at (gt=-1, ours=-1) is NOT a disagreement,
    but (gt=0, ours=-1) IS (ours called a core point noise). The comparison
    runs on aligned labels so label-permutation artefacts don't leak.
    """
    base = aligned_ours != gt
    if aligned_theirs is not None:
        base = base | (aligned_theirs != gt)
    return base


__all__ = [
    "MultiDimVisInputs",
    "VisInputs",
    "build_empty_vis_figure",
    "build_multidim_vis_figure",
    "build_vis_figure",
]
