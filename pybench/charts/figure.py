from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from pybench.charts.meta import RunMetadata
from pybench.recipe import RunResult


@dataclass(frozen=True, slots=True)
class BuildFigureInputs:
    algo: str
    partition_results: tuple[RunResult, ...]
    dims_sorted: tuple[int, ...]
    title_meta: RunMetadata
    non_njobs_params: tuple[tuple[str, Any], ...]
    dataset_spec: str
    ari_threshold: float | None


# Paper-grade palette: near-black for headings, warm grey chrome, desaturated
# green for the win-region shading. Data colors come from a widened viridis
# slice for strong contrast between series (see _series_colors).
_TEXT = "#111419"
_MUTED = "#4E5866"
_AXIS = "#2A313A"
_GRID = "#E4E8ED"
_BAND_WIN = "#3EA06E"

# Decade anchors used to pick absolute log-scale ticks. _pick_abs_ticks walks
# the decade lattice and keeps ticks that land inside the facet's y-range. We
# thin the list when it would produce more than ~7 labels so the axis reads
# cleanly at paper width.
_TIME_TICK_LATTICE: tuple[float, ...] = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
)
_MEM_TICK_LATTICE: tuple[float, ...] = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
)


def _pick_abs_ticks(
    lattice: tuple[float, ...],
    lo: float,
    hi: float,
    *,
    target_max: int = 7,
) -> list[float]:
    """Pick a clean set of absolute log-scale ticks inside ``[lo, hi]``.

    Walks the decade lattice and keeps every anchor that falls within the
    padded range. When too many anchors match we thin uniformly so the axis
    never carries more than ~7 labels.
    """
    in_range = [t for t in lattice if lo <= t <= hi]
    if not in_range:
        # Fall back to the two anchors bracketing the range so the axis always
        # labels something.
        below = [t for t in lattice if t < lo]
        above = [t for t in lattice if t > hi]
        bracket: list[float] = []
        if below:
            bracket.append(below[-1])
        if above:
            bracket.append(above[0])
        return bracket
    if len(in_range) <= target_max:
        return in_range
    step = (len(in_range) - 1) / (target_max - 1)
    return [in_range[round(i * step)] for i in range(target_max)]


def _padded_log_range(
    values: list[float], *, pad_fold: float = 1.10
) -> tuple[float, float]:
    """Tight log-scale bounds around ``values`` with a small multiplicative pad.

    Only considers finite, strictly positive values. Returns a sensible default
    ``[1/pad_fold, pad_fold]`` when no valid sample exists, so a fully-missing
    facet still renders a labeled axis instead of an empty canvas.
    """
    finite = [v for v in values if math.isfinite(v) and v > 0]
    if not finite:
        return (1.0 / pad_fold, 1.0 * pad_fold)
    return (min(finite) / pad_fold, max(finite) * pad_fold)


def _round_half_up(r: float, digits: int) -> str:
    # Route through Decimal(str(...)) so half-way values like 9.95 round up
    # instead of hitting Python's default banker's rounding.
    quant = Decimal(1).scaleb(-digits)
    return str(Decimal(str(r)).quantize(quant, rounding=ROUND_HALF_UP))


def format_ratio_label(r: float) -> str:
    if not math.isfinite(r):
        return "n/a"
    if r < 1.0:
        return f"{_round_half_up(r, 2)}x"
    if r < 10.0:
        return f"{_round_half_up(r, 1)}x"
    return f"{_round_half_up(r, 0)}x"


def _fmt_time_tick(y: float, _pos: int) -> str:
    if y <= 0 or not math.isfinite(y):
        return ""
    if y < 1.0:
        return f"{y * 1000:g} us" if y * 1000 < 1000 else f"{y:g} ms"
    if y < 1000.0:
        v = y
        return f"{int(v)} ms" if v == int(v) else f"{v:g} ms"
    if y < 60_000.0:
        v = y / 1000.0
        return f"{int(v)} s" if v == int(v) else f"{v:g} s"
    v = y / 60_000.0
    return f"{int(v)} min" if v == int(v) else f"{v:g} min"


def _fmt_mem_tick(y: float, _pos: int) -> str:
    if y <= 0 or not math.isfinite(y):
        return ""
    if y < 1.0:
        v = y * 1024.0
        return f"{int(v)} KB" if v == int(v) else f"{v:g} KB"
    if y < 1024.0:
        return f"{int(y)} MB" if y == int(y) else f"{y:g} MB"
    v = y / 1024.0
    return f"{int(v)} GB" if v == int(v) else f"{v:g} GB"


def _get_n_jobs(r: RunResult) -> int:
    return int(r.params.get("n_jobs", -1))


def _format_non_njobs_params(pairs: tuple[tuple[str, Any], ...]) -> str:
    if not pairs:
        return ""
    return "  |  ".join(f"{k}={v}" for k, v in pairs)


def _metric_fields(metric: str) -> tuple[str, str]:
    if metric == "time":
        return ("ours_median_ms", "theirs_median_ms")
    return ("ours_peak_mb", "theirs_peak_mb")


def _collect_abs_series(
    results: tuple[RunResult, ...],
    dim: int,
    n_jobs: int,
    metric: str,
) -> tuple[list[int], list[float], list[float]]:
    """Return aligned ``(sizes, ours_values, theirs_values)`` for the cell.

    Values that are not positive and finite are replaced with NaN so matplotlib
    renders broken segments for missing data without the caller thinking about
    it. The size order is the ascending union of sizes present in the cell.
    """
    ours_field, theirs_field = _metric_fields(metric)
    per_size: dict[int, tuple[float, float]] = {}
    for r in results:
        if r.dims != dim or _get_n_jobs(r) != n_jobs:
            continue
        ours = float(getattr(r, ours_field))
        theirs = float(getattr(r, theirs_field))
        per_size[r.size] = (ours, theirs)
    sizes = sorted(per_size.keys())
    ours_ys = [_to_positive_or_nan(per_size[s][0]) for s in sizes]
    theirs_ys = [_to_positive_or_nan(per_size[s][1]) for s in sizes]
    return sizes, ours_ys, theirs_ys


def _to_positive_or_nan(v: float) -> float:
    if math.isfinite(v) and v > 0:
        return v
    return math.nan


def _all_sizes_for_dim(results: tuple[RunResult, ...], dim: int) -> list[int]:
    return sorted({r.size for r in results if r.dims == dim})


def _align_to(
    all_sizes: list[int],
    present_sizes: list[int],
    values: list[float],
) -> tuple[list[int], list[float]]:
    """Pad ``values`` with NaN so each column in ``all_sizes`` is covered."""
    present = dict(zip(present_sizes, values, strict=True))
    aligned = [present.get(s, math.nan) for s in all_sizes]
    return list(all_sizes), aligned


def _fmt_size_tick(x: float, _pos: int) -> str:
    if x >= 1e6:
        v = x / 1e6
        return f"{v:g}M"
    if x >= 1e3:
        v = x / 1e3
        return f"{v:g}k"
    return f"{x:g}"


def _style_axes(ax: Any, *, sizes: list[int], metric: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.7)

    ax.set_facecolor("white")
    ax.grid(
        which="major",
        axis="both",
        color=_GRID,
        linestyle="-",
        linewidth=0.6,
        alpha=1.0,
        zorder=0,
    )
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        which="major",
        color=_AXIS,
        labelcolor=_TEXT,
        labelsize=10.0,
        length=3.5,
        width=0.8,
        pad=3,
    )
    ax.tick_params(axis="both", which="minor", length=0)

    # Y-major ticks are pinned later (per-facet) once we know the data range.
    if metric == "time":
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_time_tick))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_mem_tick))
    ax.yaxis.set_minor_locator(NullLocator())

    # Pin x-major ticks to the actual dataset sizes so each data column lines up
    # with a labeled gridline.
    if sizes:
        ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_size_tick))
    ax.xaxis.set_minor_locator(NullLocator())


def _plot_one_axes(
    ax: Any,
    *,
    results: tuple[RunResult, ...],
    dim: int,
    n_jobs_values: list[int],
    metric: str,
    colors: list[Any],
) -> None:
    """Plot ours/theirs absolute curves for every n_jobs in ``n_jobs_values``.

    One solid Line2D (ours) plus one dashed Line2D (theirs) per n_jobs, sharing
    the viridis-slice color. ``fill_between`` shades the win region (ours below
    theirs) at low alpha. Each n_jobs contributes a ratio annotation at the
    rightmost finite x, vertically fanned so overlapping labels stay legible.
    """
    all_sizes = _all_sizes_for_dim(results, dim)
    j = len(n_jobs_values)
    for i, nj in enumerate(n_jobs_values):
        present_sizes, ours_present, theirs_present = _collect_abs_series(
            results, dim, nj, metric
        )
        xs, ours_ys = _align_to(all_sizes, present_sizes, ours_present)
        _, theirs_ys = _align_to(all_sizes, present_sizes, theirs_present)
        if not xs:
            continue

        color = colors[i]
        ax.plot(
            xs,
            ours_ys,
            marker="o",
            linestyle="-",
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            color=color,
            label=f"ours n_jobs={nj}",
            zorder=3,
        )
        ax.plot(
            xs,
            theirs_ys,
            marker="o",
            linestyle="--",
            linewidth=1.8,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            color=color,
            label=f"theirs n_jobs={nj}",
            zorder=2,
        )

        # Shade the win region (ours below theirs). Missing-data NaNs drop out
        # of the mask cleanly so partial curves don't inherit a stray polygon.
        where_mask = [
            math.isfinite(o) and math.isfinite(t) and o < t
            for o, t in zip(ours_ys, theirs_ys, strict=True)
        ]
        ax.fill_between(
            xs,
            ours_ys,
            theirs_ys,
            where=where_mask,
            interpolate=False,
            facecolor=_BAND_WIN,
            edgecolor="none",
            alpha=0.08,
            zorder=1,
        )

        # Endpoint ratio label: take the last position where both sides are
        # finite, fan vertically so labels across n_jobs don't overlap.
        rightmost_idx = _rightmost_matched_index(ours_ys, theirs_ys)
        if rightmost_idx is None:
            continue
        x_pt = xs[rightmost_idx]
        o = ours_ys[rightmost_idx]
        t = theirs_ys[rightmost_idx]
        ratio = t / o if math.isfinite(o) and o > 0 and math.isfinite(t) else math.nan
        label = format_ratio_label(ratio)
        fontsize = 9.5
        y_offset_pt = fontsize * (i - (j - 1) / 2.0)
        ax.annotate(
            label,
            xy=(x_pt, o),
            xytext=(6, y_offset_pt),
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            ha="left",
            va="center",
            zorder=4,
        )


def _rightmost_matched_index(
    ours_ys: list[float], theirs_ys: list[float]
) -> int | None:
    for idx in range(len(ours_ys) - 1, -1, -1):
        o = ours_ys[idx]
        t = theirs_ys[idx]
        if math.isfinite(o) and o > 0 and math.isfinite(t) and t > 0:
            return idx
    return None


def _series_colors(cmap: Any, n: int) -> list[Any]:
    # Sample viridis across a wider slice than the default so adjacent series
    # read as visually distinct on screen and in print. Skipping the extreme
    # dark (unreadable) and extreme yellow (washes out on white) ends keeps the
    # palette print-safe while preserving monotonic luminance (pinned by test).
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(0.08 + 0.84 * i / (n - 1)) for i in range(n)]


def _draw_title_block(
    fig: Figure,
    *,
    algo: str,
    param_summary: str,
    dataset_spec: str,
) -> None:
    fig.text(
        0.06,
        0.955,
        f"{algo}  vs  scikit-learn",
        fontsize=19.0,
        fontweight="bold",
        color=_TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.99,
        0.958,
        "lower is better",
        fontsize=10.5,
        color=_MUTED,
        ha="right",
        va="top",
    )
    subtitle_parts = [s for s in (param_summary, dataset_spec) if s]
    if subtitle_parts:
        fig.text(
            0.06,
            0.915,
            "    |    ".join(subtitle_parts),
            fontsize=11.5,
            color=_MUTED,
            ha="left",
            va="top",
        )


def _draw_header_rule(fig: Figure) -> None:
    from matplotlib.lines import Line2D as _L2D

    rule = _L2D(
        [0.06, 0.99],
        [0.87, 0.87],
        transform=fig.transFigure,
        color=_AXIS,
        linewidth=0.9,
        alpha=0.25,
        solid_capstyle="butt",
    )
    fig.add_artist(rule)


def _draw_legend(fig: Figure, n_jobs_values: list[int], colors: list[Any]) -> None:
    # Two style entries ("ours" solid, "theirs" dashed) rendered in a neutral
    # grey because color is the n_jobs axis. Then one swatch per n_jobs in the
    # viridis slice. No parity entry -- absolute chart has no parity line.
    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            color=_AXIS,
            label="ours",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="--",
            linewidth=1.8,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            color=_AXIS,
            label="theirs",
        ),
    ]
    for i, nj in enumerate(n_jobs_values):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=9.0,
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=colors[i],
                label=f"n_jobs = {nj}",
            )
        )
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.048),
        ncol=len(handles),
        frameon=False,
        fontsize=11.0,
        handlelength=2.6,
        handletextpad=0.7,
        columnspacing=2.6,
    )
    for txt in legend.get_texts():
        txt.set_color(_TEXT)


def _draw_caption(fig: Figure, meta: RunMetadata) -> None:
    caption = f"git  {meta.git_sha}    |    {meta.machine}    |    {meta.timestamp_iso}"
    fig.text(
        0.5,
        0.014,
        caption,
        fontsize=9.0,
        color=_MUTED,
        ha="center",
        va="bottom",
    )


def build_figure(inputs: BuildFigureInputs) -> Figure:
    dims = tuple(inputs.dims_sorted)
    d = len(dims)
    if d == 0:
        return build_empty_figure(inputs.algo, "no dims to render")

    width = max(10.0, 3.4 * d + 1.6)
    height = 8.1
    fig = Figure(figsize=(width, height), facecolor="white")
    FigureCanvasAgg(fig)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=d,
        left=0.055,
        right=0.99,
        top=0.81,
        bottom=0.145,
        hspace=0.38,
        wspace=0.28,
    )

    # Share x per column (same dataset sizes across the two metric rows), but
    # not y per row -- each (metric, dim) facet picks its own y-range so the
    # curves in low-variance facets don't collapse to a flat line.
    axes = [[fig.add_subplot(gs[r, c]) for c in range(d)] for r in range(2)]
    for c in range(d):
        axes[1][c].sharex(axes[0][c])

    n_jobs_values = sorted({_get_n_jobs(r) for r in inputs.partition_results})
    colors = _series_colors(colormaps["viridis"], len(n_jobs_values))

    for col, dim in enumerate(dims):
        top = axes[0][col]
        bot = axes[1][col]
        sizes_for_dim = _all_sizes_for_dim(inputs.partition_results, dim)
        _style_axes(top, sizes=sizes_for_dim, metric="time")
        _style_axes(bot, sizes=sizes_for_dim, metric="mem")

        top.set_title(
            f"d = {dim}",
            fontsize=12.0,
            fontweight="bold",
            color=_TEXT,
            pad=6,
            loc="left",
        )

        _plot_one_axes(
            top,
            results=inputs.partition_results,
            dim=dim,
            n_jobs_values=n_jobs_values,
            metric="time",
            colors=colors,
        )
        _plot_one_axes(
            bot,
            results=inputs.partition_results,
            dim=dim,
            n_jobs_values=n_jobs_values,
            metric="mem",
            colors=colors,
        )

        if col == 0:
            top.set_ylabel(
                "time (ms, log)",
                fontsize=11.5,
                fontweight="medium",
                color=_TEXT,
                labelpad=8,
            )
            bot.set_ylabel(
                "peak memory (MB, log)",
                fontsize=11.5,
                fontweight="medium",
                color=_TEXT,
                labelpad=8,
            )

    # Pin axis limits AFTER drawing so autoscaled ranges don't run away when a
    # facet has only one finite point. Each facet gets a tight y-range around
    # the union of ours and theirs in its own (metric, dim) cell.
    for c, dim in enumerate(dims):
        facet = [r for r in inputs.partition_results if r.dims == dim]
        time_vals: list[float] = []
        mem_vals: list[float] = []
        for r in facet:
            time_vals.append(float(r.ours_median_ms))
            time_vals.append(float(r.theirs_median_ms))
            mem_vals.append(float(r.ours_peak_mb))
            mem_vals.append(float(r.theirs_peak_mb))
        t_lo, t_hi = _padded_log_range(time_vals)
        m_lo, m_hi = _padded_log_range(mem_vals)
        axes[0][c].set_ylim(t_lo, t_hi)
        axes[1][c].set_ylim(m_lo, m_hi)
        axes[0][c].yaxis.set_major_locator(
            FixedLocator(_pick_abs_ticks(_TIME_TICK_LATTICE, t_lo, t_hi))
        )
        axes[1][c].yaxis.set_major_locator(
            FixedLocator(_pick_abs_ticks(_MEM_TICK_LATTICE, m_lo, m_hi))
        )
        sizes_for_dim = _all_sizes_for_dim(inputs.partition_results, dim)
        if sizes_for_dim:
            lo = sizes_for_dim[0] / 1.15
            hi = sizes_for_dim[-1] * 1.15
            axes[0][c].set_xlim(lo, hi)

    # Single shared x-axis caption in the gutter between the plots and the
    # legend row; clearer than repeating "dataset size" on every axes.
    fig.text(
        (0.06 + 0.99) / 2.0,
        0.098,
        "dataset size  (n)",
        fontsize=11.5,
        fontweight="medium",
        color=_TEXT,
        ha="center",
        va="bottom",
    )

    _draw_title_block(
        fig,
        algo=inputs.algo,
        param_summary=_format_non_njobs_params(inputs.non_njobs_params),
        dataset_spec=inputs.dataset_spec,
    )
    # Hairline rule between the header block and the plot grid -- reads as a
    # paper "abstract / body" separator without being loud.
    _draw_header_rule(fig)
    _draw_legend(fig, n_jobs_values, colors)
    _draw_caption(fig, inputs.title_meta)

    if inputs.ari_threshold is not None:
        fig.text(
            0.99,
            0.014,
            f"ARI >= {inputs.ari_threshold:g}",
            fontsize=9.0,
            color=_MUTED,
            ha="right",
            va="bottom",
        )

    return fig


def build_empty_figure(algo: str, reason: str) -> Figure:
    fig = Figure(figsize=(6.0, 4.0), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.set_title(f"{algo}: no data", color=_TEXT)
    ax.text(
        0.5,
        0.5,
        reason,
        fontsize=10,
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


__all__ = [
    "BuildFigureInputs",
    "build_empty_figure",
    "build_figure",
    "format_ratio_label",
]
