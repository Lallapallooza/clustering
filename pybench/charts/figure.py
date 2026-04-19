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

from pybench.charts.data import safe_ratio
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
# green/red for the win/loss bands. Data colors come from a widened viridis
# slice for strong contrast between series (see _series_colors).
_TEXT = "#111419"
_MUTED = "#4E5866"
_AXIS = "#2A313A"
_GRID = "#E4E8ED"
_PARITY = "#6C7480"
_BAND_WIN = "#3EA06E"
_BAND_LOSS = "#D06963"

# Tiered tick lists ordered coarse -> fine. _pick_log_ticks walks the tiers and
# picks the first one that produces 3-7 ticks inside a facet's y-range. Every
# tier includes 1.0 so parity lines up on a gridline whenever it's in view. The
# low end extends below 0.1 so facets where one side dominates (e.g. memory
# ratios of 0.1x-0.5x) still get labeled gridlines instead of a blank axis.
_TICK_TIERS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    (
        0.03,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        0.7,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        7.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ),
    (0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    (0.7, 0.8, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0, 2.5, 3.0),
    (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5),
)


def _pick_log_ticks(
    lo: float, hi: float, *, target_min: int = 3, target_max: int = 7
) -> list[float]:
    """Pick a clean set of major ticks for a log-scale range ``[lo, hi]``.

    Walks the curated tiers coarse -> fine and returns the first that lands in
    the target count window. If no tier matches, returns the non-empty tier
    whose count is closest to the window so the axis never renders blank.
    """
    best: list[float] = []
    best_score = math.inf
    for tier in _TICK_TIERS:
        in_range = [t for t in tier if lo <= t <= hi]
        n = len(in_range)
        if target_min <= n <= target_max:
            return in_range
        if n == 0:
            continue
        score = target_min - n if n < target_min else n - target_max
        if score < best_score:
            best_score = score
            best = in_range
    if len(best) > target_max:
        step = (len(best) - 1) / (target_max - 1)
        return [best[round(i * step)] for i in range(target_max)]
    return best


def _padded_log_range(
    values: list[float], *, pad_fold: float = 1.10
) -> tuple[float, float]:
    """Tight log-scale bounds around ``values`` with a small multiplicative pad.

    Intentionally does NOT force parity (1.0) into view -- when every value is
    comfortably above 1x, including 1x would drag the scale down and compress
    the data. The green/red win/loss shading continues to mark the parity
    crossover wherever it falls inside the axis range.
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


def _get_n_jobs(r: RunResult) -> int:
    return int(r.params.get("n_jobs", -1))


def _format_non_njobs_params(pairs: tuple[tuple[str, Any], ...]) -> str:
    if not pairs:
        return ""
    return "  |  ".join(f"{k}={v}" for k, v in pairs)


def _collect_line(
    results: tuple[RunResult, ...],
    dim: int,
    n_jobs: int,
    metric: str,
) -> tuple[list[int], list[float]]:
    by_size: dict[int, float] = {}
    for r in results:
        if r.dims != dim or _get_n_jobs(r) != n_jobs:
            continue
        if metric == "time":
            ratio = safe_ratio(r.theirs_median_ms, r.ours_median_ms)
        else:
            ratio = safe_ratio(r.theirs_peak_mb, r.ours_peak_mb)
        by_size[r.size] = ratio
    sizes = sorted(by_size.keys())
    ratios = [by_size[s] for s in sizes]
    return sizes, ratios


def _all_sizes_for_dim(results: tuple[RunResult, ...], dim: int) -> list[int]:
    return sorted({r.size for r in results if r.dims == dim})


def _fill_gaps(
    all_sizes: list[int],
    present_sizes: list[int],
    ratios: list[float],
) -> tuple[list[int], list[float]]:
    # NaN entries render as broken segments, which is the desired visual for
    # a missing (size, n_jobs) cell.
    present = dict(zip(present_sizes, ratios, strict=True))
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


def _fmt_ratio_tick(y: float, _pos: int) -> str:
    if y <= 0 or not math.isfinite(y):
        return ""
    if y >= 1.0:
        if y == int(y):
            return f"{int(y)}x"
        return f"{y:g}x"
    return f"{y:g}x"


def _style_axes(ax: Any, *, sizes: list[int]) -> None:
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
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ratio_tick))
    ax.yaxis.set_minor_locator(NullLocator())

    # Pin x-major ticks to the actual dataset sizes so each data column lines up
    # with a labeled gridline.
    if sizes:
        ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_size_tick))
    ax.xaxis.set_minor_locator(NullLocator())

    # Every facet carries its own x- and y-tick labels. Rows no longer share a
    # y-scale (each (metric, dim) has its own natural range), and the speedup
    # row would otherwise sit above unlabeled x-ticks -- so we label every axis
    # instead of relying on the shared-axis convention.


def _plot_one_axes(
    ax: Any,
    *,
    results: tuple[RunResult, ...],
    dim: int,
    n_jobs_values: list[int],
    metric: str,
    colors: list[Any],
) -> None:
    # Win/loss bands first so they sit under everything else. A touch more
    # saturated than a typical paper preprint so the reader sees at a glance
    # which side of parity the curve lives on.
    ax.axhspan(1.0, 1e8, facecolor=_BAND_WIN, alpha=0.07, zorder=0)
    ax.axhspan(1e-8, 1.0, facecolor=_BAND_LOSS, alpha=0.07, zorder=0)
    # Parity line (dashed). A Line2D with linestyle="--" is what the tests
    # distinguish from the solid series lines.
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.1,
        color=_PARITY,
        alpha=0.95,
        zorder=1,
    )
    all_sizes = _all_sizes_for_dim(results, dim)
    for i, nj in enumerate(n_jobs_values):
        present_sizes, ratios = _collect_line(results, dim, nj, metric)
        xs, ys = _fill_gaps(all_sizes, present_sizes, ratios)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            linewidth=2.3,
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=1.1,
            color=colors[i],
            label=f"n_jobs={nj}",
            zorder=3,
        )


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
        "higher is better",
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
    handles: list[Line2D] = []
    for i, nj in enumerate(n_jobs_values):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                linewidth=2.3,
                markersize=7.0,
                markeredgecolor="white",
                markeredgewidth=1.1,
                color=colors[i],
                label=f"n_jobs = {nj}",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=1.1,
            color=_PARITY,
            label="parity (theirs = ours)",
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
        _style_axes(top, sizes=sizes_for_dim)
        _style_axes(bot, sizes=sizes_for_dim)

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
                "time speedup   (theirs / ours)",
                fontsize=11.5,
                fontweight="medium",
                color=_TEXT,
                labelpad=8,
            )
            bot.set_ylabel(
                "memory savings   (theirs / ours)",
                fontsize=11.5,
                fontweight="medium",
                color=_TEXT,
                labelpad=8,
            )

    # Pin axis limits AFTER drawing so the win/loss axhspans (extending to
    # ±1e8 to clip the plot area) can't drag the autoscaled range out to
    # something meaningless. Each facet gets a y-range tight to its own data
    # so the curves actually fill the plot area, and its tick list is picked
    # to match that range's width.
    for c, dim in enumerate(dims):
        facet = [r for r in inputs.partition_results if r.dims == dim]
        time_vals = [safe_ratio(r.theirs_median_ms, r.ours_median_ms) for r in facet]
        mem_vals = [safe_ratio(r.theirs_peak_mb, r.ours_peak_mb) for r in facet]
        t_lo, t_hi = _padded_log_range(time_vals)
        m_lo, m_hi = _padded_log_range(mem_vals)
        axes[0][c].set_ylim(t_lo, t_hi)
        axes[1][c].set_ylim(m_lo, m_hi)
        axes[0][c].yaxis.set_major_locator(FixedLocator(_pick_log_ticks(t_lo, t_hi)))
        axes[1][c].yaxis.set_major_locator(FixedLocator(_pick_log_ticks(m_lo, m_hi)))
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
