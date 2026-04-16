from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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
    return "+".join(f"{k}={v}" for k, v in pairs)


def _meta_line(meta: RunMetadata, dataset_spec: str) -> str:
    return (
        f"{dataset_spec}  |  {meta.timestamp_iso}  |  "
        f"git={meta.git_sha}  |  {meta.machine}"
    )


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


def _plot_one_axes(
    ax: Any,
    *,
    results: tuple[RunResult, ...],
    dim: int,
    n_jobs_values: list[int],
    metric: str,
    cmap: Any,
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.5)
    all_sizes = _all_sizes_for_dim(results, dim)
    n = max(1, len(n_jobs_values))
    for i, nj in enumerate(n_jobs_values):
        color = cmap((i + 0.5) / n)
        present_sizes, ratios = _collect_line(results, dim, nj, metric)
        xs, ys = _fill_gaps(all_sizes, present_sizes, ratios)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            color=color,
            label=f"n_jobs={nj}",
        )
        last_finite_idx = None
        for j in range(len(ys) - 1, -1, -1):
            if math.isfinite(ys[j]):
                last_finite_idx = j
                break
        last_label = format_ratio_label(
            ys[last_finite_idx] if last_finite_idx is not None else math.nan
        )
        anchor_x = xs[last_finite_idx] if last_finite_idx is not None else xs[-1]
        anchor_y = ys[last_finite_idx] if last_finite_idx is not None else 1.0
        ax.annotate(
            last_label,
            xy=(anchor_x, anchor_y),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )


def build_figure(inputs: BuildFigureInputs) -> Figure:
    dims = tuple(inputs.dims_sorted)
    d = len(dims)
    if d == 0:
        return build_empty_figure(inputs.algo, "no dims to render")

    width = max(6.0, 3.5 * d)
    fig = Figure(figsize=(width, 6.0))
    # Attaching the Agg canvas here makes fig.savefig work without pyplot.
    FigureCanvasAgg(fig)

    axes_grid = fig.subplots(2, d, sharex=True, squeeze=False)

    n_jobs_values = sorted({_get_n_jobs(r) for r in inputs.partition_results})
    cmap = colormaps["viridis"]

    for col, dim in enumerate(dims):
        top = axes_grid[0, col]
        bot = axes_grid[1, col]
        top.set_title(f"dim={dim}")
        _plot_one_axes(
            top,
            results=inputs.partition_results,
            dim=dim,
            n_jobs_values=n_jobs_values,
            metric="time",
            cmap=cmap,
        )
        _plot_one_axes(
            bot,
            results=inputs.partition_results,
            dim=dim,
            n_jobs_values=n_jobs_values,
            metric="mem",
            cmap=cmap,
        )
        if col == 0:
            top.set_ylabel("time (theirs / ours)")
            bot.set_ylabel("memory (theirs / ours)")
        bot.set_xlabel("dataset size")

    param_summary = _format_non_njobs_params(inputs.non_njobs_params)
    suptitle = inputs.algo if not param_summary else f"{inputs.algo}   {param_summary}"
    fig.suptitle(suptitle)
    fig.text(
        0.5,
        0.92,
        _meta_line(inputs.title_meta, inputs.dataset_spec),
        fontsize=8,
        ha="center",
    )
    if inputs.ari_threshold is not None:
        fig.text(
            0.99,
            0.01,
            f"ARI>={inputs.ari_threshold:g}",
            fontsize=7,
            ha="right",
            va="bottom",
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return fig


def build_empty_figure(algo: str, reason: str) -> Figure:
    fig = Figure(figsize=(6.0, 4.0))
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.set_title(f"{algo}: no data")
    ax.text(
        0.5,
        0.5,
        reason,
        fontsize=10,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


__all__ = [
    "BuildFigureInputs",
    "build_empty_figure",
    "build_figure",
    "format_ratio_label",
]
