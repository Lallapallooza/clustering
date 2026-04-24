from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from pybench.charts.data import payload_hash6, slug_from_params
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


# Paper-grade palette: near-black for headings, warm grey chrome. Data colors
# come from the Okabe-Ito colourblind-safe set (see _series_colors). Win and
# loss fills use Paul Tol's muted pair (#117733 / #CC6677) so they read as
# distinct hue-against-hue even under deuteranopia or protanopia.
_TEXT = "#111419"
_MUTED = "#4E5866"
_AXIS = "#2A313A"
_GRID = "#E4E8ED"
_BAND_WIN = "#117733"
_BAND_LOSS = "#CC6677"

# Okabe-Ito palette, sorted by rec-601 luminance (dark to light). Callers that
# want monotonic dark-to-light adjacency (for example per-n_jobs series where
# luminance order cues "this series is darker = lower parallelism") sample a
# leading prefix of this list. Drops "black" to keep contrast with axis chrome.
_OKABE_ITO_BY_LUMA: tuple[str, ...] = (
    "#0072B2",  # blue          (lum 0.342)
    "#009E73",  # bluish green  (lum 0.415)
    "#D55E00",  # vermillion    (lum 0.466)
    "#CC79A7",  # reddish purple (lum 0.592)
    "#56B4E9",  # sky blue      (lum 0.619)
    "#E69F00",  # orange        (lum 0.636)
    "#F0E442",  # yellow        (lum 0.836)
)

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
    1_024.0,
    10_240.0,
    102_400.0,
    1_024_000.0,
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
    """Render the non-n_jobs params as a separator-joined summary.

    ``n_clusters`` is relabelled to ``k`` in the header so the reader sees the
    k-means / HDBSCAN convention directly; the stored key stays
    ``n_clusters`` for compatibility with the JSON schema.
    """
    if not pairs:
        return ""

    def _fmt_pair(k: str, v: Any) -> str:
        if k == "n_clusters":
            return f"k={v}"
        return f"{k}={v}"

    return "    |    ".join(_fmt_pair(k, v) for k, v in pairs)


def _metric_fields(metric: str) -> tuple[str, str]:
    if metric == "time":
        return ("ours_median_ms", "theirs_median_ms")
    return ("ours_peak_mb", "theirs_peak_mb")


def is_ours_only_sentinel(result: RunResult) -> bool:
    """Return ``True`` if ``result`` is the ``--ours-only`` sentinel tuple.

    Public predicate consumed by the CLI (to keep vis regen on the
    ``--ours-only`` branch) and by this module's plotting (to suppress the
    theirs curve, fill band, and endpoint ratio label at sentinel rows).

    The runner stores ``theirs_median_ms=0.0, theirs_peak_mb=0.0, ari=1.0,
    speedup=0.0`` together at a single construct site when ``ours_only`` is
    true. The check is strict exact equality on all four fields -- a partial
    match (three of four zeros) is a real row, not a sentinel.
    """
    return (
        result.theirs_median_ms == 0.0
        and result.theirs_peak_mb == 0.0
        and result.ari == 1.0
        and result.speedup == 0.0
    )


def _collect_abs_series(
    results: tuple[RunResult, ...],
    dim: int,
    n_jobs: int,
    metric: str,
) -> tuple[list[int], list[float], list[float], list[bool]]:
    """Return aligned ``(sizes, ours_values, theirs_values, sentinel_mask)``.

    Values that are not positive and finite are replaced with NaN so matplotlib
    renders broken segments for missing data without the caller thinking about
    it. ``sentinel_mask[i]`` is ``True`` when the row at ``sizes[i]`` is an
    ``--ours-only`` sentinel; callers use this to suppress the theirs dashed
    curve, fill band, and endpoint ratio label at those positions. The size
    order is the ascending union of sizes present in the cell.
    """
    ours_field, theirs_field = _metric_fields(metric)
    per_size: dict[int, tuple[float, float, bool]] = {}
    for r in results:
        if r.dims != dim or _get_n_jobs(r) != n_jobs:
            continue
        ours = float(getattr(r, ours_field))
        theirs = float(getattr(r, theirs_field))
        per_size[r.size] = (ours, theirs, is_ours_only_sentinel(r))
    sizes = sorted(per_size.keys())
    ours_ys = [_to_positive_or_nan(per_size[s][0]) for s in sizes]
    # Suppress the theirs value at sentinel rows by writing NaN regardless of
    # the stored 0.0 -- _to_positive_or_nan would already map 0.0 -> NaN, but
    # the explicit path keeps the coupling to the sentinel predicate obvious.
    theirs_ys = [
        math.nan if per_size[s][2] else _to_positive_or_nan(per_size[s][1])
        for s in sizes
    ]
    sentinel_mask = [per_size[s][2] for s in sizes]
    return sizes, ours_ys, theirs_ys, sentinel_mask


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
    the per-series colourblind-safe colour. A single facet-level win fill
    (green) shades the x-range where ours wins across every n_jobs; a
    facet-level loss fill (red) shades the x-range where theirs wins across
    every n_jobs. Per-n_jobs fills are intentionally dropped -- they stacked
    into unreadable bands at J=3. Each n_jobs contributes a ratio annotation
    at the rightmost finite x, vertically fanned so overlapping labels stay
    legible; near-noise-floor ratios render grey/italic so the reader treats
    them as qualitative.

    ``--ours-only`` sentinel rows suppress the dashed theirs curve and
    endpoint ratio label at their x-positions. When every row for an n_jobs
    is a sentinel, the dashed curve and label are omitted. When every row
    across every n_jobs is a sentinel, the facet carries a single
    "theirs skipped (ours-only run)" annotation instead.
    """
    all_sizes = _all_sizes_for_dim(results, dim)
    j = len(n_jobs_values)

    # Compute per-n_jobs presence + sentinel stats once so we can tell "facet
    # has at least one real theirs curve" from "every n_jobs is fully ours-only".
    per_nj: list[tuple[list[int], list[float], list[float], list[bool]]] = []
    facet_has_any_real = False
    for nj in n_jobs_values:
        present_sizes, ours_present, theirs_present, sentinel_present = (
            _collect_abs_series(results, dim, nj, metric)
        )
        per_nj.append((present_sizes, ours_present, theirs_present, sentinel_present))
        if present_sizes and not all(sentinel_present):
            facet_has_any_real = True

    # Per-n_jobs aligned series cache, used both for line plotting and for the
    # facet-level envelope fill / near-parity detection.
    aligned: list[tuple[list[int], list[float], list[float], list[bool]]] = []

    for i, nj in enumerate(n_jobs_values):
        present_sizes, ours_present, theirs_present, sentinel_present = per_nj[i]
        if not present_sizes:
            aligned.append(([], [], [], []))
            continue
        xs, ours_ys = _align_to(all_sizes, present_sizes, ours_present)
        _, theirs_ys = _align_to(all_sizes, present_sizes, theirs_present)
        sentinel_by_size = dict(zip(present_sizes, sentinel_present, strict=True))
        sentinel_ys = [sentinel_by_size.get(s, False) for s in xs]
        aligned.append((xs, ours_ys, theirs_ys, sentinel_ys))
        all_sentinel_for_nj = all(sentinel_ys) and len(sentinel_ys) > 0

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
        if not all_sentinel_for_nj:
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

    # Facet-level win/loss fills. For each x in all_sizes, compute the
    # envelope of ours and theirs across every n_jobs (ignoring sentinel rows
    # and NaN). A win is "max(ours) < min(theirs)"; a loss is the inverse.
    # Missing bins drop out of the mask cleanly so partial facets still fill.
    if all_sizes and facet_has_any_real:
        win_mask: list[bool] = []
        loss_mask: list[bool] = []
        ours_env_max: list[float] = []
        ours_env_min: list[float] = []
        theirs_env_max: list[float] = []
        theirs_env_min: list[float] = []
        for col in range(len(all_sizes)):
            ours_col: list[float] = []
            theirs_col: list[float] = []
            for _xs, o_ys, t_ys, s_ys in aligned:
                if not o_ys:
                    continue
                if col < len(o_ys) and math.isfinite(o_ys[col]):
                    ours_col.append(o_ys[col])
                if col < len(t_ys) and not s_ys[col] and math.isfinite(t_ys[col]):
                    theirs_col.append(t_ys[col])
            if ours_col and theirs_col:
                o_max = max(ours_col)
                o_min = min(ours_col)
                t_max = max(theirs_col)
                t_min = min(theirs_col)
                ours_env_max.append(o_max)
                ours_env_min.append(o_min)
                theirs_env_max.append(t_max)
                theirs_env_min.append(t_min)
                win_mask.append(o_max < t_min)
                loss_mask.append(o_min > t_max)
            else:
                ours_env_max.append(math.nan)
                ours_env_min.append(math.nan)
                theirs_env_max.append(math.nan)
                theirs_env_min.append(math.nan)
                win_mask.append(False)
                loss_mask.append(False)
        if any(win_mask):
            ax.fill_between(
                all_sizes,
                ours_env_max,
                theirs_env_min,
                where=win_mask,
                interpolate=False,
                facecolor=_BAND_WIN,
                edgecolor="none",
                alpha=0.08,
                zorder=1,
            )
        if any(loss_mask):
            ax.fill_between(
                all_sizes,
                theirs_env_max,
                ours_env_min,
                where=loss_mask,
                interpolate=False,
                facecolor=_BAND_LOSS,
                edgecolor="none",
                alpha=0.08,
                zorder=1,
            )

    # Endpoint ratio labels, fanned vertically so adjacent n_jobs labels stay
    # legible at default figure width. The fan factor was widened from 1.0 to
    # 1.8 per fontsize because three n_jobs lines at the same rightmost x
    # collided at the old spacing. Near-noise-floor ratios render italic and
    # muted so the eye treats them as qualitative.
    near_parity_seen = False
    for i, nj in enumerate(n_jobs_values):
        xs, ours_ys, theirs_ys, sentinel_ys = aligned[i]
        if not xs:
            continue
        all_sentinel_for_nj = all(sentinel_ys) and len(sentinel_ys) > 0
        if all_sentinel_for_nj:
            continue
        # Track near-parity (any finite column within 1% of theirs) for a
        # single facet-level annotation below.
        for o, t in zip(ours_ys, theirs_ys, strict=True):
            if math.isfinite(o) and o > 0 and math.isfinite(t) and t > 0:
                if abs(t / o - 1.0) < 0.01:
                    near_parity_seen = True
                    break

        rightmost_idx = _rightmost_matched_index(ours_ys, theirs_ys)
        last_present_is_sentinel = bool(sentinel_ys) and sentinel_ys[-1]
        if (
            rightmost_idx is None
            or sentinel_ys[rightmost_idx]
            or last_present_is_sentinel
        ):
            continue
        x_pt = xs[rightmost_idx]
        o = ours_ys[rightmost_idx]
        t = theirs_ys[rightmost_idx]
        ratio = t / o if math.isfinite(o) and o > 0 and math.isfinite(t) else math.nan
        label = format_ratio_label(ratio)
        fontsize = 9.5
        # Fan factor 1.8 keeps the three-n_jobs case clear at paper width; the
        # older factor of 1.0 stacked labels on the rightmost x.
        y_offset_pt = 1.8 * fontsize * (i - (j - 1) / 2.0)
        is_near_noise = (
            metric == "time"
            and math.isfinite(ratio)
            and abs(ratio - 1.0) < 0.2
            and (o < 1.0 or t < 1.0)
        )
        color = colors[i]
        # Place labels just inside the plot area (negative x offset) so three
        # fanned labels at a wide figure width don't clip the right frame.
        ax.annotate(
            label,
            xy=(x_pt, o),
            xytext=(-6, y_offset_pt),
            textcoords="offset points",
            fontsize=fontsize,
            color=_MUTED if is_near_noise else color,
            fontstyle="italic" if is_near_noise else "normal",
            ha="right",
            va="center",
            zorder=4,
        )

    # Single facet-level "ours ~= theirs" note when any n_jobs curve is within
    # 1% of theirs. One annotation per facet, not per n_jobs.
    if near_parity_seen:
        ax.text(
            0.02,
            0.98,
            "ours ~= theirs",
            transform=ax.transAxes,
            fontsize=9.0,
            color=_MUTED,
            ha="left",
            va="top",
            zorder=5,
        )

    # Fully-ours-only facet: no n_jobs carried any real theirs data. Drop a
    # single text annotation in the facet center where the dashed curves would
    # otherwise live.
    if not facet_has_any_real and any(sizes for sizes, *_ in per_nj):
        ax.text(
            0.5,
            0.5,
            "theirs skipped (ours-only run)",
            transform=ax.transAxes,
            fontsize=10.5,
            color=_MUTED,
            ha="center",
            va="center",
            zorder=5,
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


def _series_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Pick ``n`` colourblind-safe series colors in increasing luminance.

    Samples the Okabe-Ito palette in luminance-sorted order so that adjacent
    series stay visually distinct under deuteranopia and protanopia, and the
    "darker = lower n_jobs" visual cue from the older viridis slice is
    preserved (the test suite pins monotonic luminance across the first three
    series).
    """
    palette = _OKABE_ITO_BY_LUMA
    if n <= 0:
        return []
    if n == 1:
        return [to_rgba(palette[0])]
    if n <= len(palette):
        return [to_rgba(palette[i]) for i in range(n)]
    # More series than palette entries: cycle with a luminance perturbation so
    # the extras still read as distinct. In practice we never see more than 6
    # n_jobs values against a single recipe, so this branch is defensive.
    picked: list[tuple[float, float, float, float]] = []
    for i in range(n):
        picked.append(to_rgba(palette[i % len(palette)]))
    return picked


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
        fontsize=11.5,
        fontweight="medium",
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
        fontsize=10.0,
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.8,
    )
    for txt in legend.get_texts():
        txt.set_color(_TEXT)


def _vis_filename_for_inputs(inputs: BuildFigureInputs) -> str | None:
    """Compute the vis PNG filename that corresponds to this bench chart.

    Mirrors the CLI's ``chart_filename(..., suffix="vis")`` scheme so the
    caption cross-reference is always accurate without coupling the chart
    module to the CLI. Returns ``None`` when the partition is empty.
    """
    if not inputs.partition_results:
        return None
    hash6 = payload_hash6(inputs.partition_results)
    slug = slug_from_params(dict(inputs.non_njobs_params))
    if slug:
        return f"{inputs.algo}_{slug}_vis_{hash6}.png"
    return f"{inputs.algo}_vis_{hash6}.png"


def _draw_caption(
    fig: Figure,
    meta: RunMetadata,
    *,
    vis_filename: str | None = None,
) -> None:
    """Draw the figure-level caption bar.

    ``vis_filename`` is an optional cross-reference to the companion vis PNG
    rendered from the same partition; the ``pybench vis --results ...`` hint
    tells the reader exactly how to regenerate it. Separator padding uses
    four spaces on each side of the ``|`` so the three caption cells don't
    cramp at narrow figure widths.
    """
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
    if vis_filename:
        fig.text(
            0.5,
            0.002,
            f"vis: {vis_filename}  (pybench vis --results ...)",
            fontsize=8.5,
            color=_MUTED,
            ha="center",
            va="bottom",
        )


def build_figure(inputs: BuildFigureInputs) -> Figure:
    dims = tuple(inputs.dims_sorted)
    d = len(dims)
    if d == 0:
        return build_empty_figure(inputs.algo, "no dims to render")

    width = max(10.0, 3.8 * d + 1.6)
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
    colors = _series_colors(len(n_jobs_values))

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
    # Compute the companion vis filename via the same hash6 + slug scheme the
    # CLI uses. Kept local to figure.py (rather than routed through a new
    # dataclass field) so the public BuildFigureInputs signature doesn't
    # change. Skips when the partition is empty.
    vis_fn = _vis_filename_for_inputs(inputs)
    _draw_caption(fig, inputs.title_meta, vis_filename=vis_fn)

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
    "is_ours_only_sentinel",
]
