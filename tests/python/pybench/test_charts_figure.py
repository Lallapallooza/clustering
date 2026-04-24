from __future__ import annotations

import math
import re
from typing import Any

import matplotlib._pylab_helpers as pylab_helpers
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from pybench.charts.figure import (
    BuildFigureInputs,
    build_empty_figure,
    build_figure,
    format_ratio_label,
)
from pybench.charts.meta import RunMetadata
from pybench.recipe import RunResult


def _make_result(
    *,
    recipe_name: str = "dbscan",
    size: int = 1000,
    dims: int = 2,
    n_jobs: int = 1,
    params: dict[str, Any] | None = None,
    ours_median_ms: float = 10.0,
    theirs_median_ms: float = 20.0,
    ours_peak_mb: float = 50.0,
    theirs_peak_mb: float = 100.0,
    ari: float = 0.99,
    ours_noise_frac: float = 0.01,
    theirs_noise_frac: float = 0.01,
    speedup: float = 2.0,
    timestamp: str = "2026-04-16T00:00:00+00:00",
) -> RunResult:
    p: dict[str, Any] = {"eps": 0.5, "min_samples": 5}
    if params is not None:
        p.update(params)
    p["n_jobs"] = n_jobs
    return RunResult(
        recipe_name=recipe_name,
        size=size,
        dims=dims,
        params=p,
        ours_median_ms=ours_median_ms,
        theirs_median_ms=theirs_median_ms,
        ours_peak_mb=ours_peak_mb,
        theirs_peak_mb=theirs_peak_mb,
        ari=ari,
        ours_noise_frac=ours_noise_frac,
        theirs_noise_frac=theirs_noise_frac,
        speedup=speedup,
        timestamp=timestamp,
    )


def _meta() -> RunMetadata:
    return RunMetadata(
        timestamp_iso="2026-04-16T12:34:56+00:00",
        git_sha="abc1234",
        machine="Xeon-7777",
        canonical_encoding_version=1,
    )


def _grid_inputs(
    *,
    dims: tuple[int, ...],
    sizes: tuple[int, ...],
    n_jobs: tuple[int, ...],
    extra_params: tuple[tuple[str, Any], ...] = (),
    drop: set[tuple[int, int, int]] | None = None,
    make_zero_ours: set[tuple[int, int, int]] | None = None,
    make_sentinel: set[tuple[int, int, int]] | None = None,
    ari_threshold: float | None = 0.85,
    dataset_spec: str = "blobs centers=20",
    algo: str = "dbscan",
) -> BuildFigureInputs:
    drop = drop or set()
    make_zero_ours = make_zero_ours or set()
    make_sentinel = make_sentinel or set()
    results: list[RunResult] = []
    for d in dims:
        for s in sizes:
            for nj in n_jobs:
                if (d, s, nj) in drop:
                    continue
                kwargs: dict[str, Any] = {}
                if (d, s, nj) in make_zero_ours:
                    kwargs["ours_median_ms"] = 0.0
                    kwargs["ours_peak_mb"] = 0.0
                if (d, s, nj) in make_sentinel:
                    # Exact sentinel tuple the runner writes under --ours-only:
                    # theirs_median_ms=0, theirs_peak_mb=0, ari=1.0, speedup=0.
                    kwargs["theirs_median_ms"] = 0.0
                    kwargs["theirs_peak_mb"] = 0.0
                    kwargs["ari"] = 1.0
                    kwargs["speedup"] = 0.0
                results.append(
                    _make_result(
                        dims=d,
                        size=s,
                        n_jobs=nj,
                        params=dict(extra_params),
                        **kwargs,
                    )
                )
    return BuildFigureInputs(
        algo=algo,
        partition_results=tuple(results),
        dims_sorted=tuple(sorted(dims)),
        title_meta=_meta(),
        non_njobs_params=extra_params,
        dataset_spec=dataset_spec,
        ari_threshold=ari_threshold,
    )


def _solid_lines(ax: Any) -> list[Line2D]:
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"]


def _dashed_lines(ax: Any) -> list[Line2D]:
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]


def test_figure_has_2xD_axes_for_D_dims() -> None:
    inputs = _grid_inputs(
        dims=(2, 8, 32, 128),
        sizes=(1000, 5000, 10000),
        n_jobs=(1, 4, 8),
    )
    fig = build_figure(inputs)
    assert len(fig.axes) == 8
    for ax in fig.axes:
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"


def test_figure_has_no_parity_line_per_axes() -> None:
    # Absolute-value chart: there is no parity "theirs = ours" line. The
    # dashed Line2Ds on each axes are the per-n_jobs "theirs" curves, none of
    # which sit at y=1.0 across every x.
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4),
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        for ln in ax.get_lines():
            ydata = list(ln.get_ydata())
            if not ydata:
                continue
            flat_at_one = all(y == 1.0 for y in ydata)
            assert not flat_at_one


def test_figure_has_no_axhspan_win_loss_bands() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4),
    )
    fig = build_figure(inputs)
    # axhspan creates a Rectangle patch whose transform uses blended
    # axes+data coords; we assert there is no large rectangle patch covering
    # the axis by checking that no axes carries Rectangle patches outside of
    # the matplotlib-internal frame.
    from matplotlib.patches import Rectangle

    for ax in fig.axes:
        # The standard axes frame rectangles live in ax.patch, not ax.patches.
        patches = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert patches == []


def test_figure_has_ours_and_theirs_line_per_njobs_per_axes() -> None:
    # 2 * J lines per facet: one solid "ours" + one dashed "theirs" per n_jobs.
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000, 10000),
        n_jobs=(1, 4, 8),
    )
    fig = build_figure(inputs)
    n_njobs = 3
    for ax in fig.axes:
        solid = _solid_lines(ax)
        dashed = _dashed_lines(ax)
        assert len(solid) == n_njobs
        assert len(dashed) == n_njobs
        # Sanity: total Line2Ds on the data axes is 2 * J.
        assert len(ax.get_lines()) == 2 * n_njobs


def test_fill_between_marks_win_region() -> None:
    # Every (size, dim, n_jobs) has ours < theirs (default fixture has
    # ours=10ms, theirs=20ms), so every facet renders a SINGLE facet-level
    # win fill (green). Per-n_jobs fills were replaced with one envelope fill
    # per facet because the per-n_jobs variant stacked into unreadable bands.
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000, 10000),
        n_jobs=(1, 4),
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        # Exactly one PolyCollection per facet: the win envelope fill.
        assert len(polys) == 1


def test_viridis_color_ordering() -> None:
    inputs = _grid_inputs(
        dims=(2,),
        sizes=(1000, 5000),
        n_jobs=(1, 4, 16),
    )
    fig = build_figure(inputs)
    top_ax = fig.axes[0]
    solid = _solid_lines(top_ax)
    assert len(solid) == 3
    # Sort by the n_jobs encoded in the label (format "ours n_jobs=K").
    solid.sort(key=lambda ln: int(ln.get_label().split("=")[1]))
    colors = [ln.get_color() for ln in solid]
    # Viridis goes dark/low-luminance to light/high-luminance. Compare perceived
    # luminance (rec 601 weights) so ordering survives the RGBA encoding.
    lumas = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in colors]
    assert lumas[0] < lumas[1] < lumas[2]


def test_ours_and_theirs_share_color_per_njobs() -> None:
    inputs = _grid_inputs(
        dims=(2,),
        sizes=(1000, 5000),
        n_jobs=(1, 4, 16),
    )
    fig = build_figure(inputs)
    top_ax = fig.axes[0]
    solid = {
        int(ln.get_label().split("=")[1]): ln.get_color() for ln in _solid_lines(top_ax)
    }
    dashed = {
        int(ln.get_label().split("=")[1]): ln.get_color()
        for ln in _dashed_lines(top_ax)
    }
    for nj in (1, 4, 16):
        assert solid[nj] == dashed[nj]


def test_missing_cells_render_as_gaps() -> None:
    inputs = _grid_inputs(
        dims=(2,),
        sizes=(1000, 5000, 10000),
        n_jobs=(1, 4),
        drop={(2, 5000, 4)},
    )
    fig = build_figure(inputs)
    top_ax = fig.axes[0]
    line_for_nj4 = next(
        ln for ln in _solid_lines(top_ax) if ln.get_label() == "ours n_jobs=4"
    )
    ys = list(line_for_nj4.get_ydata())
    assert any(math.isnan(y) for y in ys)


def test_zero_ours_renders_as_nan_gap() -> None:
    # Defensive: a zero (non-finite on log scale) ours value should not emit
    # inf/nan propagation; the curve simply breaks at that point.
    inputs = _grid_inputs(
        dims=(2,),
        sizes=(1000, 5000),
        n_jobs=(1,),
        make_zero_ours={(2, 5000, 1)},
    )
    fig = build_figure(inputs)
    for ax in fig.axes[:2]:
        for ln in _solid_lines(ax):
            ys = list(ln.get_ydata())
            assert not any(math.isinf(y) for y in ys)
            assert any(math.isnan(y) for y in ys)


def test_no_pyplot_figure_leak() -> None:
    # Baseline check: nothing leaked from other tests into the global registry
    # before our call.
    before = list(pylab_helpers.Gcf.get_all_fig_managers())
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4),
    )
    build_figure(inputs)
    after = list(pylab_helpers.Gcf.get_all_fig_managers())
    assert after == before
    assert pylab_helpers.Gcf.get_all_fig_managers() == []


def test_title_contains_meta() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000,),
        n_jobs=(1,),
        extra_params=(("eps", 0.5), ("min_samples", 5)),
        dataset_spec="blobs centers=20 std=3.0",
        algo="dbscan",
    )
    fig = build_figure(inputs)
    suptitle_text = fig.get_suptitle()
    extras = " ".join(t.get_text() for t in fig.texts)
    all_title_text = suptitle_text + " " + extras
    assert "dbscan" in all_title_text
    assert "eps=0.5" in all_title_text
    assert "min_samples=5" in all_title_text
    assert "blobs centers=20 std=3.0" in all_title_text
    assert "2026-04-16T12:34:56+00:00" in all_title_text
    assert "abc1234" in all_title_text
    assert "Xeon-7777" in all_title_text


def test_figure_legend_lists_each_njobs_and_curve_style() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4, 8),
    )
    fig = build_figure(inputs)
    assert len(fig.legends) == 1
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    # ours + theirs style swatches exactly once.
    assert labels.count("ours") == 1
    assert labels.count("theirs") == 1
    # One n_jobs entry per observed n_jobs value.
    for nj in (1, 4, 8):
        assert any(f"n_jobs = {nj}" == lbl for lbl in labels)
    # No parity entry on an absolute-value chart.
    assert not any("parity" in lbl for lbl in labels)


def test_axes_have_endpoint_label_per_njobs() -> None:
    # Each data facet carries exactly J text annotations -- one ratio label per
    # n_jobs curve at its rightmost finite x.
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4),
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        assert len(ax.texts) == 2
        ratio_pattern = re.compile(r"^(\d+(\.\d+)?x|n/a)$")
        for txt in ax.texts:
            assert ratio_pattern.match(txt.get_text())


def test_endpoint_label_uses_format_ratio_label() -> None:
    # One cell, three different (size, n_jobs) combinations with distinct
    # ratios so the rightmost label matches format_ratio_label exactly.
    results = [
        _make_result(
            size=1000, dims=2, n_jobs=1, ours_median_ms=5.0, theirs_median_ms=15.0
        ),
        _make_result(
            size=2000, dims=2, n_jobs=1, ours_median_ms=4.0, theirs_median_ms=12.0
        ),
        _make_result(
            size=1000, dims=2, n_jobs=4, ours_median_ms=3.0, theirs_median_ms=9.0
        ),
        _make_result(
            size=2000, dims=2, n_jobs=4, ours_median_ms=2.0, theirs_median_ms=6.0
        ),
    ]
    inputs = BuildFigureInputs(
        algo="dbscan",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    top_ax = fig.axes[0]
    assert len(top_ax.texts) == 2
    # At rightmost size (2000), ratios are theirs/ours = 12/4 = 3.0 (nj=1) and
    # 6/2 = 3.0 (nj=4); both render as "3.0x".
    labels = sorted(t.get_text() for t in top_ax.texts)
    assert labels == ["3.0x", "3.0x"]
    # Memory row uses the default ours=50, theirs=100 -> 2.0x.
    bot_ax = fig.axes[1]
    labels_mem = sorted(t.get_text() for t in bot_ax.texts)
    assert labels_mem == ["2.0x", "2.0x"]


def test_ours_curve_ydata_equals_run_result_field() -> None:
    # Hand-built single-cell input: the solid line's ydata must equal
    # ours_median_ms on the time axis and ours_peak_mb on the memory axis.
    # Same invariant for dashed = theirs.
    results = [
        _make_result(
            size=1000,
            dims=2,
            n_jobs=1,
            ours_median_ms=7.5,
            theirs_median_ms=11.0,
            ours_peak_mb=42.0,
            theirs_peak_mb=97.0,
        ),
        _make_result(
            size=5000,
            dims=2,
            n_jobs=1,
            ours_median_ms=23.0,
            theirs_median_ms=80.0,
            ours_peak_mb=120.0,
            theirs_peak_mb=310.0,
        ),
    ]
    inputs = BuildFigureInputs(
        algo="dbscan",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    top_ax = fig.axes[0]
    bot_ax = fig.axes[1]
    solid_top = _solid_lines(top_ax)[0]
    dashed_top = _dashed_lines(top_ax)[0]
    assert list(solid_top.get_ydata()) == [7.5, 23.0]
    assert list(dashed_top.get_ydata()) == [11.0, 80.0]
    solid_bot = _solid_lines(bot_ax)[0]
    dashed_bot = _dashed_lines(bot_ax)[0]
    assert list(solid_bot.get_ydata()) == [42.0, 120.0]
    assert list(dashed_bot.get_ydata()) == [97.0, 310.0]


def test_time_row_y_ticks_format_as_ms_or_s() -> None:
    # Time axis labels read as "12 ms", "1.2 s", etc -- not as ratio "12x".
    # We ensure the data range crosses the ms/s boundary so both units appear.
    results = [
        _make_result(
            size=s, dims=2, n_jobs=1, ours_median_ms=ms_ours, theirs_median_ms=ms_theirs
        )
        for s, ms_ours, ms_theirs in (
            (1000, 5.0, 10.0),
            (5000, 50.0, 100.0),
            (10000, 500.0, 1000.0),
            (50000, 5000.0, 10000.0),
        )
    ]
    inputs = BuildFigureInputs(
        algo="kmeans",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    time_ax = fig.axes[0]
    # Force a draw so tick labels materialize.
    fig.canvas.draw()
    labels = [t.get_text() for t in time_ax.get_yticklabels() if t.get_text().strip()]
    assert labels, "time axis produced no tick labels"
    # Every non-empty label ends with an ms or s unit; no ratio "x".
    for lbl in labels:
        stripped = lbl.strip()
        assert not stripped.endswith("x"), f"time label {lbl!r} looks like a ratio"
        assert stripped.endswith("ms") or stripped.endswith("s"), (
            f"time label {lbl!r} missing ms/s suffix"
        )


def test_mem_row_y_ticks_format_as_MB_or_GB() -> None:
    results = [
        _make_result(
            size=s, dims=2, n_jobs=1, ours_peak_mb=mb_ours, theirs_peak_mb=mb_theirs
        )
        for s, mb_ours, mb_theirs in (
            (1000, 5.0, 10.0),
            (5000, 50.0, 200.0),
            (10000, 500.0, 2000.0),
        )
    ]
    inputs = BuildFigureInputs(
        algo="kmeans",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    mem_ax = fig.axes[1]
    fig.canvas.draw()
    labels = [t.get_text() for t in mem_ax.get_yticklabels() if t.get_text().strip()]
    assert labels, "memory axis produced no tick labels"
    for lbl in labels:
        stripped = lbl.strip()
        assert not stripped.endswith("x"), f"memory label {lbl!r} looks like a ratio"
        assert (
            stripped.endswith("MB")
            or stripped.endswith("GB")
            or stripped.endswith("KB")
        ), f"memory label {lbl!r} missing MB/GB/KB suffix"


def test_row_ylabels_describe_absolute_metric() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1,),
    )
    fig = build_figure(inputs)
    top_left = fig.axes[0]
    bot_left = fig.axes[2]
    # The leftmost-column axes carry the row labels; others leave it blank to
    # avoid repetition.
    assert "time" in top_left.get_ylabel()
    assert "memory" in bot_left.get_ylabel()
    # Crucially, the label does NOT read "theirs / ours" (old ratio chart).
    assert "theirs / ours" not in top_left.get_ylabel()
    assert "theirs / ours" not in bot_left.get_ylabel()


def test_build_empty_figure() -> None:
    fig = build_empty_figure("dbscan", "no rows matched")
    assert len(fig.axes) == 1
    title = fig.axes[0].get_title()
    assert "dbscan" in title
    assert "no data" in title


@pytest.mark.parametrize(
    ("r", "expected"),
    [
        (0.42, "0.42x"),
        (0.99, "0.99x"),
        (1.0, "1.0x"),
        (3.1, "3.1x"),
        (9.95, "10.0x"),
        (12.0, "12x"),
        (math.nan, "n/a"),
        (math.inf, "n/a"),
    ],
)
def test_format_ratio_label(r: float, expected: str) -> None:
    assert format_ratio_label(r) == expected


def test_facet_has_major_yticks_for_memory_in_MB_range() -> None:
    # Regression: the memory axis must always carry labeled gridlines, no
    # matter where the data sits on the log scale.
    results = [
        _make_result(
            size=s,
            dims=2,
            n_jobs=1,
            ours_peak_mb=mb_ours,
            theirs_peak_mb=mb_theirs,
        )
        for s, mb_ours, mb_theirs in (
            (5000, 4.5, 0.44),
            (10000, 5.0, 0.67),
            (50000, 8.5, 2.81),
            (100000, 12.9, 5.48),
            (250000, 26.0, 13.49),
        )
    ]
    inputs = BuildFigureInputs(
        algo="kmeans",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="blobs centers=16",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    mem_ax = fig.axes[1]
    tick_positions = [t for t in mem_ax.get_yticks() if math.isfinite(t)]
    tick_labels = [t.get_text() for t in mem_ax.get_yticklabels()]
    assert tick_positions, "memory axis has no major ticks"
    assert any(lbl.strip() for lbl in tick_labels), "memory axis ticks are unlabeled"


def test_figure_savefig_produces_nonempty_png(tmp_path: Any) -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4),
    )
    fig = build_figure(inputs)
    out = tmp_path / "x.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_fully_ours_only_facet_renders_ours_only_annotation() -> None:
    # Every row is the --ours-only sentinel tuple. The facet must drop the
    # dashed theirs curve, the fill band, and the endpoint ratio label, and
    # instead carry exactly one "ours-only" text annotation per facet.
    sizes = (1000, 5000, 10000)
    n_jobs = (1, 4)
    dims = (2, 8)
    make_sentinel = {(d, s, nj) for d in dims for s in sizes for nj in n_jobs}
    inputs = _grid_inputs(
        dims=dims,
        sizes=sizes,
        n_jobs=n_jobs,
        make_sentinel=make_sentinel,
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        # No dashed theirs lines at all.
        assert _dashed_lines(ax) == []
        # No win-region fills.
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert polys == []
        # Exactly one annotation: the "ours-only" placeholder. No ratio
        # labels appear.
        assert len(ax.texts) == 1
        only = ax.texts[0].get_text().lower()
        assert "ours-only" in only
        # Solid ours lines still render normally, one per n_jobs.
        assert len(_solid_lines(ax)) == len(n_jobs)


def test_row_mixed_theirs_has_nan_gap_at_sentinel() -> None:
    # Within a single (dim, n_jobs), some sizes are sentinel and others are
    # real. The theirs dashed line must carry NaN at sentinel x-positions
    # while the ours solid line remains finite there.
    dims = (2,)
    n_jobs = (1, 4)
    sizes = (1000, 5000, 10000)
    # Make the MIDDLE size a sentinel for nj=1 only, leaving the rightmost
    # size as a real row so the endpoint label still renders for that n_jobs.
    # For nj=4 make the RIGHTMOST size sentinel so its endpoint label must
    # be suppressed.
    make_sentinel = {(2, 5000, 1), (2, 10000, 4)}
    inputs = _grid_inputs(
        dims=dims,
        sizes=sizes,
        n_jobs=n_jobs,
        make_sentinel=make_sentinel,
    )
    fig = build_figure(inputs)
    time_ax = fig.axes[0]
    # Ours lines: both finite everywhere.
    for ln in _solid_lines(time_ax):
        ys = list(ln.get_ydata())
        for y in ys:
            assert math.isfinite(y)
    # Dashed line for nj=1: NaN at idx of the 5000 sentinel, finite at 1000
    # and 10000.
    dashed_nj1 = next(
        ln for ln in _dashed_lines(time_ax) if ln.get_label() == "theirs n_jobs=1"
    )
    xs1 = list(dashed_nj1.get_xdata())
    ys1 = list(dashed_nj1.get_ydata())
    assert xs1 == list(sizes)
    # Sentinel at x=5000 -> NaN; real rows at 1000 and 10000 -> finite.
    idx_5000 = xs1.index(5000)
    idx_10000 = xs1.index(10000)
    assert math.isnan(ys1[idx_5000])
    assert math.isfinite(ys1[idx_10000])
    # Dashed line for nj=4: NaN at the rightmost x, endpoint label suppressed.
    dashed_nj4 = next(
        ln for ln in _dashed_lines(time_ax) if ln.get_label() == "theirs n_jobs=4"
    )
    xs4 = list(dashed_nj4.get_xdata())
    ys4 = list(dashed_nj4.get_ydata())
    idx_last4 = xs4.index(10000)
    assert math.isnan(ys4[idx_last4])
    # Endpoint ratio labels: one for nj=1 (real rightmost), zero for nj=4
    # (rightmost is sentinel).
    ratio_pattern = re.compile(r"^(\d+(\.\d+)?x|n/a)$")
    ratio_labels = [t for t in time_ax.texts if ratio_pattern.match(t.get_text())]
    assert len(ratio_labels) == 1


def test_column_mixed_dims_handled() -> None:
    # Some dims are fully ours-only (all sizes x n_jobs are sentinel). Others
    # are fully matched. The ours-only dim renders the annotation; the matched
    # dim renders normal dashed curves.
    sizes = (1000, 5000)
    n_jobs = (1, 4)
    # Dim 2: fully matched (no sentinel). Dim 8: fully ours-only.
    make_sentinel = {(8, s, nj) for s in sizes for nj in n_jobs}
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=sizes,
        n_jobs=n_jobs,
        make_sentinel=make_sentinel,
    )
    fig = build_figure(inputs)
    # Axes layout: [time_dim2, time_dim8, mem_dim2, mem_dim8].
    time_dim2 = fig.axes[0]
    time_dim8 = fig.axes[1]
    mem_dim2 = fig.axes[2]
    mem_dim8 = fig.axes[3]
    # Matched dim: dashed curves present, endpoint labels present.
    assert len(_dashed_lines(time_dim2)) == len(n_jobs)
    assert len(_dashed_lines(mem_dim2)) == len(n_jobs)
    for ax in (time_dim2, mem_dim2):
        ratio_pattern = re.compile(r"^(\d+(\.\d+)?x|n/a)$")
        assert any(ratio_pattern.match(t.get_text()) for t in ax.texts)
    # Ours-only dim: no dashed curves, annotation present.
    for ax in (time_dim8, mem_dim8):
        assert _dashed_lines(ax) == []
        assert len(ax.texts) == 1
        assert "ours-only" in ax.texts[0].get_text().lower()


def test_sentinel_predicate_requires_all_four_zeros() -> None:
    # A row with only one or two sentinel fields is a REAL row, not a
    # sentinel. Here theirs_median_ms=0 but theirs_peak_mb, ari, speedup are
    # all non-sentinel values, so the predicate must NOT match.
    results = [
        _make_result(
            size=1000,
            dims=2,
            n_jobs=1,
            ours_median_ms=10.0,
            theirs_median_ms=0.0,  # only one of the four matches
            ours_peak_mb=40.0,
            theirs_peak_mb=5.0,  # non-zero
            ari=0.7,  # not 1.0
            speedup=3.0,  # not 0.0
        ),
        _make_result(
            size=5000,
            dims=2,
            n_jobs=1,
            ours_median_ms=20.0,
            theirs_median_ms=80.0,
            ours_peak_mb=80.0,
            theirs_peak_mb=200.0,
            ari=0.9,
            speedup=4.0,
        ),
    ]
    inputs = BuildFigureInputs(
        algo="dbscan",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    time_ax = fig.axes[0]
    # Facet treated as matched: dashed curve present, no "ours-only" annotation.
    assert len(_dashed_lines(time_ax)) == 1
    for txt in time_ax.texts:
        assert "ours-only" not in txt.get_text().lower()
    # Memory axis: theirs_peak_mb=5.0 at size=1000 is a real finite value, so
    # the dashed mem curve's ydata at size=1000 is NOT NaN -- the sentinel
    # predicate didn't match.
    mem_ax = fig.axes[1]
    dashed_mem = _dashed_lines(mem_ax)[0]
    xs_mem = list(dashed_mem.get_xdata())
    ys_mem = list(dashed_mem.get_ydata())
    idx_1000 = xs_mem.index(1000)
    assert math.isfinite(ys_mem[idx_1000])


def test_sentinel_check_is_exact_float_equality() -> None:
    # A row with theirs_median_ms=0.0000001 is NOT a sentinel -- the predicate
    # uses exact == on the float, not math.isclose. The 1e-7 value is finite,
    # positive, and must flow through as a real point.
    results = [
        _make_result(
            size=1000,
            dims=2,
            n_jobs=1,
            ours_median_ms=10.0,
            theirs_median_ms=0.0000001,  # NOT exactly 0.0
            ours_peak_mb=40.0,
            theirs_peak_mb=0.0,
            ari=1.0,
            speedup=0.0,
        ),
        _make_result(
            size=5000,
            dims=2,
            n_jobs=1,
            ours_median_ms=20.0,
            theirs_median_ms=40.0,
            ours_peak_mb=80.0,
            theirs_peak_mb=160.0,
            ari=0.99,
            speedup=2.0,
        ),
    ]
    inputs = BuildFigureInputs(
        algo="dbscan",
        partition_results=tuple(results),
        dims_sorted=(2,),
        title_meta=_meta(),
        non_njobs_params=(),
        dataset_spec="",
        ari_threshold=None,
    )
    fig = build_figure(inputs)
    time_ax = fig.axes[0]
    # Dashed theirs line exists (the sentinel predicate did not fire).
    dashed = _dashed_lines(time_ax)
    assert len(dashed) == 1
    # No "ours-only" annotation, because row is not a sentinel.
    for txt in time_ax.texts:
        assert "ours-only" not in txt.get_text().lower()
    xs = list(dashed[0].get_xdata())
    ys = list(dashed[0].get_ydata())
    # The 1e-7 value is finite and positive, so _to_positive_or_nan keeps it.
    idx_1000 = xs.index(1000)
    assert math.isfinite(ys[idx_1000])
    assert ys[idx_1000] == 0.0000001
