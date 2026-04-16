from __future__ import annotations

import math
from typing import Any

import matplotlib._pylab_helpers as pylab_helpers
import pytest
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
    ari_threshold: float | None = 0.85,
    dataset_spec: str = "blobs centers=20",
    algo: str = "dbscan",
) -> BuildFigureInputs:
    drop = drop or set()
    make_zero_ours = make_zero_ours or set()
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


def _ratio_lines(ax: Any) -> list[Line2D]:
    # Solid lines are the per-n_jobs series; the dashed one is the parity line.
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"]


def _parity_lines(ax: Any) -> list[Line2D]:
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


def test_figure_has_one_parity_line_per_axes() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4),
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        parity = _parity_lines(ax)
        assert len(parity) == 1
        ydata = parity[0].get_ydata()
        assert all(y == 1.0 for y in ydata)


def test_figure_has_one_line_per_njobs_per_axes() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000, 10000),
        n_jobs=(1, 4, 8),
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        assert len(_ratio_lines(ax)) == 3


def test_viridis_color_ordering() -> None:
    inputs = _grid_inputs(
        dims=(2,),
        sizes=(1000, 5000),
        n_jobs=(1, 4, 16),
    )
    fig = build_figure(inputs)
    top_ax = fig.axes[0]
    ratio_lines = _ratio_lines(top_ax)
    assert len(ratio_lines) == 3
    # Sort by the n_jobs encoded in the label.
    ratio_lines.sort(key=lambda ln: int(ln.get_label().split("=")[1]))
    colors = [ln.get_color() for ln in ratio_lines]
    # Viridis goes dark/low-luminance to light/high-luminance. Compare perceived
    # luminance (rec 601 weights) so ordering survives the RGBA encoding.
    lumas = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in colors]
    assert lumas[0] < lumas[1] < lumas[2]


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
        ln for ln in _ratio_lines(top_ax) if ln.get_label() == "n_jobs=4"
    )
    ys = list(line_for_nj4.get_ydata())
    assert any(math.isnan(y) for y in ys)


def test_ratio_is_nan_when_ours_is_zero() -> None:
    inputs = _grid_inputs(
        dims=(2,),
        sizes=(1000, 5000),
        n_jobs=(1,),
        make_zero_ours={(2, 5000, 1)},
    )
    fig = build_figure(inputs)
    for ax in fig.axes[:2]:
        for ln in _ratio_lines(ax):
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


def test_endpoint_annotations_placed_per_ratio_line() -> None:
    inputs = _grid_inputs(
        dims=(2, 8),
        sizes=(1000, 5000),
        n_jobs=(1, 4, 8),
    )
    fig = build_figure(inputs)
    for ax in fig.axes:
        ratio_lines = _ratio_lines(ax)
        assert len(ax.texts) == len(ratio_lines) > 0
        for t in ax.texts:
            text = t.get_text()
            assert text.endswith("x") or text == "n/a"


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
