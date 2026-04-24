from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pybench import cli as cli_module
from pybench.charts.meta import RunMetadata
from pybench.charts.results_io import save_results
from pybench.recipe import RunResult
from pybench.runner import LabelsBundle


def _make_result(
    *,
    recipe_name: str = "dbscan",
    size: int = 1000,
    dims: int = 2,
    params: dict[str, Any] | None = None,
    ari: float = 0.99,
    ours_peak_mb: float = 50.0,
    theirs_peak_mb: float = 100.0,
) -> RunResult:
    return RunResult(
        recipe_name=recipe_name,
        size=size,
        dims=dims,
        params=dict(params)
        if params is not None
        else {"eps": 0.5, "min_samples": 5, "n_jobs": 1},
        ours_median_ms=10.0,
        theirs_median_ms=20.0,
        ours_peak_mb=ours_peak_mb,
        theirs_peak_mb=theirs_peak_mb,
        ari=ari,
        ours_noise_frac=0.01,
        theirs_noise_frac=0.01,
        speedup=2.0,
        timestamp="2026-04-16T00:00:00+00:00",
    )


def _meta() -> RunMetadata:
    return RunMetadata(
        timestamp_iso="2026-04-16T12:34:56+00:00",
        git_sha="abcdef0",
        machine="Test CPU",
        canonical_encoding_version=1,
    )


def _stub_run_one_factory(*, ari: float = 0.99, ours_peak_mb: float = 50.0):
    def _stub_run_one(recipe, size, dims=None, params=None, ours_only=False):
        effective_params = (
            dict(params) if params is not None else dict(recipe.default_params)
        )
        return _make_result(
            recipe_name=recipe.name,
            size=size,
            dims=dims if dims is not None else recipe.dataset.n_features,
            params=effective_params,
            ari=ari,
            ours_peak_mb=ours_peak_mb,
        )

    return _stub_run_one


def _stub_run_one_with_labels_factory(
    *, ari: float = 0.99, ours_peak_mb: float = 50.0, capture_labels: bool = True
):
    """Stub that returns @c (RunResult, LabelsBundle | None).

    Honors @c capture_labels so tests can drive the @c --no-labels path without
    going through the real :func:`make_data_with_gt`.
    """

    def _stub(
        recipe,
        size,
        dims=None,
        params=None,
        ours_only=False,
        capture_labels=capture_labels,
    ):
        effective_params = (
            dict(params) if params is not None else dict(recipe.default_params)
        )
        result = _make_result(
            recipe_name=recipe.name,
            size=size,
            dims=dims if dims is not None else recipe.dataset.n_features,
            params=effective_params,
            ari=ari,
            ours_peak_mb=ours_peak_mb,
        )
        if not capture_labels:
            return result, None
        n = size
        bundle = LabelsBundle(
            gt_labels=np.zeros(n, dtype=np.int32),
            ours_labels=np.zeros(n, dtype=np.int32),
            theirs_labels=(None if ours_only else np.zeros(n, dtype=np.int32)),
            projection_2d=np.zeros((n, 2), dtype=np.float32),
        )
        return result, bundle

    return _stub


def _invoke_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])
    cli_module.main()


def test_cli_live_writes_wrapper_json_and_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )
    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        [
            "--algo",
            "dbscan",
            "--sizes",
            "1000",
            "--dims",
            "2",
            "--out",
            str(out_dir),
        ],
    )

    json_path = out_dir / "results.json"
    assert json_path.is_file()
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert set(raw.keys()) == {"meta", "results"}

    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 1
    assert all(p.name.startswith("dbscan") for p in pngs)


def test_cli_replot_regenerates_from_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "in" / "results.json"
    results = [
        _make_result(size=1000, params={"eps": 0.5, "min_samples": 5, "n_jobs": 1}),
        _make_result(size=5000, params={"eps": 0.5, "min_samples": 5, "n_jobs": 1}),
    ]
    save_results(json_path, _meta(), results)

    mtime_before = json_path.stat().st_mtime_ns

    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        ["--replot", str(json_path), "--out", str(out_dir)],
    )

    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 1

    mtime_after = json_path.stat().st_mtime_ns
    assert mtime_before == mtime_after


def test_cli_replot_mutually_exclusive_with_algo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "results.json"
    save_results(json_path, _meta(), [_make_result()])

    with pytest.raises(SystemExit) as exc:
        _invoke_main(
            monkeypatch,
            ["--replot", str(json_path), "--algo", "dbscan"],
        )
    assert exc.value.code != 0


def test_cli_gate_fail_no_charts_no_json_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.4),
    )
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit) as exc:
        _invoke_main(
            monkeypatch,
            [
                "--algo",
                "dbscan",
                "--sizes",
                "1000",
                "--dims",
                "2",
                "--out",
                str(out_dir),
            ],
        )
    assert exc.value.code not in (0, None)

    assert not (out_dir / "results.json").exists()
    assert not list(out_dir.glob("*.png"))


def test_cli_live_default_writes_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default (labels-on) CLI run writes one @c .labels.npz per partition."""
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )
    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        [
            "--algo",
            "dbscan",
            "--sizes",
            "1000",
            "--dims",
            "2",
            "--out",
            str(out_dir),
        ],
    )

    sidecars = list(out_dir.glob("*.labels.npz"))
    assert len(sidecars) >= 1
    assert all(p.name.startswith("dbscan") for p in sidecars)

    # Sidecar prefix must match the PNG so both artifacts co-locate.
    pngs = list(out_dir.glob("*.png"))
    png_prefixes = {p.name[: -len(".png")] for p in pngs}
    sidecar_prefixes = {p.name[: -len(".labels.npz")] for p in sidecars}
    assert png_prefixes == sidecar_prefixes


def test_cli_live_no_labels_skips_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """@c --no-labels disables both label capture and sidecar write."""
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )
    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        [
            "--algo",
            "dbscan",
            "--sizes",
            "1000",
            "--dims",
            "2",
            "--out",
            str(out_dir),
            "--no-labels",
        ],
    )

    assert list(out_dir.glob("*.png")), "sanity: CLI still writes charts"
    assert not list(out_dir.glob("*.labels.npz")), (
        "sidecars must not exist under --no-labels"
    )


# ---------------------------------------------------------------------------
# Subparser routing and the vis subcommand.
# ---------------------------------------------------------------------------


def test_cli_bench_subcommand_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit @c pybench bench subcommand accepts the same flags as before."""
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )
    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        [
            "bench",
            "--algo",
            "dbscan",
            "--sizes",
            "1000",
            "--dims",
            "2",
            "--out",
            str(out_dir),
        ],
    )

    assert (out_dir / "results.json").is_file()
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 1
    assert all(p.name.startswith("dbscan") for p in pngs)


def test_cli_flat_flags_route_to_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flat-flag invocation (no subcommand) is routed to @c bench by the compat shim."""
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )
    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        [
            "--algo",
            "dbscan",
            "--sizes",
            "1000",
            "--dims",
            "2",
            "--out",
            str(out_dir),
        ],
    )

    assert (out_dir / "results.json").is_file()


def test_cli_unknown_subcommand_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown positional subcommand is rejected with a clear error, not silently routed."""
    with pytest.raises(SystemExit) as exc:
        _invoke_main(monkeypatch, ["diff"])
    # _normalize_argv raises SystemExit with a string message; argparse uses int codes.
    assert exc.value.code != 0


def test_cli_replot_routes_via_compat_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """@c --replot PATH still works as a flat flag via the compat shim."""
    json_path = tmp_path / "in" / "results.json"
    results = [
        _make_result(size=1000, params={"eps": 0.5, "min_samples": 5, "n_jobs": 1}),
        _make_result(size=5000, params={"eps": 0.5, "min_samples": 5, "n_jobs": 1}),
    ]
    save_results(json_path, _meta(), results)

    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        ["--replot", str(json_path), "--out", str(out_dir)],
    )

    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 1
    # Ensure the byte-exact filename scheme (no "_vis_" suffix) is used.
    assert all("_vis_" not in p.name for p in pngs)


def _run_bench_to_produce_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Drive @c pybench bench to produce a populated @c out_dir.

    Returns the out_dir so the caller can locate the @c results.json and the
    @c *.labels.npz sidecar(s).
    """
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )
    out_dir = tmp_path / "out"
    _invoke_main(
        monkeypatch,
        [
            "bench",
            "--algo",
            "dbscan",
            "--sizes",
            "1000",
            "--dims",
            "2",
            "--out",
            str(out_dir),
        ],
    )
    return out_dir


def test_cli_vis_subcommand_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a fresh bench run's sidecar present, @c pybench vis produces PNGs."""
    out_dir = _run_bench_to_produce_sidecar(tmp_path, monkeypatch)
    assert list(out_dir.glob("*.labels.npz"))

    vis_dir = tmp_path / "vis"
    _invoke_main(
        monkeypatch,
        [
            "vis",
            "--results",
            str(out_dir / "results.json"),
            "--out",
            str(vis_dir),
        ],
    )

    vis_pngs = list(vis_dir.glob("*.png"))
    assert len(vis_pngs) >= 1
    assert all("_vis_" in p.name for p in vis_pngs)


def test_cli_vis_missing_sidecar_regens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the sidecar triggers full regen fallback; PNGs are still produced."""
    out_dir = _run_bench_to_produce_sidecar(tmp_path, monkeypatch)
    # Remove every sidecar so vis must regen.
    for p in out_dir.glob("*.labels.npz"):
        p.unlink()

    # Re-monkeypatch run_one_with_labels for the regen path; same stub.
    monkeypatch.setattr(
        "pybench.cli.run_one_with_labels",
        _stub_run_one_with_labels_factory(ari=0.99),
    )

    vis_dir = tmp_path / "vis"
    _invoke_main(
        monkeypatch,
        [
            "vis",
            "--results",
            str(out_dir / "results.json"),
            "--out",
            str(vis_dir),
        ],
    )

    vis_pngs = list(vis_dir.glob("*.png"))
    assert len(vis_pngs) >= 1


def test_cli_vis_no_regen_errors_on_missing_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """@c --no-regen + missing sidecar exits non-zero instead of regenerating."""
    out_dir = _run_bench_to_produce_sidecar(tmp_path, monkeypatch)
    for p in out_dir.glob("*.labels.npz"):
        p.unlink()

    vis_dir = tmp_path / "vis"
    with pytest.raises(SystemExit) as exc:
        _invoke_main(
            monkeypatch,
            [
                "vis",
                "--results",
                str(out_dir / "results.json"),
                "--out",
                str(vis_dir),
                "--no-regen",
            ],
        )
    assert exc.value.code not in (0, None)
    # No vis PNG should have been written.
    assert not list(vis_dir.glob("*.png"))


def test_cli_vis_filename_carries_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vis PNGs carry the @c _vis_ infix so they do not collide with @c bench charts."""
    out_dir = _run_bench_to_produce_sidecar(tmp_path, monkeypatch)
    vis_dir = tmp_path / "vis"
    _invoke_main(
        monkeypatch,
        [
            "vis",
            "--results",
            str(out_dir / "results.json"),
            "--out",
            str(vis_dir),
        ],
    )
    vis_pngs = list(vis_dir.glob("*.png"))
    assert vis_pngs
    # The expected filename shape ends with `_vis_{hash6}.png`. Every name must
    # contain the `_vis_` token before the hash block (i.e. before `.png`).
    for p in vis_pngs:
        stem = p.name[: -len(".png")]
        assert "_vis_" in stem, f"{p.name} missing _vis_ infix"


def test_cli_bench_and_vis_filenames_do_not_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendering bench PNGs and vis PNGs into the same dir must never collide."""
    out_dir = _run_bench_to_produce_sidecar(tmp_path, monkeypatch)
    bench_pngs = {p.name for p in out_dir.glob("*.png")}

    # Write vis PNGs alongside bench PNGs in the same out dir to stress-test
    # filename isolation.
    _invoke_main(
        monkeypatch,
        [
            "vis",
            "--results",
            str(out_dir / "results.json"),
            "--out",
            str(out_dir),
        ],
    )
    all_pngs = {p.name for p in out_dir.glob("*.png")}
    vis_pngs = all_pngs - bench_pngs
    # Sanity: vis did produce new files.
    assert vis_pngs
    # No vis filename matches any bench filename.
    assert bench_pngs.isdisjoint(vis_pngs)


def _make_sentinel_result(
    *,
    recipe_name: str = "dbscan",
    size: int = 1000,
    dims: int = 2,
    params: dict[str, Any] | None = None,
) -> RunResult:
    """Construct a @c --ours-only sentinel :class:`RunResult`.

    Matches the exact tuple that :func:`pybench.runner.run_one_with_labels`
    writes when @c ours_only=True: @c theirs_median_ms=0.0,
    @c theirs_peak_mb=0.0, @c ari=1.0, @c speedup=0.0.
    """
    return RunResult(
        recipe_name=recipe_name,
        size=size,
        dims=dims,
        params=dict(params)
        if params is not None
        else {"eps": 0.5, "min_samples": 5, "n_jobs": 1},
        ours_median_ms=10.0,
        theirs_median_ms=0.0,
        ours_peak_mb=0.0,
        theirs_peak_mb=0.0,
        ari=1.0,
        ours_noise_frac=0.01,
        theirs_noise_frac=0.0,
        speedup=0.0,
        timestamp="2026-04-16T00:00:00+00:00",
    )


def test_cli_vis_sentinel_regen_propagates_ours_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vis regen on a sentinel (ours-only) row must call @c run_one_with_labels
    with @c ours_only=True so the sklearn baseline is not silently re-run."""
    # Build an @c --ours-only sentinel results.json with no sidecar alongside.
    json_path = tmp_path / "bench" / "results.json"
    sentinel = _make_sentinel_result()
    save_results(json_path, _meta(), [sentinel])

    # Capture every call to @c run_one_with_labels so we can assert the
    # @c ours_only kwarg value.
    seen_calls: list[dict[str, Any]] = []

    def _recording_stub(
        recipe,
        size,
        dims=None,
        params=None,
        ours_only=False,
        capture_labels=True,
    ):
        seen_calls.append(
            {
                "recipe": recipe.name,
                "size": size,
                "dims": dims,
                "ours_only": ours_only,
                "capture_labels": capture_labels,
            }
        )
        effective_params = (
            dict(params) if params is not None else dict(recipe.default_params)
        )
        # Returning a sentinel-shaped result keeps the regen path self-consistent.
        fresh = _make_sentinel_result(
            recipe_name=recipe.name,
            size=size,
            dims=dims if dims is not None else recipe.dataset.n_features,
            params=effective_params,
        )
        n = size
        bundle = LabelsBundle(
            gt_labels=np.zeros(n, dtype=np.int32),
            ours_labels=np.zeros(n, dtype=np.int32),
            theirs_labels=None,  # Mirrors what ours_only=True would really produce.
            projection_2d=np.zeros((n, 2), dtype=np.float32),
        )
        return fresh, bundle

    monkeypatch.setattr("pybench.cli.run_one_with_labels", _recording_stub)

    vis_dir = tmp_path / "vis"
    _invoke_main(
        monkeypatch,
        [
            "vis",
            "--results",
            str(json_path),
            "--out",
            str(vis_dir),
        ],
    )

    # Regen must have fired (no sidecar on disk) AND propagated ours_only=True.
    assert seen_calls, "expected run_one_with_labels to be called for regen"
    assert all(call["ours_only"] is True for call in seen_calls), (
        f"regen must propagate ours_only=True on a sentinel row; saw {seen_calls}"
    )
    # Vis must still have produced its PNG on the regen path.
    assert list(vis_dir.glob("*.png"))


def test_cli_vis_regen_caption_uses_fresh_ari(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When regen fires, the figure caption's ARI comes from the fresh
    :class:`RunResult`, not the stale one loaded from @c results.json."""
    # Bench-time row claims ari=0.40; regen stub returns ari=0.91. Vis must
    # use 0.91 in the caption.
    stale_ari = 0.40
    fresh_ari = 0.91
    json_path = tmp_path / "bench" / "results.json"
    stale = _make_result(ari=stale_ari)
    save_results(json_path, _meta(), [stale])

    def _regen_stub(
        recipe,
        size,
        dims=None,
        params=None,
        ours_only=False,
        capture_labels=True,
    ):
        effective_params = (
            dict(params) if params is not None else dict(recipe.default_params)
        )
        fresh = _make_result(
            recipe_name=recipe.name,
            size=size,
            dims=dims if dims is not None else recipe.dataset.n_features,
            params=effective_params,
            ari=fresh_ari,
        )
        n = size
        bundle = LabelsBundle(
            gt_labels=np.zeros(n, dtype=np.int32),
            ours_labels=np.zeros(n, dtype=np.int32),
            theirs_labels=(None if ours_only else np.zeros(n, dtype=np.int32)),
            projection_2d=np.zeros((n, 2), dtype=np.float32),
        )
        return fresh, bundle

    monkeypatch.setattr("pybench.cli.run_one_with_labels", _regen_stub)

    captured_inputs: list[Any] = []
    real_build_vis_figure = cli_module.build_vis_figure

    def _spy_build_vis(inputs):
        captured_inputs.append(inputs)
        return real_build_vis_figure(inputs)

    monkeypatch.setattr("pybench.cli.build_vis_figure", _spy_build_vis)

    vis_dir = tmp_path / "vis"
    _invoke_main(
        monkeypatch,
        [
            "vis",
            "--results",
            str(json_path),
            "--out",
            str(vis_dir),
        ],
    )

    assert captured_inputs, "expected build_vis_figure to be called"
    # The VisInputs fed to the figure carries the fresh ari, not the stale one.
    assert captured_inputs[0].ari == pytest.approx(fresh_ari)
    assert captured_inputs[0].ari != pytest.approx(stale_ari)
