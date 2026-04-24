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
