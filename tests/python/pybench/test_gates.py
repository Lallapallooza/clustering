from __future__ import annotations

import math
from typing import Any

import pytest

from pybench.charts.gates import GateFailure, evaluate_gates
from pybench.recipe import RunResult


def _make_result(
    *,
    recipe_name: str = "dbscan",
    size: int = 1000,
    dims: int = 2,
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
    return RunResult(
        recipe_name=recipe_name,
        size=size,
        dims=dims,
        params=dict(params) if params is not None else {"eps": 0.5, "min_samples": 5},
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


_THRESHOLDS = {"dbscan": 0.85, "kmeans": 0.85}


def test_gate_pass_when_all_finite_and_above_threshold() -> None:
    r = _make_result(ari=0.99, ours_peak_mb=50.0, theirs_peak_mb=100.0)
    assert evaluate_gates([r], _THRESHOLDS) == []


def test_gate_fail_when_ari_below_threshold() -> None:
    r = _make_result(ari=0.5)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "ari" in reason
    assert "0.5" in reason


def test_gate_fail_when_ari_nan() -> None:
    r = _make_result(ari=math.nan)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "ari" in reason
    assert "nan" in reason.lower()


def test_gate_fail_when_ours_peak_mb_zero() -> None:
    r = _make_result(ours_peak_mb=0.0)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "ours_peak_mb" in reason
    assert "0" in reason


def test_gate_fail_when_theirs_peak_mb_zero() -> None:
    r = _make_result(theirs_peak_mb=0.0)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "theirs_peak_mb" in reason


def test_gate_fail_when_ours_peak_mb_nan() -> None:
    r = _make_result(ours_peak_mb=math.nan)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "ours_peak_mb" in reason
    assert "nan" in reason.lower()


def test_gate_fail_when_ours_peak_mb_negative() -> None:
    r = _make_result(ours_peak_mb=-1.0)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "ours_peak_mb" in reason
    assert "-1" in reason


def test_gate_merges_multiple_failures_for_single_row() -> None:
    r = _make_result(ari=0.1, ours_peak_mb=0.0)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    reasons = failures[0].reasons
    assert len(reasons) == 2
    joined = " | ".join(reasons)
    assert "ari" in joined
    assert "ours_peak_mb" in joined


def test_gate_skips_unknown_recipe_with_warning() -> None:
    r = _make_result(recipe_name="unknown_algo", ari=0.01, ours_peak_mb=0.0)
    with pytest.warns(UserWarning, match="no ARI threshold for recipe unknown_algo"):
        failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    reasons = failures[0].reasons
    # ARI is skipped -> only the RSS failure is flagged.
    assert all("ari" not in reason for reason in reasons)
    assert any("ours_peak_mb" in reason for reason in reasons)


def test_gate_failure_preserves_identifying_fields() -> None:
    r = _make_result(
        recipe_name="dbscan",
        size=2500,
        dims=8,
        params={"eps": 0.7, "min_samples": 10, "n_jobs": 4},
        ari=0.0,
    )
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    failure = failures[0]
    assert isinstance(failure, GateFailure)
    assert failure.recipe_name == "dbscan"
    assert failure.size == 2500
    assert failure.dims == 8
    assert failure.params == {"eps": 0.7, "min_samples": 10, "n_jobs": 4}


def test_gate_fail_when_ari_infinite() -> None:
    r = _make_result(ari=math.inf)
    failures = evaluate_gates([r], _THRESHOLDS)
    assert len(failures) == 1
    (reason,) = failures[0].reasons
    assert "ari" in reason


def test_gate_passes_multiple_rows_independently() -> None:
    good = _make_result(size=1000, ari=0.95)
    bad = _make_result(size=5000, ari=0.1)
    failures = evaluate_gates([good, bad], _THRESHOLDS)
    assert len(failures) == 1
    assert failures[0].size == 5000
