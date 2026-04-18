from __future__ import annotations

import math
import os

import numpy as np
import pytest

from pybench.recipe import DatasetSpec, Recipe
from pybench.runner import (
    _thread_limit_for,
    _knee_eps,
    _prepare_params,
    _resolve_eps,
    _with_thread_limits,
    make_data,
)


def _stub_recipe(eps_policy: str) -> Recipe:
    return Recipe(
        name="x",
        ours=lambda data, **kw: data,
        theirs=lambda data, **kw: data,
        eps_policy=eps_policy,
    )


def test_make_data_blobs_below_vmf_threshold() -> None:
    spec = DatasetSpec(n_features=8, centers=5, random_state=0, vmf_switch_dim=16)
    data = make_data(1000, spec)
    assert data.shape == (1000, 8)
    assert data.dtype == np.float32
    norms = np.linalg.norm(data, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_make_data_vmf_above_vmf_threshold_is_unit_norm() -> None:
    spec = DatasetSpec(
        n_features=64, centers=5, random_state=0, vmf_switch_dim=16, vmf_kappa=50.0
    )
    data = make_data(1000, spec)
    assert data.shape == (1000, 64)
    assert data.dtype == np.float32
    norms = np.linalg.norm(data, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_resolve_eps_sqrt_d_scales_with_dim() -> None:
    base = 10.0
    eps_2 = _resolve_eps(
        "sqrt_d", base, np.zeros((10, 2), dtype=np.float32), min_samples=5
    )
    eps_128 = _resolve_eps(
        "sqrt_d", base, np.zeros((10, 128), dtype=np.float32), min_samples=5
    )
    assert math.isclose(eps_2, 10.0, rel_tol=1e-6)
    assert math.isclose(eps_128, 10.0 * math.sqrt(64), rel_tol=1e-6)


def test_knee_eps_on_blob_data_is_positive_and_finite() -> None:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 4)) * 10
    data = np.vstack(
        [centers[i] + rng.normal(size=(100, 4)) for i in range(len(centers))]
    ).astype(np.float32)
    eps = _knee_eps(data, k=5)
    assert math.isfinite(eps)
    assert eps > 0


def test_prepare_params_fixed_policy_returns_copy() -> None:
    params = {"eps": 3.0, "min_samples": 5, "n_jobs": 1}
    effective = _prepare_params(
        _stub_recipe("fixed"), np.zeros((10, 2), dtype=np.float32), params
    )
    assert effective == params
    assert effective is not params


def test_prepare_params_sqrt_d_rescales_eps() -> None:
    params = {"eps": 10.0, "min_samples": 5}
    effective = _prepare_params(
        _stub_recipe("sqrt_d"), np.zeros((10, 128), dtype=np.float32), params
    )
    assert math.isclose(effective["eps"], 10.0 * math.sqrt(64), rel_tol=1e-6)
    assert params["eps"] == 10.0  # input untouched


def test_prepare_params_without_eps_key_is_passthrough() -> None:
    params = {"n_clusters": 8}
    assert (
        _prepare_params(
            _stub_recipe("knee"), np.zeros((10, 32), dtype=np.float32), params
        )
        == params
    )


@pytest.mark.parametrize(
    ("n_jobs", "expected"),
    [
        (4, 4),
        (1, 1),
        (0, 1),
        ("abc", 1),
        (-1, os.cpu_count() or 1),
        (None, os.cpu_count() or 1),
    ],
)
def test_thread_limit_for(n_jobs, expected: int) -> None:
    assert _thread_limit_for(n_jobs) == expected


def test_with_thread_limits_is_transparent_to_wrapped_callable() -> None:
    # The runner applies this to every theirs callable; verify kwargs pass through and the
    # context manager doesn't change the return shape/dtype.
    from sklearn.cluster import DBSCAN

    def raw(data, *, eps, min_samples, n_jobs):
        return (
            DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
            .fit_predict(data)
            .astype(np.int32)
        )

    wrapped = _with_thread_limits(raw, n_jobs=2)
    data = np.random.default_rng(0).normal(size=(200, 2)).astype(np.float32) * 3.0
    labels = wrapped(data, eps=5.0, min_samples=3, n_jobs=2)
    assert labels.shape == (200,)
    assert labels.dtype == np.int32
