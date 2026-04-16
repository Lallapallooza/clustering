from __future__ import annotations

import math

import numpy as np

from pybench.recipe import DatasetSpec, Recipe
from pybench.runner import (
    _blas_limit_for,
    _knee_eps,
    _prepare_params,
    _resolve_eps,
    make_data,
    make_sklearn_reference,
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


def test_make_data_is_deterministic() -> None:
    spec = DatasetSpec(n_features=32, centers=5, random_state=7)
    a = make_data(500, spec)
    b = make_data(500, spec)
    assert np.array_equal(a, b)


def test_resolve_eps_fixed_returns_base() -> None:
    data = np.zeros((10, 8), dtype=np.float32)
    assert _resolve_eps("fixed", 1.5, data, min_samples=5) == 1.5


def test_resolve_eps_sqrt_d_scales_with_dim() -> None:
    base = 10.0
    data_2d = np.zeros((10, 2), dtype=np.float32)
    data_128d = np.zeros((10, 128), dtype=np.float32)
    eps_2 = _resolve_eps("sqrt_d", base, data_2d, min_samples=5)
    eps_128 = _resolve_eps("sqrt_d", base, data_128d, min_samples=5)
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
    recipe = Recipe(name="x", ours=lambda data, **kw: data, eps_policy="fixed")
    params = {"eps": 3.0, "min_samples": 5, "n_jobs": 1}
    data = np.zeros((10, 2), dtype=np.float32)
    effective = _prepare_params(recipe, data, params)
    assert effective == params
    assert effective is not params


def test_prepare_params_sqrt_d_rescales_eps() -> None:
    recipe = Recipe(name="x", ours=lambda data, **kw: data, eps_policy="sqrt_d")
    params = {"eps": 10.0, "min_samples": 5}
    data_128 = np.zeros((10, 128), dtype=np.float32)
    effective = _prepare_params(recipe, data_128, params)
    assert math.isclose(effective["eps"], 10.0 * math.sqrt(64), rel_tol=1e-6)
    assert params["eps"] == 10.0  # input untouched


def test_prepare_params_without_eps_key_is_passthrough() -> None:
    recipe = Recipe(name="x", ours=lambda data, **kw: data, eps_policy="knee")
    params = {"n_clusters": 8}
    data = np.zeros((10, 32), dtype=np.float32)
    assert _prepare_params(recipe, data, params) == params


def test_blas_limit_for_negative_uses_cpu_count() -> None:
    import os

    assert _blas_limit_for(-1) == (os.cpu_count() or 1)
    assert _blas_limit_for(None) == (os.cpu_count() or 1)


def test_blas_limit_for_positive() -> None:
    assert _blas_limit_for(4) == 4
    assert _blas_limit_for(1) == 1
    assert _blas_limit_for(0) == 1


def test_blas_limit_for_handles_garbage() -> None:
    assert _blas_limit_for("abc") == 1


def test_sklearn_reference_honors_blas_limit_in_context() -> None:
    # The reference callable must not crash when invoked with n_jobs; we only
    # verify it runs and returns integer labels of the right shape -- the
    # threadpool_limits context manager resets on exit.
    ref = make_sklearn_reference("dbscan")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 2)).astype(np.float32) * 3.0
    labels = ref(data, eps=5.0, min_samples=3, n_jobs=2)
    assert labels.shape == (200,)
    assert labels.dtype == np.int32
