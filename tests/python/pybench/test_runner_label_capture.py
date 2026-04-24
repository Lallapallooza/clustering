from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from pybench.recipe import DatasetSpec, Recipe
from pybench.runner import (
    LabelsBundle,
    _project_2d,
    make_data_with_gt,
    run_one,
    run_one_with_labels,
)


def _stub_ours(data, **_: object) -> np.ndarray:
    n = data.shape[0]
    # Deterministic, non-trivial labels so ARI != 1.0 unless theirs agrees.
    return (np.arange(n) % 4).astype(np.int32)


def _stub_theirs(data, **_: object) -> np.ndarray:
    n = data.shape[0]
    # Identical assignment as ours so ARI == 1.0, which makes the ARI-equivalence
    # assertion robust without needing full reference agreement.
    return (np.arange(n) % 4).astype(np.int32)


def _stub_ours_differs(data, **_: object) -> np.ndarray:
    n = data.shape[0]
    return (np.arange(n) % 3).astype(np.int32)


def _make_recipe(*, ours=_stub_ours, theirs=_stub_theirs, n_runs: int = 2) -> Recipe:
    return Recipe(
        name="stub",
        ours=ours,
        theirs=theirs,
        default_params={"n_jobs": 1},
        default_sizes=(128,),
        default_dims=(2,),
        dataset=DatasetSpec(
            n_features=2,
            centers=4,
            cluster_std=1.0,
            random_state=0,
            vmf_switch_dim=1_000_000,
        ),
        ari_threshold=0.0,
        n_runs=n_runs,
    )


def test_make_data_with_gt_shape_blobs() -> None:
    spec = DatasetSpec(
        n_features=8, centers=5, random_state=0, vmf_switch_dim=16, cluster_std=1.0
    )
    X, gt = make_data_with_gt(300, spec)
    assert X.shape == (300, 8)
    assert X.dtype == np.float32
    assert gt.shape == (300,)
    assert gt.dtype == np.int32
    assert set(np.unique(gt).tolist()) == {0, 1, 2, 3, 4}


def test_make_data_with_gt_shape_vmf() -> None:
    spec = DatasetSpec(
        n_features=64, centers=6, random_state=0, vmf_switch_dim=16, vmf_kappa=50.0
    )
    X, gt = make_data_with_gt(300, spec)
    assert X.shape == (300, 64)
    assert X.dtype == np.float32
    assert gt.shape == (300,)
    assert gt.dtype == np.int32
    # vMF path assigns exactly `centers` distinct labels.
    assert set(np.unique(gt).tolist()) == {0, 1, 2, 3, 4, 5}


def test_make_data_matches_make_data_with_gt_X() -> None:
    """The thin wrapper must not change the @c X bytes; otherwise call sites
    that layered on @c make_data_with_gt would measure a different dataset."""
    from pybench.runner import make_data

    spec = DatasetSpec(
        n_features=4, centers=3, random_state=7, vmf_switch_dim=16, cluster_std=2.0
    )
    X_gt, _ = make_data_with_gt(200, spec)
    X_plain = make_data(200, spec)
    assert np.array_equal(X_gt, X_plain)


def test_project_2d_passthrough_when_d_is_2() -> None:
    spec = DatasetSpec(random_state=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2)).astype(np.float32)
    proj = _project_2d(X, 2, spec)
    assert proj.shape == (100, 2)
    assert proj.dtype == np.float32
    assert np.array_equal(proj, X)


def test_project_2d_pca_when_d_leq_15() -> None:
    from sklearn.decomposition import PCA

    spec = DatasetSpec(random_state=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    proj = _project_2d(X, 8, spec)
    assert proj.shape == (200, 2)
    assert proj.dtype == np.float32
    expected = PCA(
        n_components=2, svd_solver="randomized", random_state=0
    ).fit_transform(X)
    assert np.allclose(proj, expected.astype(np.float32), atol=1e-5)


def test_project_2d_svd_when_d_geq_16() -> None:
    from sklearn.decomposition import TruncatedSVD

    spec = DatasetSpec(random_state=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 32)).astype(np.float32)
    proj = _project_2d(X, 32, spec)
    assert proj.shape == (200, 2)
    assert proj.dtype == np.float32
    expected = TruncatedSVD(
        n_components=2, algorithm="randomized", random_state=0
    ).fit_transform(X)
    assert np.allclose(proj, expected.astype(np.float32), atol=1e-5)


def test_project_2d_deterministic_under_seed() -> None:
    spec = DatasetSpec(random_state=123)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 40)).astype(np.float32)
    p_a = _project_2d(X, 40, spec)
    p_b = _project_2d(X, 40, spec)
    assert np.array_equal(p_a, p_b)


def test_run_one_with_labels_returns_bundle() -> None:
    recipe = _make_recipe(n_runs=2)
    result, bundle = run_one_with_labels(recipe, size=64, dims=2, capture_labels=True)
    assert bundle is not None
    assert isinstance(bundle, LabelsBundle)
    assert bundle.gt_labels.shape == (64,)
    assert bundle.gt_labels.dtype == np.int32
    assert bundle.ours_labels.shape == (64,)
    assert bundle.ours_labels.dtype == np.int32
    assert bundle.theirs_labels is not None
    assert bundle.theirs_labels.shape == (64,)
    assert bundle.theirs_labels.dtype == np.int32
    assert bundle.projection_2d.shape == (64, 2)
    assert bundle.projection_2d.dtype == np.float32
    # Sanity: stubs agree, so ARI == 1.
    assert result.ari == pytest.approx(1.0)


def test_run_one_with_labels_ari_matches_run_result() -> None:
    """The captured @c ours/@c theirs labels must yield the same ARI as
    @c RunResult.ari -- downstream tooling relies on that equality."""
    recipe = _make_recipe(ours=_stub_ours_differs, theirs=_stub_theirs, n_runs=2)
    result, bundle = run_one_with_labels(recipe, size=120, dims=2, capture_labels=True)
    assert bundle is not None
    assert bundle.theirs_labels is not None
    captured_ari = float(adjusted_rand_score(bundle.theirs_labels, bundle.ours_labels))
    assert captured_ari == pytest.approx(result.ari, abs=1e-12)


def test_run_one_with_labels_ours_only_returns_theirs_none() -> None:
    """@c --ours-only mode must capture gt + ours + projection but leave
    @c theirs_labels as @c None."""
    recipe = _make_recipe()
    result, bundle = run_one_with_labels(
        recipe, size=48, dims=2, ours_only=True, capture_labels=True
    )
    assert bundle is not None
    assert bundle.theirs_labels is None
    assert bundle.gt_labels.shape == (48,)
    assert bundle.ours_labels.shape == (48,)
    assert bundle.projection_2d.shape == (48, 2)
    # Sentinel for ours-only per existing runner contract.
    assert result.ari == pytest.approx(1.0)


def test_run_one_with_labels_returns_none_bundle_when_disabled() -> None:
    recipe = _make_recipe()
    result, bundle = run_one_with_labels(recipe, size=32, dims=2, capture_labels=False)
    assert bundle is None
    assert result.size == 32


def test_run_one_preserves_external_signature() -> None:
    """@c run_one must still accept @c (recipe, size, dims, params, ours_only)
    so existing callers (e.g. @c test_cli_smoke._stub_run_one) keep working."""
    recipe = _make_recipe()
    result = run_one(recipe, size=32, dims=2, params={"n_jobs": 1}, ours_only=False)
    assert result.size == 32
    assert result.dims == 2


def test_run_one_with_labels_high_dim_projection_shape() -> None:
    """At @c dims >= 16, the projection uses @c TruncatedSVD and still returns
    @c float32[n, 2]."""
    spec = DatasetSpec(
        n_features=32,
        centers=4,
        random_state=0,
        vmf_switch_dim=1_000_000,
        cluster_std=1.0,
    )
    recipe = Recipe(
        name="stub_hd",
        ours=_stub_ours,
        theirs=_stub_theirs,
        default_params={"n_jobs": 1},
        default_sizes=(64,),
        default_dims=(32,),
        dataset=spec,
        ari_threshold=0.0,
        n_runs=1,
    )
    _, bundle = run_one_with_labels(recipe, size=96, dims=32, capture_labels=True)
    assert bundle is not None
    assert bundle.projection_2d.shape == (96, 2)
    assert bundle.projection_2d.dtype == np.float32
