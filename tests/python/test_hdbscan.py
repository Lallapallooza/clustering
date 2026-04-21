"""Python tests for the nanobind HDBSCAN binding.

Run with:

    uv run pytest tests/python/test_hdbscan.py

Or directly:

    uv run python tests/python/test_hdbscan.py

The script imports the compiled _clustering extension. Exits with status 0
when every assertion passes and raises otherwise.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.cluster import HDBSCAN as SkHDBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

import _clustering as m


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _make_two_moons(n: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    X, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    return np.ascontiguousarray(X, dtype=np.float32)


def test_shape_and_dtype_contract() -> None:
    X = _make_two_moons(n=100, seed=1)
    labels, scores, n_clusters = m.hdbscan(X, min_cluster_size=5)
    _assert(labels.shape == (100,), f"labels shape: expected (100,) got {labels.shape}")
    _assert(
        labels.dtype == np.int32, f"labels dtype: expected int32 got {labels.dtype}"
    )
    _assert(
        scores.shape == (100,),
        f"outlier_scores shape: expected (100,) got {scores.shape}",
    )
    _assert(
        scores.dtype == np.float32,
        f"outlier_scores dtype: expected float32 got {scores.dtype}",
    )
    _assert(
        isinstance(n_clusters, int),
        f"n_clusters type: expected int got {type(n_clusters)}",
    )
    _assert(n_clusters >= 0, f"n_clusters must be >= 0, got {n_clusters}")


def test_two_moon_ari_matches_sklearn() -> None:
    # Two-moon ARI: the binding's labels agree with sklearn's HDBSCAN on a well-separated
    # synthetic dataset within ARI >= 0.99.
    X = _make_two_moons(n=300, noise=0.05, seed=0)
    labels_ours, _, _ = m.hdbscan(X, min_cluster_size=5)
    labels_sk = SkHDBSCAN(min_cluster_size=5, algorithm="brute", copy=True).fit_predict(
        X
    )
    ari = adjusted_rand_score(labels_ours, labels_sk)
    _assert(ari >= 0.99, f"two-moon ARI vs sklearn: {ari:.4f} (expected >= 0.99)")


def test_outlier_score_bounds() -> None:
    X = _make_two_moons(n=200, noise=0.08, seed=2)
    _, scores, _ = m.hdbscan(X, min_cluster_size=5)
    _assert(
        float(scores.min()) >= 0.0,
        f"outlier score min = {scores.min()} (must be >= 0)",
    )
    _assert(
        float(scores.max()) <= 1.0,
        f"outlier score max = {scores.max()} (must be <= 1)",
    )


def test_leaf_method_completes() -> None:
    # Verify method="leaf" is accepted and produces labels of the correct shape and dtype.
    X = _make_two_moons(n=150, seed=3)
    labels, scores, n_clusters = m.hdbscan(X, min_cluster_size=5, method="leaf")
    _assert(labels.shape == (150,), f"leaf labels shape: {labels.shape}")
    _assert(labels.dtype == np.int32, f"leaf labels dtype: {labels.dtype}")
    _assert(scores.shape == (150,), f"leaf scores shape: {scores.shape}")
    _assert(
        isinstance(n_clusters, int),
        f"leaf n_clusters type: {type(n_clusters)}",
    )


def test_min_samples_sentinel_and_explicit() -> None:
    # min_samples=0 (sentinel) and min_samples=5 should both produce valid results on
    # the same data without raising.
    X = _make_two_moons(n=200, seed=4)
    labels_default, _, n_default = m.hdbscan(X, min_cluster_size=5, min_samples=0)
    labels_explicit, _, n_explicit = m.hdbscan(X, min_cluster_size=5, min_samples=5)
    _assert(
        labels_default.shape == (200,),
        f"sentinel labels shape: {labels_default.shape}",
    )
    _assert(
        labels_explicit.shape == (200,),
        f"explicit labels shape: {labels_explicit.shape}",
    )
    _assert(
        n_default >= 0 and n_explicit >= 0,
        f"cluster counts must be non-negative: {n_default}, {n_explicit}",
    )


def main() -> int:
    tests = [
        test_shape_and_dtype_contract,
        test_two_moon_ari_matches_sklearn,
        test_outlier_score_bounds,
        test_leaf_method_completes,
        test_min_samples_sentinel_and_explicit,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {exc!r}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
