"""Python tests for the nanobind kmeans binding.

Run with:

    uv run python tests/python/test_kmeans_binding.py

The script imports the compiled _clustering extension. It exits with status 0
when every assertion passes and raises otherwise.
"""

from __future__ import annotations

import sys

import numpy as np

import _clustering as m


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _make_blobs(n: int, d: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.zeros((k, d), dtype=np.float32)
    centers[:, 0] = np.arange(k, dtype=np.float32) * 50.0
    assignments = rng.integers(0, k, size=n)
    noise = rng.normal(scale=0.5, size=(n, d)).astype(np.float32)
    return centers[assignments] + noise


def test_shape_and_dtype_contract() -> None:
    X = _make_blobs(n=300, d=8, k=5, seed=1)
    labels, centroids, inertia, n_iter, converged = m.kmeans(
        X, k=5, max_iter=100, tol=1e-4, seed=42, n_jobs=1
    )
    _assert(labels.shape == (300,), f"labels shape: expected (300,) got {labels.shape}")
    _assert(
        labels.dtype == np.int32, f"labels dtype: expected int32 got {labels.dtype}"
    )
    _assert(
        centroids.shape == (5, 8),
        f"centroids shape: expected (5, 8) got {centroids.shape}",
    )
    _assert(
        centroids.dtype == np.float32,
        f"centroids dtype: expected float32 got {centroids.dtype}",
    )
    _assert(
        isinstance(inertia, float), f"inertia type: expected float got {type(inertia)}"
    )
    _assert(isinstance(n_iter, int), f"n_iter type: expected int got {type(n_iter)}")
    _assert(
        isinstance(converged, bool),
        f"converged type: expected bool got {type(converged)}",
    )


def test_labels_in_range() -> None:
    X = _make_blobs(n=400, d=4, k=6, seed=2)
    labels, _, _, _, _ = m.kmeans(X, k=6, max_iter=100, tol=1e-4, seed=7, n_jobs=1)
    _assert(int(labels.min()) >= 0, f"labels.min() = {labels.min()} (must be >= 0)")
    _assert(int(labels.max()) < 6, f"labels.max() = {labels.max()} (must be < k=6)")


def test_determinism_same_seed_bit_identical() -> None:
    # Same seed + n_jobs + data must produce bit-identical labels and centroids across
    # repeated calls.
    X = _make_blobs(n=500, d=12, k=8, seed=3)
    labels1, centroids1, inertia1, _, _ = m.kmeans(
        X, k=8, max_iter=100, tol=1e-4, seed=99, n_jobs=2
    )
    labels2, centroids2, inertia2, _, _ = m.kmeans(
        X, k=8, max_iter=100, tol=1e-4, seed=99, n_jobs=2
    )
    _assert(
        np.array_equal(labels1, labels2),
        "labels diverge across two same-seed calls",
    )
    # Bit-identical centroids -- compare raw bytes via tobytes, not float compare.
    _assert(
        centroids1.tobytes() == centroids2.tobytes(),
        "centroids differ bitwise across two same-seed calls",
    )
    _assert(
        inertia1.hex() == inertia2.hex(),
        f"inertia differs bitwise: {inertia1} vs {inertia2}",
    )


def test_converged_flag_roundtrip() -> None:
    # Well-separated blobs converge in well under max_iter; the binding must report
    # converged=True and an n_iter strictly below the cap.
    X = _make_blobs(n=500, d=4, k=5, seed=4)
    _, _, _, n_iter, converged = m.kmeans(
        X, k=5, max_iter=300, tol=1e-4, seed=11, n_jobs=1
    )
    _assert(converged is True, "expected convergence on well-separated blobs")
    _assert(n_iter > 0, f"n_iter must advance: got {n_iter}")
    _assert(n_iter < 300, f"n_iter must be below max_iter cap: got {n_iter}")


def test_high_k_completes_via_auto_dispatch() -> None:
    # The binding does not expose forceAlgorithm; auto-dispatch must produce a result for a
    # high-k input without error. Behavior, not dispatch choice, is what's verifiable here.
    X = _make_blobs(n=2000, d=8, k=64, seed=5)
    labels, centroids, _, _, _ = m.kmeans(
        X, k=64, max_iter=50, tol=1e-3, seed=13, n_jobs=2
    )
    _assert(labels.shape == (2000,), f"labels shape: {labels.shape}")
    _assert(centroids.shape == (64, 8), f"centroids shape: {centroids.shape}")
    _assert(int(labels.max()) < 64, f"labels.max() = {labels.max()} (must be < 64)")


def test_solver_reusable_after_first_kmeans_call() -> None:
    # Each binding invocation is independent: a second call on a different shape must not
    # leak state from the first.
    X1 = _make_blobs(n=200, d=4, k=3, seed=6)
    X2 = _make_blobs(n=300, d=8, k=5, seed=7)
    labels1, centroids1, _, _, _ = m.kmeans(X1, k=3, seed=21, n_jobs=1)
    labels2, centroids2, _, _, _ = m.kmeans(X2, k=5, seed=22, n_jobs=1)
    _assert(labels1.shape == (200,) and centroids1.shape == (3, 4), "first call shape")
    _assert(labels2.shape == (300,) and centroids2.shape == (5, 8), "second call shape")


def main() -> int:
    tests = [
        test_shape_and_dtype_contract,
        test_labels_in_range,
        test_determinism_same_seed_bit_identical,
        test_converged_flag_roundtrip,
        test_high_k_completes_via_auto_dispatch,
        test_solver_reusable_after_first_kmeans_call,
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
