from __future__ import annotations


import numpy as np
from _clustering import kmeans_best_of as cpp_kmeans_best_of
from sklearn.cluster import KMeans

from pybench.alignment import as_aligned
from pybench.recipe import DatasetSpec, Recipe


_N_INIT = 5


def _ours(data: np.ndarray, *, n_clusters: int, n_jobs: int = 1) -> np.ndarray:
    # One-shot align so the C++ binding takes its zero-copy borrow path rather than
    # the memcpy fallback. The n_init loop now lives inside the binding so a single
    # KMeans<float> instance, thread pool, and policy scratch amortize across restarts.
    data = as_aligned(data)
    labels, _, _, _, _ = cpp_kmeans_best_of(
        data,
        k=n_clusters,
        max_iter=300,
        tol=1e-4,
        seed_first=0,
        n_jobs=n_jobs,
        n_init=_N_INIT,
    )
    return labels


def _theirs(data: np.ndarray, *, n_clusters: int, **_: object) -> np.ndarray:
    # sklearn.KMeans dropped @c n_jobs in 1.0; parallelism is scoped via
    # threadpool_limits around the call, applied by the runner on every theirs_fn.
    return (
        KMeans(
            n_clusters=n_clusters,
            n_init=_N_INIT,
            max_iter=300,
            tol=1e-4,
            random_state=0,
        )
        .fit_predict(data)
        .astype(np.int32)
    )


recipe = Recipe(
    name="kmeans",
    ours=_ours,
    theirs=_theirs,
    default_params={"n_clusters": 16},
    param_grid={"n_jobs": [1, 4, 16]},
    default_sizes=(5000, 10000, 50000, 100000, 250000),
    default_dims=(2, 8, 32, 128, 256, 512, 1024),
    # KMeans' assumption is isotropic Euclidean Gaussians, so keep blobs at
    # every dim. vMF-on-sphere collapses all pairwise distances into a narrow
    # band, which is exactly the regime KMeans cannot resolve.
    dataset=DatasetSpec(
        n_features=2, centers=16, cluster_std=3.0, vmf_switch_dim=1_000_000
    ),
    ari_threshold=0.98,
    n_runs=3,
    tags=("centroid", "kmeans"),
)
