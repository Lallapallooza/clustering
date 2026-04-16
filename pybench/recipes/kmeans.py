from __future__ import annotations

import numpy as np
import threadpoolctl
from _clustering import kmeans as cpp_kmeans
from sklearn.cluster import KMeans

from pybench.recipe import DatasetSpec, Recipe


def _ours(data: np.ndarray, *, n_clusters: int, n_jobs: int = 1) -> np.ndarray:
    with threadpoolctl.threadpool_limits(1, user_api="blas"):
        labels, _, _, _, _ = cpp_kmeans(
            data, k=n_clusters, max_iter=300, tol=1e-4, seed=0, n_jobs=n_jobs
        )
    return labels


def _theirs(data: np.ndarray, *, n_clusters: int, n_jobs: int = 1) -> np.ndarray:
    with threadpoolctl.threadpool_limits(1, user_api="blas"):
        model = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            max_iter=300,
            tol=1e-4,
            random_state=0,
        )
        return model.fit_predict(data).astype(np.int32)


recipe = Recipe(
    name="kmeans",
    ours=_ours,
    theirs=_theirs,
    default_params={"n_clusters": 16},
    param_grid={"n_jobs": [1]},
    default_sizes=(1000, 5000, 10000, 50000, 100000),
    default_dims=(2, 4, 8, 32),
    dataset=DatasetSpec(n_features=2, centers=16, cluster_std=3.0),
    ari_threshold=0.85,
    n_runs=3,
    tags=("centroid", "kmeans"),
)
