from __future__ import annotations

import numpy as np
from _clustering import hdbscan as cpp_hdbscan
from sklearn.cluster import HDBSCAN

from pybench.recipe import DatasetSpec, Recipe


def _ours(
    data: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 0,
    method: str = "eom",
    n_jobs: int = -1,
    **_: object,
) -> np.ndarray:
    labels, _scores, _nclusters = cpp_hdbscan(
        data,
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        method=method,
        n_jobs=n_jobs,
    )
    return labels


def _theirs(
    data: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 0,
    method: str = "eom",
    n_jobs: int = -1,
    **_: object,
) -> np.ndarray:
    resolved_min_samples = min_samples if min_samples > 0 else None
    return (
        HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=resolved_min_samples,
            cluster_selection_method=method,
            n_jobs=n_jobs,
            copy=True,
        )
        .fit_predict(data)
        .astype(np.int32)
    )


recipe = Recipe(
    name="hdbscan",
    ours=_ours,
    theirs=_theirs,
    default_params={"min_cluster_size": 5, "min_samples": 0, "method": "eom"},
    param_grid={"n_jobs": [1, 4, 16]},
    default_sizes=(5000, 25000, 100000),
    default_dims=(2, 8, 32, 64, 128),
    dataset=DatasetSpec(n_features=2, centers=20, cluster_std=3.0),
    ari_threshold=0.80,
    n_runs=3,
    tags=("density", "hierarchical"),
)
