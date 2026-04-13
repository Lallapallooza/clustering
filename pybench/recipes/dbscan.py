from __future__ import annotations

import numpy as np
from _clustering import dbscan as cpp_dbscan

from pybench.recipe import DatasetSpec, Recipe


def _ours(
    data: np.ndarray, *, eps: float, min_samples: int, n_jobs: int = -1
) -> np.ndarray:
    return cpp_dbscan(data, eps=eps, min_pts=min_samples, n_jobs=n_jobs)


recipe = Recipe(
    name="dbscan",
    ours=_ours,
    theirs=None,
    default_params={"eps": 10.0, "min_samples": 5},
    param_grid={"n_jobs": [1, 2, 4, 8, 16, 32]},
    default_sizes=(1000, 5000, 10000, 50000, 100000),
    default_dims=(2, 8),
    dataset=DatasetSpec(n_features=2, centers=20, cluster_std=3.0),
    ari_threshold=0.85,
    n_runs=3,
    tags=("density", "spatial"),
)
