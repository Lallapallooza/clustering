from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    n_features: int = 2
    centers: int = 20
    center_box: tuple[float, float] = (-150.0, 150.0)
    cluster_std: float = 3.0
    random_state: int = 42


@dataclass(frozen=True)
class Recipe:
    name: str
    ours: Callable[..., np.ndarray]
    theirs: Callable[..., np.ndarray] | None = None
    default_params: dict[str, Any] = field(default_factory=dict)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    default_sizes: tuple[int, ...] = (1000, 5000, 10000, 50000)
    default_dims: tuple[int, ...] = (2,)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    ari_threshold: float = 0.85
    n_runs: int = 5
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunResult:
    recipe_name: str
    size: int
    dims: int
    params: dict[str, Any]
    ours_median_ms: float
    theirs_median_ms: float
    ours_peak_mb: float
    theirs_peak_mb: float
    ari: float
    ours_noise_frac: float
    theirs_noise_frac: float
    speedup: float
    timestamp: str
