from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    n_features: int = 2
    centers: int = 20
    center_box: tuple[float, float] = (-150.0, 150.0)
    cluster_std: float = 3.0
    random_state: int = 42
    # Above this dim, switch from isotropic Gaussian blobs to a vMF mixture on
    # the unit sphere. Real embeddings (SBERT, CLIP, etc.) are unit-normalized
    # with intrinsic dim << ambient dim, and isotropic blobs at high ambient
    # dim degenerate to ring-shell noise where DBSCAN with fixed eps finds no
    # clusters. Switch threshold 16 matches sklearn's NearestNeighbors
    # auto-brute-force cut-off at n_features > 15.
    vmf_switch_dim: int = 16
    # vMF concentration on the sphere. kappa=20 is borderline at d=32 and
    # unclusterable at d>=64 (intra/inter separation -> 1.0). Recipes that
    # care about high-dim separation should scale kappa with dim; DBSCAN's
    # Kneedle-gated eps absorbs the weak signal.
    vmf_kappa: float = 20.0


EpsPolicy = Literal["fixed", "sqrt_d", "knee"]


@dataclass(frozen=True)
class Recipe:
    name: str
    ours: Callable[..., np.ndarray]
    theirs: Callable[..., np.ndarray]
    default_params: dict[str, Any] = field(default_factory=dict)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    default_sizes: tuple[int, ...] = (1000, 5000, 10000, 50000)
    default_dims: tuple[int, ...] = (2,)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    ari_threshold: float = 0.85
    n_runs: int = 5
    tags: tuple[str, ...] = ()
    # How to derive the actual eps passed to ours/theirs from default_params.
    # "fixed" uses default_params["eps"] verbatim; "sqrt_d" scales it by
    # sqrt(d / 2) so isotropic blob clusters stay connectable at high dim;
    # "knee" replaces it with the k-distance knee (Ester 1996 + Satopaa
    # 2011) computed on the generated fixture.
    eps_policy: EpsPolicy = "fixed"
    # Optional per-dim DatasetSpec override. When set, the runner calls this with the
    # target dim and uses the returned spec instead of only replacing `n_features`. Recipes
    # whose fixture parameters (e.g. vMF kappa) have to scale with dim to keep cluster
    # structure meaningful wire this in place of the default n_features-only shim.
    dataset_for_dim: Callable[[int], DatasetSpec] | None = None


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
    # Params actually passed to ours/theirs after eps policy + data generation
    # (e.g. computed knee eps). Equal to `params` for fixed-eps recipes.
    # Optional on legacy JSONs, which hydrate with an empty dict.
    effective_params: dict[str, Any] = field(default_factory=dict)
