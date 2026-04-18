from __future__ import annotations

import itertools
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

import memray
import numpy as np
from kneed import KneeLocator
from scipy.stats import vonmises_fisher
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from pybench.recipe import DatasetSpec, EpsPolicy, Recipe, RunResult


def _make_blobs(n_samples: int, spec: DatasetSpec) -> np.ndarray:
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        center_box=spec.center_box,
        cluster_std=spec.cluster_std,
        random_state=spec.random_state,
    )
    return X.astype(np.float32)


def _make_vmf(n_samples: int, spec: DatasetSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.random_state)
    raw = rng.normal(size=(spec.centers, spec.n_features))
    mus = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    per = n_samples // spec.centers
    remainder = n_samples - per * spec.centers
    sizes = np.full(spec.centers, per, dtype=np.int64)
    sizes[:remainder] += 1
    out = np.empty((n_samples, spec.n_features), dtype=np.float32)
    offset = 0
    for mu, n_k in zip(mus, sizes, strict=True):
        dist = vonmises_fisher(mu=mu, kappa=spec.vmf_kappa, seed=rng)
        out[offset : offset + n_k] = dist.rvs(size=int(n_k)).astype(np.float32)
        offset += n_k
    return out


def make_data(n_samples: int, spec: DatasetSpec) -> np.ndarray:
    """Reproducible float32 dataset. Branches to vMF above vmf_switch_dim."""
    if spec.n_features > spec.vmf_switch_dim:
        return _make_vmf(n_samples, spec)
    return _make_blobs(n_samples, spec)


def _knee_eps(data: np.ndarray, k: int) -> float:
    """k-distance knee per Ester 1996 + Satopaa 2011 Kneedle."""
    nn = NearestNeighbors(n_neighbors=max(2, k))
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    k_dists = np.sort(distances[:, -1])
    finite = k_dists[np.isfinite(k_dists)]
    if finite.size < 3:
        return float(np.median(k_dists)) if k_dists.size else 1.0
    xs = np.arange(finite.size)
    knee = KneeLocator(xs, finite, curve="convex", direction="increasing")
    if knee.knee is None:
        return float(np.median(finite))
    return float(finite[int(knee.knee)])


def _resolve_eps(
    policy: EpsPolicy, base_eps: float, data: np.ndarray, min_samples: int
) -> float:
    if policy == "fixed":
        return base_eps
    if policy == "sqrt_d":
        d = max(1, data.shape[1])
        return base_eps * math.sqrt(d / 2.0)
    if policy == "knee":
        return _knee_eps(data, min_samples)
    raise ValueError(f"unknown eps_policy: {policy!r}")


def _prepare_params(
    recipe: Recipe, data: np.ndarray, params: dict[str, Any]
) -> dict[str, Any]:
    """Apply the recipe's eps policy. Returns a new dict; input untouched."""
    if "eps" not in params or recipe.eps_policy == "fixed":
        return dict(params)
    min_samples = int(params.get("min_samples", 5))
    eps_eff = _resolve_eps(recipe.eps_policy, float(params["eps"]), data, min_samples)
    return {**params, "eps": eps_eff}


def _thread_limit_for(n_jobs: Any) -> int:
    if n_jobs in (None, -1):
        return os.cpu_count() or 1
    try:
        requested = int(n_jobs)
    except (TypeError, ValueError):
        return 1
    return max(1, requested)


def _with_thread_limits(fn: Callable, n_jobs: Any) -> Callable:
    """Wrap @p fn so every call runs under @c threadpool_limits(limits=n_jobs).

    Applied to every @c theirs callable in @ref run_one so BLAS + OpenMP are
    scoped to match ours' clamped C++ pool. Without this scoping, sklearn's
    OpenMP inner loops grab every core regardless of @c n_jobs, making the
    per-cell ratio meaningless at small @c n_jobs.
    """
    limit = _thread_limit_for(n_jobs)

    def _wrapped(data: np.ndarray, **params: Any) -> np.ndarray:
        with threadpool_limits(limits=limit):
            return fn(data, **params)

    return _wrapped


def expand_param_grid(recipe: Recipe) -> list[dict[str, Any]]:
    """Expand default_params + param_grid into a list of param dicts."""
    if not recipe.param_grid:
        return [dict(recipe.default_params)]

    grid_keys = sorted(recipe.param_grid.keys())
    grid_values = [recipe.param_grid[k] for k in grid_keys]

    combos = []
    for values in itertools.product(*grid_values):
        params = dict(recipe.default_params)
        for k, v in zip(grid_keys, values):
            params[k] = v
        combos.append(params)
    return combos


def _measure_peak_rss_mb(
    fn: Callable, data: np.ndarray, params: dict[str, Any]
) -> float:
    """Capture peak bytes allocated during `fn(data, **params)`.

    memray intercepts libc allocations, so the number reflects both the
    Python side and C-extension / sklearn worker allocations. follow_fork
    lets it follow joblib's fork-based workers on Linux; on platforms
    where sklearn spawns via exec (macOS), only the parent is seen.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        capture = Path(f.name)
    capture.unlink()
    try:
        with memray.Tracker(
            capture,
            follow_fork=True,
            native_traces=False,
            trace_python_allocators=False,
        ):
            fn(data, **params)
        reader = memray.FileReader(capture)
        peak_bytes = reader.metadata.peak_memory
        return round(peak_bytes / (1024 * 1024), 2)
    finally:
        capture.unlink(missing_ok=True)


def run_one(
    recipe: Recipe,
    size: int,
    dims: int | None = None,
    params: dict[str, Any] | None = None,
) -> RunResult:
    """Run both implementations n_runs times, return median timing + ARI."""
    from dataclasses import replace

    if params is None:
        params = dict(recipe.default_params)

    dataset = recipe.dataset
    if dims is not None and dims != dataset.n_features:
        dataset = replace(dataset, n_features=dims)

    data = make_data(size, dataset)

    theirs_fn = _with_thread_limits(recipe.theirs, params.get("n_jobs"))

    effective = _prepare_params(recipe, data, params)

    ours_times: list[float] = []
    theirs_times: list[float] = []

    for _ in range(recipe.n_runs):
        t0 = time.perf_counter()
        result_ours = recipe.ours(data, **effective)
        ours_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result_theirs = theirs_fn(data, **effective)
        theirs_times.append(time.perf_counter() - t0)

    ours_labels = result_ours
    theirs_labels = result_theirs

    ours_peak_mb = _measure_peak_rss_mb(recipe.ours, data, effective)
    theirs_peak_mb = _measure_peak_rss_mb(theirs_fn, data, effective)

    ours_median_ms = median(ours_times) * 1000.0
    theirs_median_ms = median(theirs_times) * 1000.0
    speedup = theirs_median_ms / ours_median_ms if ours_median_ms > 0 else float("inf")

    ari = float(adjusted_rand_score(theirs_labels, ours_labels))
    ours_noise_frac = float(np.mean(ours_labels == -1))
    theirs_noise_frac = float(np.mean(theirs_labels == -1))

    return RunResult(
        recipe_name=recipe.name,
        size=size,
        dims=dataset.n_features,
        params=params,
        effective_params=effective,
        ours_median_ms=ours_median_ms,
        theirs_median_ms=theirs_median_ms,
        ours_peak_mb=round(ours_peak_mb, 2),
        theirs_peak_mb=round(theirs_peak_mb, 2),
        ari=ari,
        ours_noise_frac=ours_noise_frac,
        theirs_noise_frac=theirs_noise_frac,
        speedup=speedup,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_suite(
    recipes: dict[str, Recipe],
    sizes: list[int] | None = None,
    dims: list[int] | None = None,
) -> list[RunResult]:
    """Run all recipes across sizes x dims x param_grid, return results."""
    results: list[RunResult] = []
    for recipe in recipes.values():
        effective_sizes = sizes if sizes is not None else list(recipe.default_sizes)
        effective_dims = dims if dims is not None else list(recipe.default_dims)
        param_combos = expand_param_grid(recipe)

        for dim in effective_dims:
            for size in effective_sizes:
                for params in param_combos:
                    result = run_one(recipe, size, dims=dim, params=params)
                    results.append(result)
    return results
