from __future__ import annotations

import itertools
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

import memray
import numpy as np
import sklearn.cluster
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from pybench.recipe import DatasetSpec, Recipe, RunResult


def make_data(n_samples: int, spec: DatasetSpec) -> np.ndarray:
    """Generate reproducible float32 dataset from spec."""
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        center_box=spec.center_box,
        cluster_std=spec.cluster_std,
        random_state=spec.random_state,
    )
    return X.astype(np.float32)


def make_sklearn_reference(algo_name: str) -> Callable:
    """Auto-construct a sklearn reference callable by algorithm name."""
    class_name = (
        algo_name.upper()
        if algo_name.upper() in ("DBSCAN", "OPTICS")
        else algo_name.capitalize()
    )
    cls = getattr(sklearn.cluster, class_name, None)
    if cls is None:
        cls = getattr(sklearn.cluster, algo_name.upper(), None)
    if cls is None:
        raise ValueError(f"No sklearn.cluster class found for algorithm '{algo_name}'")

    def reference(data: np.ndarray, **params: Any) -> np.ndarray:
        model = cls(**params)
        labels = model.fit_predict(data)
        return labels.astype(np.int32)

    return reference


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

    theirs_fn = recipe.theirs
    if theirs_fn is None:
        theirs_fn = make_sklearn_reference(recipe.name)

    ours_times: list[float] = []
    theirs_times: list[float] = []

    for _ in range(recipe.n_runs):
        t0 = time.perf_counter()
        result_ours = recipe.ours(data, **params)
        ours_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result_theirs = theirs_fn(data, **params)
        theirs_times.append(time.perf_counter() - t0)

    ours_labels = result_ours
    theirs_labels = result_theirs

    # Measure peak RSS delta for one run of each (captures C++ allocations too)
    ours_peak_mb = _measure_peak_rss_mb(recipe.ours, data, params)
    theirs_peak_mb = _measure_peak_rss_mb(theirs_fn, data, params)

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
