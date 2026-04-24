from __future__ import annotations

import itertools
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

import memray
import numpy as np
from kneed import KneeLocator
from scipy.stats import vonmises_fisher
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from pybench.alignment import as_aligned
from pybench.recipe import DatasetSpec, EpsPolicy, Recipe, RunResult


def _make_blobs(n_samples: int, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        center_box=spec.center_box,
        cluster_std=spec.cluster_std,
        random_state=spec.random_state,
    )
    return X.astype(np.float32), y.astype(np.int32)


def _make_vmf(n_samples: int, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.random_state)
    raw = rng.normal(size=(spec.centers, spec.n_features))
    mus = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    per = n_samples // spec.centers
    remainder = n_samples - per * spec.centers
    sizes = np.full(spec.centers, per, dtype=np.int64)
    sizes[:remainder] += 1
    out = np.empty((n_samples, spec.n_features), dtype=np.float32)
    gt = np.empty(n_samples, dtype=np.int32)
    offset = 0
    for k, (mu, n_k) in enumerate(zip(mus, sizes, strict=True)):
        dist = vonmises_fisher(mu=mu, kappa=spec.vmf_kappa, seed=rng)
        out[offset : offset + n_k] = dist.rvs(size=int(n_k)).astype(np.float32)
        gt[offset : offset + n_k] = k
        offset += n_k
    return out, gt


def make_data_with_gt(
    n_samples: int, spec: DatasetSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible 32-byte aligned float32 dataset plus @c int32 ground-truth labels.

    Branches to vMF above @c spec.vmf_switch_dim. The label array is what
    ``make_blobs`` already produces internally or the per-component index the
    vMF offset loop assigns; see :func:`make_data` for the alignment rationale.
    """
    X, gt = (
        _make_vmf(n_samples, spec)
        if spec.n_features > spec.vmf_switch_dim
        else _make_blobs(n_samples, spec)
    )
    return as_aligned(X), gt


def make_data(n_samples: int, spec: DatasetSpec) -> np.ndarray:
    """Reproducible 32-byte aligned float32 dataset. Branches to vMF above vmf_switch_dim.

    numpy's default allocator lands on 16-byte boundaries non-deterministically for contiguous
    arrays, which routes the C++ binding down its memcpy fallback on half of cases and makes
    memray peak numbers compare unevenly between @c ours and @c theirs (the copy lands inside
    the tracker scope on one run and outside on another). Aligning at the data source pins both
    implementations to the same input layout so the recorded peak reflects implementation
    footprint, not allocator luck.
    """
    X, _ = make_data_with_gt(n_samples, spec)
    return X


# Above this dim, we skip full-eigenvector PCA in favor of randomized TruncatedSVD which
# avoids materializing the full d x d covariance / gram path. Tuned from the runtime-budget
# spike: at d=1024, n=250k, PCA(svd_solver='randomized') already drives projection cost well
# under the AC ceiling, so we only cross over when d clearly makes a full PCA layout wasteful.
_PROJ_PCA_MAX_DIMS = 15


def _project_2d(X: np.ndarray, dims: int, spec: DatasetSpec) -> np.ndarray:
    """Return a @c float32[n, 2] 2-D projection of @p X, seeded from the dataset.

    - @c dims == 2: passthrough @c X[:, :2].
    - @c 3 <= dims <= 15: :class:`sklearn.decomposition.PCA` with
      @c svd_solver='randomized'.
    - @c dims >= 16: :class:`sklearn.decomposition.TruncatedSVD` with
      @c algorithm='randomized'.

    The chart-render side only needs a visualization; the projection
    estimator's @c random_state is @c spec.random_state so two calls on the
    same inputs produce byte-identical output.
    """
    if dims == 2:
        return np.ascontiguousarray(X[:, :2], dtype=np.float32)
    if dims <= _PROJ_PCA_MAX_DIMS:
        reducer: Any = PCA(
            n_components=2,
            svd_solver="randomized",
            random_state=spec.random_state,
        )
    else:
        reducer = TruncatedSVD(
            n_components=2,
            algorithm="randomized",
            random_state=spec.random_state,
        )
    # PCA/TruncatedSVD both default to float64 internally; cast back to match
    # sidecar contract (projection_2d must be float32).
    return np.ascontiguousarray(reducer.fit_transform(X), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class LabelsBundle:
    """Captured labels + projection for a single @c run_one_with_labels call.

    @c theirs_labels is @c None when the call ran under @c ours_only.
    """

    gt_labels: np.ndarray
    ours_labels: np.ndarray
    projection_2d: np.ndarray
    theirs_labels: np.ndarray | None = None


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


def run_one_with_labels(
    recipe: Recipe,
    size: int,
    dims: int | None = None,
    params: dict[str, Any] | None = None,
    ours_only: bool = False,
    capture_labels: bool = True,
) -> tuple[RunResult, LabelsBundle | None]:
    """Run both implementations @c n_runs times and optionally capture labels.

    Behaves like :func:`run_one` with one addition: when @p capture_labels is
    @c True, the FIRST timed invocation of each implementation's labels are
    snapshotted into a :class:`LabelsBundle` and returned alongside the
    :class:`RunResult`. The labels returned are byte-identical to the ones used
    to compute the returned @c ari (both come from the first iteration of the
    timed loop, not the last).

    Under @p ours_only: @c theirs_labels is @c None in the bundle; the
    projection is still captured.

    Returns @c (result, None) when @p capture_labels is @c False.
    """
    from dataclasses import replace

    if params is None:
        params = dict(recipe.default_params)

    dataset = recipe.dataset
    if dims is not None:
        if recipe.dataset_for_dim is not None:
            dataset = recipe.dataset_for_dim(dims)
        elif dims != dataset.n_features:
            dataset = replace(dataset, n_features=dims)

    if capture_labels:
        data, gt_labels = make_data_with_gt(size, dataset)
    else:
        data = make_data(size, dataset)
        gt_labels = None

    effective = _prepare_params(recipe, data, params)

    ours_times: list[float] = []
    theirs_times: list[float] = []

    theirs_fn = (
        _with_thread_limits(recipe.theirs, params.get("n_jobs"))
        if not ours_only
        else None
    )

    # Labels captured from the FIRST timed invocation of each side. Holding on
    # to the first-iteration arrays keeps @c RunResult.ari and the
    # @c LabelsBundle byte-identical: both are computed from the same labels,
    # so a @c pybench vis caption derived from @c ari can never disagree with
    # the labels it draws. For deterministic algorithms (every recipe shipped
    # today) the first and last iterations coincide anyway, so the choice
    # between first and last has no observable effect on @c payload_hash6.
    first_ours_labels: np.ndarray | None = None
    first_theirs_labels: np.ndarray | None = None

    for i in range(recipe.n_runs):
        t0 = time.perf_counter()
        result_ours = recipe.ours(data, **effective)
        ours_times.append(time.perf_counter() - t0)
        if i == 0:
            first_ours_labels = result_ours

        if theirs_fn is not None:
            t0 = time.perf_counter()
            result_theirs = theirs_fn(data, **effective)
            theirs_times.append(time.perf_counter() - t0)
            if i == 0:
                first_theirs_labels = result_theirs

    assert first_ours_labels is not None
    ours_labels = first_ours_labels
    ours_median_ms = median(ours_times) * 1000.0
    ours_noise_frac = float(np.mean(ours_labels == -1))

    if ours_only:
        ours_peak_mb = 0.0
        theirs_peak_mb = 0.0
        theirs_median_ms = 0.0
        # 0.0 rather than NaN so the canonical JSON serializer accepts the row; callers that
        # care about the comparison must not pass --ours-only.
        speedup = 0.0
        ari = 1.0
        theirs_noise_frac = 0.0
    else:
        assert first_theirs_labels is not None
        theirs_labels = first_theirs_labels
        ours_peak_mb = _measure_peak_rss_mb(recipe.ours, data, effective)
        theirs_peak_mb = _measure_peak_rss_mb(theirs_fn, data, effective)
        theirs_median_ms = median(theirs_times) * 1000.0
        speedup = (
            theirs_median_ms / ours_median_ms if ours_median_ms > 0 else float("inf")
        )
        ari = float(adjusted_rand_score(theirs_labels, ours_labels))
        theirs_noise_frac = float(np.mean(theirs_labels == -1))

    run_result = RunResult(
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

    if not capture_labels:
        return run_result, None

    assert gt_labels is not None
    projection = _project_2d(data, dataset.n_features, dataset)
    bundle = LabelsBundle(
        gt_labels=gt_labels,
        ours_labels=ours_labels.astype(np.int32, copy=False),
        theirs_labels=(
            first_theirs_labels.astype(np.int32, copy=False)
            if first_theirs_labels is not None
            else None
        ),
        projection_2d=projection,
    )
    return run_result, bundle


def run_one(
    recipe: Recipe,
    size: int,
    dims: int | None = None,
    params: dict[str, Any] | None = None,
    ours_only: bool = False,
) -> RunResult:
    """Run both implementations n_runs times, return median timing + ARI.

    When @p ours_only is @c True, skip the @c theirs (sklearn) run, ARI computation, and the
    memray peak-RSS tracker for both sides; @c theirs_median_ms / @c theirs_peak_mb collapse
    to @c 0.0, @c speedup to @c nan, @c ari to @c 1.0 (so the gate stays green when the CLI
    chooses not to enforce it). Intended for CPU-perf iteration on large shapes where the
    baseline alone is minutes per cell.
    """
    result, _ = run_one_with_labels(
        recipe,
        size,
        dims=dims,
        params=params,
        ours_only=ours_only,
        capture_labels=False,
    )
    return result


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
