from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import os
import tempfile
import time
import traceback
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
    arrays, which routes the C++ binding down its memcpy fallback on half of cases. Aligning at
    the data source pins @c ours onto its zero-copy borrow path so the recorded peak allocation
    reflects implementation footprint, not allocator luck.
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

    Applied to the @c theirs callable inside its worker subprocess so BLAS,
    OpenMP, and oneDAL honor @p n_jobs. Without this scoping, those inner loops
    grab every core regardless of @c n_jobs, making the per-cell ratio
    meaningless at small @c n_jobs.
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


def _measure_peak_alloc_mb(
    fn: Callable, data: np.ndarray, params: dict[str, Any]
) -> float:
    """Peak MB allocated during one `fn(data, **params)` call, via memray.

    memray intercepts this process's libc allocations, so the figure captures
    the fit's own footprint plus any input copy the binding makes, isolated from
    the interpreter and import baseline. follow_fork keeps joblib's fork workers
    in scope on Linux.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
        capture = Path(handle.name)
    capture.unlink()
    try:
        with memray.Tracker(
            capture,
            follow_fork=True,
            native_traces=False,
            trace_python_allocators=False,
        ):
            fn(data, **params)
        return memray.FileReader(capture).metadata.peak_memory / (1024 * 1024)
    finally:
        capture.unlink(missing_ok=True)


def _engine_worker(
    conn: Any,
    engine: str,
    baseline: str,
    recipe_name: str,
    data_path: str,
    params: dict[str, Any],
    n_runs: int,
) -> None:
    """Time one engine in an isolated @c spawn child and ship the result back.

    Each engine runs in its own process so neither thread pool (ours' citor
    pool, theirs' oneDAL / OpenMP pool) leaks into the other's measurement. The
    child first restores affinity to every CPU, undoing any pin a parent run
    inherited, then applies the engine's own thread control. @c theirs is
    wrapped in @ref _with_thread_limits so it honors @c n_jobs; @c ours reads
    @c n_jobs straight from @p params and clamps its own pool.

    For @c intelex the patch must precede the recipe import (recipes import
    sklearn at module load), so it runs before :func:`all_recipes`. The per-fit
    peak allocation comes from one extra memray-tracked pass after the timed
    loop, kept out of the timed runs so tracking overhead never pollutes timing.
    """
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))

        if engine == "theirs" and baseline == "intelex":
            from sklearnex import patch_sklearn

            patch_sklearn()

        from pybench.recipes import all_recipes

        recipe = all_recipes()[recipe_name]
        data = as_aligned(np.load(data_path))
        if engine == "ours":
            fn: Callable = recipe.ours
        else:
            fn = _with_thread_limits(recipe.theirs, params.get("n_jobs"))

        times: list[float] = []
        first_labels: np.ndarray | None = None
        for i in range(n_runs):
            t0 = time.perf_counter()
            labels = fn(data, **params)
            times.append(time.perf_counter() - t0)
            if i == 0:
                first_labels = np.asarray(labels)

        # One untimed pass under memray for the per-fit peak allocation. memray
        # tracks this child's own libc allocations, so the number isolates the
        # fit's footprint (ours' borrow vs theirs' copy) from the interpreter
        # baseline that whole-process RSS would otherwise swamp.
        peak_mb = _measure_peak_alloc_mb(fn, data, params)
        conn.send(("ok", times, first_labels, peak_mb))
    except Exception:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _run_engine_subprocess(
    engine: str,
    baseline: str,
    recipe_name: str,
    data_path: str,
    params: dict[str, Any],
    n_runs: int,
) -> tuple[list[float], np.ndarray, float]:
    """Spawn @ref _engine_worker, block for its result, and return it.

    Raises @c RuntimeError carrying the child's traceback when the worker fails
    so a recipe error surfaces in the parent instead of a bare pipe EOF.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_engine_worker,
        args=(child_conn, engine, baseline, recipe_name, data_path, params, n_runs),
    )
    proc.start()
    child_conn.close()
    try:
        message = parent_conn.recv()
    except EOFError:
        message = ("error", "worker exited before sending a result")
    parent_conn.close()
    proc.join()
    if message[0] == "error":
        raise RuntimeError(
            f"{engine} worker failed (exit code {proc.exitcode}):\n{message[1]}"
        )
    _, times, labels, peak_mb = message
    return times, labels, peak_mb


def run_one_with_labels(
    recipe: Recipe,
    size: int,
    dims: int | None = None,
    params: dict[str, Any] | None = None,
    ours_only: bool = False,
    capture_labels: bool = True,
    baseline: str = "sklearn",
) -> tuple[RunResult, LabelsBundle | None]:
    """Run both implementations @c n_runs times and optionally capture labels.

    Each engine is timed in its own @c spawn subprocess via
    :func:`_run_engine_subprocess` so the two thread pools never contend or
    interfere: ours' citor pool would otherwise defeat @c theirs' thread limits
    and pin the main thread to one CPU. The parent generates the dataset and
    resolves @c eps once, writes the data to a temporary @c .npy, and both
    children read it so they see byte-identical inputs.

    When @p capture_labels is @c True, the FIRST timed invocation of each
    implementation's labels are snapshotted into a :class:`LabelsBundle` and
    returned alongside the :class:`RunResult`. The labels returned are
    byte-identical to the ones used to compute the returned @c ari.

    @p baseline selects the library backing @c theirs (@c "sklearn" or
    @c "intelex"); it is applied inside the @c theirs child, not the parent.

    Under @p ours_only: @c theirs_labels is @c None in the bundle; the
    projection is still captured. Returns @c (result, None) when @p
    capture_labels is @c False.
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

    # The two engines run in separate processes, so both read the same dataset
    # off disk rather than sharing the parent's array. Labels captured from the
    # FIRST timed invocation of each side keep @c RunResult.ari and the
    # @c LabelsBundle byte-identical: both come from the same arrays, so a
    # @c pybench vis caption derived from @c ari can never disagree with the
    # labels it draws.
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
        data_path = handle.name
    try:
        np.save(data_path, data)
        ours_times, ours_labels, ours_peak_mb = _run_engine_subprocess(
            "ours", baseline, recipe.name, data_path, effective, recipe.n_runs
        )
        first_theirs_labels: np.ndarray | None = None
        if not ours_only:
            theirs_times, first_theirs_labels, theirs_peak_mb = _run_engine_subprocess(
                "theirs", baseline, recipe.name, data_path, effective, recipe.n_runs
            )
    finally:
        Path(data_path).unlink(missing_ok=True)

    ours_median_ms = median(ours_times) * 1000.0
    ours_noise_frac = float(np.mean(ours_labels == -1))

    if ours_only:
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
    baseline: str = "sklearn",
) -> RunResult:
    """Run both implementations n_runs times, return median timing + ARI.

    When @p ours_only is @c True, skip the @c theirs run and the ARI
    computation; @c theirs_median_ms and @c theirs_peak_mb collapse to @c 0.0,
    @c speedup to @c 0.0, @c ari to @c 1.0 (so the gate stays green when the CLI
    chooses not to enforce it). Intended for CPU-perf iteration on large shapes
    where the baseline alone is minutes per cell.
    """
    result, _ = run_one_with_labels(
        recipe,
        size,
        dims=dims,
        params=params,
        ours_only=ours_only,
        capture_labels=False,
        baseline=baseline,
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
