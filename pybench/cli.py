from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pybench.charts.data import (
    PARTITION_EXCLUDED_PARAM_KEYS,
    partition,
    payload_hash6,
)
from pybench.charts.figure import BuildFigureInputs, build_figure, is_ours_only_sentinel
from pybench.charts.filenames import chart_filename
from pybench.charts.gates import GateFailure, evaluate_gates
from pybench.charts.labels_io import (
    LabelCell,
    LoadStatus,
    labels_sidecar_filename_for_params,
    load_labels,
    save_labels,
)
from pybench.charts.meta import RunMetadata
from pybench.charts.results_io import capture_metadata, load_results, save_results
from pybench.charts.vis import (
    MultiDimVisInputs,
    VisInputs,
    build_multidim_vis_figure,
    build_vis_figure,
)
from pybench.recipe import Recipe, RunResult
from pybench.recipes import all_recipes
from pybench.runner import (
    LabelsBundle,
    expand_param_grid,
    run_one_with_labels,
)

logger = logging.getLogger(__name__)

_ANSI_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"

_SUBCOMMANDS = ("bench", "vis")
_VIS_SUFFIX = "vis"

# Ceiling on the number of rows a @c --all-dims vis figure will render. Past
# this count each row's panels shrink to the point where the scatter structure
# becomes illegible; extras are dropped with a warning so the reader knows
# the figure is not the whole partition.
_ALL_DIMS_MAX_ROWS = 10


def _list_recipes(recipes: dict) -> None:
    if not recipes:
        print("No recipes found.")
        return
    for name, recipe in sorted(recipes.items()):
        tags = ", ".join(recipe.tags) if recipe.tags else "-"
        sizes = ", ".join(str(s) for s in recipe.default_sizes)
        dims = ", ".join(str(d) for d in recipe.default_dims)
        params = ", ".join(f"{k}={v}" for k, v in recipe.default_params.items())
        grid = (
            ", ".join(f"{k}={v}" for k, v in recipe.param_grid.items())
            if recipe.param_grid
            else "-"
        )
        combos = (
            len(expand_param_grid(recipe))
            * len(recipe.default_sizes)
            * len(recipe.default_dims)
        )
        print(f"{name}")
        print(f"  tags       : {tags}")
        print(f"  params     : {params}")
        print(f"  param_grid : {grid}")
        print(f"  sizes      : {sizes}")
        print(f"  dims       : {dims}")
        print(f"  n_runs     : {recipe.n_runs}")
        print(f"  threshold  : {recipe.ari_threshold}")
        print(f"  total runs : {combos}")


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--algo", nargs="*", help="Algorithm(s) to benchmark (default: all)"
    )
    parser.add_argument("--sizes", nargs="+", type=int, help="Dataset sizes to use")
    parser.add_argument("--dims", nargs="+", type=int, help="Dimensionalities to use")
    parser.add_argument(
        "--list", action="store_true", help="List discovered recipes and exit"
    )
    parser.add_argument(
        "--out", type=str, default="benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--n-runs",
        "--n_runs",
        type=int,
        default=None,
        dest="n_runs",
        help="Override n_runs for all recipes",
    )
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        help="Path to an existing results.json; regenerate charts without benchmarking",
    )
    parser.add_argument(
        "--ours-only",
        action="store_true",
        help=(
            "CPU-perf-only fast path: time the @c ours implementation only, skip the @c theirs"
            " sklearn baseline, the ARI gate, and the memray peak-RSS tracker. Useful for"
            " tight perf iteration loops on large shapes where the baseline is minutes per"
            " cell."
        ),
    )
    parser.add_argument(
        "--no-labels",
        dest="capture_labels",
        action="store_false",
        help=(
            "Skip per-cell label capture and the per-partition @c .labels.npz sidecar that"
            " @c pybench vis consumes. Default is ON (labels captured) and adds a single"
            " @c int32[n] copy per implementation per cell plus one PCA/TruncatedSVD per"
            " cell to produce the 2-D projection."
        ),
    )
    parser.set_defaults(capture_labels=True)


def _add_vis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to a results.json produced by pybench bench",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory to write vis PNG(s) into",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Only render partitions whose recipe_name matches",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Only render the cell at this size (otherwise the largest size is picked)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help=(
            "Only render the cell at this dims (one row of 4 panels). Without"
            " this flag the default is 3 representative dims (low / mid / high)"
            " from the partition; pass @c --all-dims to render every dim."
        ),
    )
    parser.add_argument(
        "--all-dims",
        dest="all_dims",
        action="store_true",
        help=(
            "Render one row of 4 panels per dim in the partition (up to"
            f" {_ALL_DIMS_MAX_ROWS} rows; extras are dropped with a warning)."
            " Mutually exclusive with @c --dims."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        "--n_jobs",
        type=int,
        dest="n_jobs",
        default=None,
        help="Only render the cell whose params.n_jobs matches (otherwise the first)",
    )
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help=(
            "Error out when a sidecar is missing or mismatched rather than regenerating"
            " labels + projection by re-running the recipe."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark C++ clustering against sklearn.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run benchmarks, write results.json + charts + label sidecars",
    )
    _add_bench_args(bench_parser)
    vis_parser = subparsers.add_parser(
        "vis",
        help="Render fixture-label vis PNGs from an existing results.json + sidecar",
    )
    _add_vis_args(vis_parser)
    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """Compat shim: insert @c "bench" when the user supplied no subcommand.

    If the first positional (non-flag) token is an explicit member of
    :data:`_SUBCOMMANDS`, the invocation is already explicit. If the first
    token is a flag (starts with @c "-") or the list is empty, we default to
    @c "bench" so every pre-subparser invocation style (``benchmark --algo X``,
    ``benchmark --replot PATH``, ``benchmark --list``, ``benchmark --help``)
    still reaches the bench subparser unchanged. Any other first positional
    token -- for example ``benchmark diff`` or ``benchmark vis-all`` -- is
    treated as an unknown subcommand and rejected with a clear error rather
    than silently routed to @c bench.
    """
    if not argv:
        return ["bench"]
    first = argv[0]
    if first in _SUBCOMMANDS:
        return list(argv)
    if first.startswith("-"):
        return ["bench", *argv]
    raise SystemExit(
        f"benchmark: invalid subcommand {first!r} (valid: {', '.join(_SUBCOMMANDS)})"
    )


def _validate_replot_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.replot is None:
        return
    conflicts = []
    if args.algo:
        conflicts.append("--algo")
    if args.sizes is not None:
        conflicts.append("--sizes")
    if args.dims is not None:
        conflicts.append("--dims")
    if args.n_runs is not None:
        conflicts.append("--n-runs")
    if conflicts:
        parser.error(
            f"--replot cannot be combined with measurement flag(s): {', '.join(conflicts)}"
        )


def _format_failure(failure: GateFailure) -> str:
    reasons = "; ".join(failure.reasons)
    return (
        f"  {failure.recipe_name} size={failure.size} dims={failure.dims} "
        f"params={dict(failure.params)}: {reasons}"
    )


def _print_fail_banner(failures: Sequence[GateFailure]) -> None:
    use_color = sys.stderr.isatty()
    prefix = _ANSI_RED if use_color else ""
    suffix = _ANSI_RESET if use_color else ""
    print(
        f"{prefix}FAIL: {len(failures)} gate failure(s):{suffix}",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"{prefix}{_format_failure(failure)}{suffix}", file=sys.stderr)


def _dataset_spec_string(recipe: Recipe) -> str:
    return f"blobs centers={recipe.dataset.centers} std={recipe.dataset.cluster_std}"


def _non_njobs_params_tuple(
    params: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (k, v) for k, v in params.items() if k not in PARTITION_EXCLUDED_PARAM_KEYS
        )
    )


def _write_charts(
    partitions: dict,
    meta: RunMetadata,
    recipes: dict[str, Recipe],
    out_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    for key, partition_results in partitions.items():
        algo = key[0]
        non_njobs_params = tuple(key[1])
        dims_sorted = tuple(sorted({r.dims for r in partition_results}))
        recipe = recipes.get(algo)
        if recipe is not None:
            dataset_spec = _dataset_spec_string(recipe)
            ari_threshold: float | None = recipe.ari_threshold
        else:
            dataset_spec = ""
            ari_threshold = None

        inputs = BuildFigureInputs(
            algo=algo,
            partition_results=tuple(partition_results),
            dims_sorted=dims_sorted,
            title_meta=meta,
            non_njobs_params=non_njobs_params,
            dataset_spec=dataset_spec,
            ari_threshold=ari_threshold,
        )
        fig = build_figure(inputs)
        hash6 = payload_hash6(partition_results)
        filename = chart_filename(algo, dict(key[1]), hash6)
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=200)
        written.append(out_path)
    return written


def _labels_cell_key(result: RunResult) -> str:
    """Build a deterministic cell key for a sidecar @c LabelCell entry.

    Shape is @c "n={size}__d={dims}__jobs={n_jobs}"; when @c n_jobs is missing
    from @c params, the @c jobs= segment is dropped. The key is only consumed
    by @c pybench vis when rendering per-cell fixture views.
    """
    base = f"n={result.size}__d={result.dims}"
    if "n_jobs" in result.params:
        return f"{base}__jobs={result.params['n_jobs']}"
    return base


def _write_label_sidecars(
    results: list[RunResult],
    bundles: list[LabelsBundle],
    out_dir: Path,
) -> list[Path]:
    """Group @p results + @p bundles by partition and write one sidecar each.

    The sidecar filename shares the chart's @c hash6-prefix so both artifacts
    sit next to each other on disk. Returns the list of written sidecar paths.
    """
    if len(results) != len(bundles):
        raise ValueError(
            f"results and bundles length mismatch: {len(results)} vs {len(bundles)}"
        )

    # Group into {(algo, non_njobs_params_tuple): (results_for_partition,
    # bundles_for_partition)} so we can compute the per-partition hash6 that
    # matches the chart filename.
    groups: dict[
        tuple[str, tuple[tuple[str, Any], ...]],
        tuple[list[RunResult], list[LabelsBundle]],
    ] = {}

    for result, bundle in zip(results, bundles, strict=True):
        key = (result.recipe_name, _non_njobs_params_tuple(result.params))
        r_list, b_list = groups.setdefault(key, ([], []))
        r_list.append(result)
        b_list.append(bundle)

    written: list[Path] = []
    for (algo, _), (part_results, part_bundles) in groups.items():
        hash6 = payload_hash6(part_results)
        # Build the sidecar filename using the first result's params so the
        # chart and sidecar slugs stay aligned.
        sidecar_name = labels_sidecar_filename_for_params(
            algo, part_results[0].params, hash6
        )
        sidecar_path = out_dir / sidecar_name
        cells: dict[str, LabelCell] = {}
        for result, bundle in zip(part_results, part_bundles, strict=True):
            cells[_labels_cell_key(result)] = LabelCell(
                gt_labels=bundle.gt_labels,
                ours_labels=bundle.ours_labels,
                theirs_labels=bundle.theirs_labels,
                projection_2d=bundle.projection_2d,
            )
        save_labels(sidecar_path, hash6, algo, cells)
        written.append(sidecar_path)
    return written


def _run_bench(args: argparse.Namespace, recipes: dict[str, Recipe]) -> int:
    if args.list:
        _list_recipes(recipes)
        return 0

    if not recipes:
        print("No recipes found. Exiting.", file=sys.stderr)
        return 1

    if args.replot is not None:
        return _run_replot(args, recipes)
    return _run_live(args, recipes)


def _run_live(args: argparse.Namespace, recipes: dict[str, Recipe]) -> int:
    if args.algo:
        missing = [a for a in args.algo if a not in recipes]
        if missing:
            print(f"Unknown algorithm(s): {', '.join(missing)}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(recipes.keys()))}", file=sys.stderr)
            return 1
        recipes = {k: recipes[k] for k in args.algo}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = capture_metadata()
    all_results: list[RunResult] = []
    all_bundles: list[LabelsBundle] = []
    recipes_run: dict[str, Recipe] = {}

    for name, recipe in sorted(recipes.items()):
        if args.n_runs is not None:
            from dataclasses import replace

            recipe = replace(recipe, n_runs=args.n_runs)
        recipes_run[name] = recipe

        effective_sizes = (
            args.sizes if args.sizes is not None else list(recipe.default_sizes)
        )
        effective_dims = (
            args.dims if args.dims is not None else list(recipe.default_dims)
        )
        param_combos = expand_param_grid(recipe)

        for dim in effective_dims:
            for size in effective_sizes:
                for params in param_combos:
                    grid_info = " ".join(f"{k}={params[k]}" for k in recipe.param_grid)
                    print(
                        f"[{name}] {dim}D size={size:>7} {grid_info} ...",
                        end=" ",
                        flush=True,
                    )
                    result, bundle = run_one_with_labels(
                        recipe,
                        size,
                        dims=dim,
                        params=params,
                        ours_only=args.ours_only,
                        capture_labels=args.capture_labels,
                    )
                    eps_info = ""
                    if "eps" in result.effective_params:
                        eps_info = f" eps={result.effective_params['eps']:.2f}"
                    if args.ours_only:
                        print(
                            f"ours={result.ours_median_ms:.1f}ms"
                            f"  mem: {result.ours_peak_mb:.1f}MB"
                            f"{eps_info}"
                        )
                    else:
                        status = (
                            "PASS" if result.ari >= recipe.ari_threshold else "FAIL"
                        )
                        print(
                            f"ari={result.ari:.2f} ({status})"
                            f"  {result.speedup:.1f}x"
                            f"  {result.ours_median_ms:.1f}ms vs {result.theirs_median_ms:.1f}ms"
                            f"  mem: {result.ours_peak_mb:.1f}MB vs {result.theirs_peak_mb:.1f}MB"
                            f"{eps_info}"
                        )
                    all_results.append(result)
                    if bundle is not None:
                        all_bundles.append(bundle)

    if not args.ours_only:
        ari_thresholds = {name: r.ari_threshold for name, r in recipes_run.items()}
        failures = evaluate_gates(all_results, ari_thresholds)
        if failures:
            _print_fail_banner(failures)
            return 2

    json_path = out_dir / "results.json"
    save_results(json_path, meta, all_results)
    print(f"\nResults saved to {json_path}")

    partitions = partition(all_results)
    written = _write_charts(partitions, meta, recipes_run, out_dir)
    print(f"Charts saved to {out_dir}/ ({len(written)} PNG(s))")

    if args.capture_labels and all_bundles:
        sidecars = _write_label_sidecars(all_results, all_bundles, out_dir)
        print(f"Labels saved to {out_dir}/ ({len(sidecars)} sidecar(s))")
    return 0


def _run_replot(args: argparse.Namespace, recipes: dict[str, Recipe]) -> int:
    json_path = Path(args.replot)
    meta, all_results = load_results(json_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    present_names = {r.recipe_name for r in all_results}
    missing = present_names - set(recipes.keys())
    if missing:
        logger.warning(
            "recipes not in registry; rendering with empty dataset_spec: %s",
            sorted(missing),
        )

    partitions = partition(all_results)
    written = _write_charts(partitions, meta, recipes, out_dir)
    print(f"Charts saved to {out_dir}/ ({len(written)} PNG(s))")
    return 0


def _pick_representative_dims(
    available_dims: Sequence[int],
) -> list[int]:
    """Pick the low / mid / high dim from a sorted-unique sequence.

    The caller passes a deduplicated ascending list; we return up to 3 dims
    in ascending order. 1 dim -> ``[d0]``; 2 dims -> ``[d0, d1]``; 3+ dims
    -> ``[low, dims[len // 2], high]`` with duplicates collapsed so a 4-dim
    partition still returns exactly 3 rows.
    """
    dims = sorted({int(d) for d in available_dims})
    if len(dims) <= 2:
        return dims
    low = dims[0]
    mid = dims[len(dims) // 2]
    high = dims[-1]
    # Collapse accidental dupes (can happen only with len == 3 and even pick).
    picked: list[int] = []
    for d in (low, mid, high):
        if d not in picked:
            picked.append(d)
    return picked


def _pick_vis_cells(
    partition_results: list[RunResult],
    *,
    size_filter: int | None,
    dims_filter: int | None,
    n_jobs_filter: int | None,
    all_dims: bool,
) -> tuple[list[RunResult], str]:
    """Pick one cell per rendered dim for a partition's vis figure.

    Returns @c (cells, selection_mode). @c cells is ordered by ascending
    dim; @c selection_mode is a short human-readable string that the figure
    header surfaces so the reader knows which dims were picked.

    Selection rules:

    - @p dims_filter supplied -> single row at that dim.
    - @p all_dims True -> every dim in the partition, clamped to the
      @c _ALL_DIMS_MAX_ROWS ceiling with a warning when rows are dropped.
    - Neither -> the low / mid / high picker across unique dims.

    For each rendered dim, the cell is the largest @c size at that dim,
    narrowed by @p size_filter and @p n_jobs_filter when supplied. Ties are
    broken by insertion order. Returns an empty list when no cell matches
    any rendered dim.
    """
    base = list(partition_results)
    if size_filter is not None:
        base = [r for r in base if r.size == size_filter]
    if n_jobs_filter is not None:
        base = [r for r in base if r.params.get("n_jobs") == n_jobs_filter]

    available_dims = sorted({r.dims for r in base})
    if not available_dims:
        return [], ""

    if dims_filter is not None:
        picked_dims = [dims_filter] if dims_filter in available_dims else []
        selection_mode = f"cell selection: single dim --dims {dims_filter}"
    elif all_dims:
        if len(available_dims) > _ALL_DIMS_MAX_ROWS:
            dropped = len(available_dims) - _ALL_DIMS_MAX_ROWS
            logger.warning(
                "vis --all-dims: partition has %d dims, rendering the first %d"
                " (dropping %d trailing dim(s))",
                len(available_dims),
                _ALL_DIMS_MAX_ROWS,
                dropped,
            )
            picked_dims = available_dims[:_ALL_DIMS_MAX_ROWS]
        else:
            picked_dims = available_dims
        selection_mode = f"cell selection: all {len(picked_dims)} dims (--all-dims)"
    else:
        picked_dims = _pick_representative_dims(available_dims)
        if len(picked_dims) >= 3:
            selection_mode = (
                f"cell selection: 3 representative dims"
                f" (low / mid / high: d={picked_dims[0]}, d={picked_dims[1]},"
                f" d={picked_dims[-1]})"
            )
        elif len(picked_dims) == 2:
            selection_mode = (
                f"cell selection: 2 dims (low / high: d={picked_dims[0]},"
                f" d={picked_dims[1]})"
            )
        else:
            selection_mode = f"cell selection: single dim (d={picked_dims[0]})"

    cells: list[RunResult] = []
    for dim in picked_dims:
        candidates = [r for r in base if r.dims == dim]
        if not candidates:
            continue
        candidates.sort(key=lambda r: (-r.size, r.dims))
        cells.append(candidates[0])
    return cells, selection_mode


def _subsample_seed_from_hash6(hash6: str) -> int:
    """Derive a deterministic 32-bit seed from the partition @c hash6.

    @c hash6 is already a deterministic fingerprint of the partition payload;
    we hash it once more and take the low 32 bits so the seed is independent
    of the 24 bits used by the chart filename. Two runs against the same
    partition produce byte-identical figures.
    """
    digest = hashlib.sha256(hash6.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _regen_cell(
    recipe: Recipe, result: RunResult, *, ours_only: bool
) -> tuple[LabelsBundle, RunResult]:
    """Re-run @p recipe for the @p result's cell to rebuild a @c LabelsBundle.

    Used by @c pybench vis when the sidecar is missing or its envelope does
    not match. @p ours_only is propagated from the original run so a cell that
    skipped the sklearn baseline is not silently re-measured: the regen will
    also skip @c theirs, matching what the user asked for at bench time. The
    caller detects this via :func:`is_ours_only_sentinel` on the stored cell.
    Returns both the fresh bundle and the fresh :class:`RunResult` (so the
    caller can use the regenerated @c ari instead of the stale one).
    """
    fresh_result, bundle = run_one_with_labels(
        recipe,
        result.size,
        dims=result.dims,
        params=dict(result.params),
        ours_only=ours_only,
        capture_labels=True,
    )
    assert bundle is not None
    return bundle, fresh_result


def _save_vis_figure(
    fig: Any,
    algo: str,
    non_njobs_params: tuple[tuple[str, Any], ...],
    hash6: str,
    out_dir: Path,
) -> Path:
    params_dict = dict(non_njobs_params)
    filename = chart_filename(algo, params_dict, hash6, suffix=_VIS_SUFFIX)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200)
    return out_path


def _vis_inputs_for_cell(
    *,
    algo: str,
    recipe: Recipe,
    result: RunResult,
    bundle: LabelsBundle,
    non_njobs_params: tuple[tuple[str, Any], ...],
    meta: RunMetadata,
    subsample_seed: int,
    bench_filename: str | None = None,
) -> VisInputs:
    return VisInputs(
        algo=algo,
        recipe_name=recipe.name,
        size=result.size,
        dims=result.dims,
        non_njobs_params=non_njobs_params,
        n_jobs=int(result.params.get("n_jobs", 1)),
        ari=float(result.ari),
        title_meta=meta,
        gt_labels=bundle.gt_labels,
        ours_labels=bundle.ours_labels,
        theirs_labels=bundle.theirs_labels,
        projection_2d=bundle.projection_2d,
        subsample_seed=subsample_seed,
        bench_filename=bench_filename,
    )


def _build_vis_for_cell(
    *,
    algo: str,
    recipe: Recipe,
    result: RunResult,
    bundle: LabelsBundle,
    non_njobs_params: tuple[tuple[str, Any], ...],
    meta: RunMetadata,
    subsample_seed: int,
    bench_filename: str | None = None,
) -> Any:
    return build_vis_figure(
        _vis_inputs_for_cell(
            algo=algo,
            recipe=recipe,
            result=result,
            bundle=bundle,
            non_njobs_params=non_njobs_params,
            meta=meta,
            subsample_seed=subsample_seed,
            bench_filename=bench_filename,
        )
    )


def _run_vis(args: argparse.Namespace, recipes: dict[str, Recipe]) -> int:
    json_path = Path(args.results)
    if not json_path.is_file():
        print(f"vis: results file not found: {json_path}", file=sys.stderr)
        return 1

    if args.all_dims and args.dims is not None:
        print(
            "vis: --all-dims and --dims are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    meta, all_results = load_results(json_path)
    sidecar_dir = json_path.parent

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recipe is not None:
        filtered = [r for r in all_results if r.recipe_name == args.recipe]
        if not filtered:
            print(
                f"vis: no results matched --recipe {args.recipe!r}",
                file=sys.stderr,
            )
            return 1
        all_results = filtered

    partitions = partition(all_results)
    if not partitions:
        print("vis: no partitions to render", file=sys.stderr)
        return 1

    written: list[Path] = []
    for key, part_results in partitions.items():
        algo, non_njobs_params = key[0], tuple(key[1])
        recipe = recipes.get(algo)
        if recipe is None:
            print(
                f"vis: unknown recipe {algo!r}; cannot render, skipping",
                file=sys.stderr,
            )
            continue

        hash6 = payload_hash6(part_results)
        subsample_seed = _subsample_seed_from_hash6(hash6)

        sidecar_name = labels_sidecar_filename_for_params(
            algo, part_results[0].params, hash6
        )
        sidecar_path = sidecar_dir / sidecar_name
        load_result = load_labels(
            sidecar_path,
            expected_hash6=hash6,
            expected_recipe_name=algo,
        )

        cells, selection_mode = _pick_vis_cells(
            part_results,
            size_filter=args.size,
            dims_filter=args.dims,
            n_jobs_filter=args.n_jobs,
            all_dims=args.all_dims,
        )
        if not cells:
            print(
                f"vis: no cell matched filters for partition {algo!r} "
                f"params={dict(non_njobs_params)}",
                file=sys.stderr,
            )
            continue

        bench_fn = chart_filename(algo, dict(non_njobs_params), hash6)
        rows: list[VisInputs] = []
        regen_failed = False
        for cell in cells:
            resolved = _resolve_bundle_for_cell(
                load_result=load_result,
                sidecar_path=sidecar_path,
                recipe=recipe,
                cell=cell,
                no_regen=args.no_regen,
            )
            if resolved is None:
                regen_failed = True
                break
            bundle, authoritative_result = resolved
            # When regen fires we prefer the fresh @c RunResult so the figure's
            # caption reports the ari of the labels actually drawn; a sidecar
            # hit returns @p cell unchanged.
            rows.append(
                _vis_inputs_for_cell(
                    algo=algo,
                    recipe=recipe,
                    result=authoritative_result,
                    bundle=bundle,
                    non_njobs_params=non_njobs_params,
                    meta=meta,
                    subsample_seed=subsample_seed,
                    bench_filename=bench_fn,
                )
            )
        if regen_failed:
            return 1

        multi_inputs = MultiDimVisInputs(
            rows=tuple(rows), selection_mode=selection_mode
        )
        fig = build_multidim_vis_figure(multi_inputs)
        out_path = _save_vis_figure(fig, algo, non_njobs_params, hash6, out_dir)
        written.append(out_path)

    print(f"Vis PNGs saved to {out_dir}/ ({len(written)} PNG(s))")
    return 0 if written else 1


def _bundle_from_cell(cell_dict: LabelCell) -> LabelsBundle:
    return LabelsBundle(
        gt_labels=cell_dict.gt_labels,
        ours_labels=cell_dict.ours_labels,
        theirs_labels=cell_dict.theirs_labels,
        projection_2d=cell_dict.projection_2d,
    )


def _resolve_bundle_for_cell(
    *,
    load_result: Any,
    sidecar_path: Path,
    recipe: Recipe,
    cell: RunResult,
    no_regen: bool,
) -> tuple[LabelsBundle, RunResult] | None:
    """Resolve a :class:`LabelsBundle` and the authoritative :class:`RunResult`.

    The returned :class:`RunResult` is @p cell when the sidecar was consumed
    as-is, and the regenerated @c RunResult when a fresh re-run produced the
    bundle; callers should use it for anything that risks drifting between
    bench time and vis time (notably @c ari in the figure caption).

    @c OK / @c WARN with the cell present -> return the sidecar cell bundle
    paired with @p cell (emitting a warning on @c WARN). @c ERROR or cell
    missing -> regen fallback unless @p no_regen is set, in which case we
    fail with a clear error and return @c None.

    The @c --ours-only sentinel on @p cell is honored: regen propagates
    @c ours_only=True so a cell the user explicitly skipped the sklearn
    baseline on is not silently re-measured. The resulting bundle's
    @c theirs_labels is @c None, matching the original run's semantics.
    """
    cell_key = _labels_cell_key(cell)
    cell_is_sentinel = is_ours_only_sentinel(cell)

    if load_result.status == LoadStatus.OK:
        sidecar_cell = load_result.cells.get(cell_key)
        if sidecar_cell is None:
            if no_regen:
                print(
                    f"vis: sidecar present but missing cell {cell_key!r}; "
                    f"--no-regen was set. Path: {sidecar_path}",
                    file=sys.stderr,
                )
                return None
            logger.warning(
                "vis: sidecar %s is OK but missing cell %r; regenerating",
                sidecar_path,
                cell_key,
            )
            bundle, fresh = _regen_cell(recipe, cell, ours_only=cell_is_sentinel)
            return bundle, fresh
        return _bundle_from_cell(sidecar_cell), cell

    if load_result.status == LoadStatus.WARN:
        logger.warning(
            "vis: sidecar %s has a soft mismatch (%s); proceeding with cached arrays",
            sidecar_path,
            load_result.reason,
        )
        sidecar_cell = load_result.cells.get(cell_key)
        if sidecar_cell is not None:
            return _bundle_from_cell(sidecar_cell), cell
        if no_regen:
            print(
                f"vis: sidecar WARN and missing cell {cell_key!r}; --no-regen was set. "
                f"Path: {sidecar_path}",
                file=sys.stderr,
            )
            return None
        bundle, fresh = _regen_cell(recipe, cell, ours_only=cell_is_sentinel)
        return bundle, fresh

    # status == ERROR
    if no_regen:
        print(
            f"vis: sidecar error for {sidecar_path}: {load_result.reason}; "
            "--no-regen was set. Re-run bench to produce a fresh sidecar, "
            "or drop --no-regen to regenerate in place.",
            file=sys.stderr,
        )
        return None
    logger.warning(
        "vis: sidecar %s unavailable (%s); regenerating cell %r",
        sidecar_path,
        load_result.reason,
        cell_key,
    )
    bundle, fresh = _regen_cell(recipe, cell, ours_only=cell_is_sentinel)
    return bundle, fresh


def main() -> None:
    argv = sys.argv[1:]
    argv = _normalize_argv(argv)

    parser = _build_parser()
    args = parser.parse_args(argv)

    recipes = all_recipes()

    if args.subcommand == "bench":
        _validate_replot_args(parser, args)
        rc = _run_bench(args, recipes)
    elif args.subcommand == "vis":
        rc = _run_vis(args, recipes)
    else:
        parser.error(f"unknown subcommand {args.subcommand!r}")
        return  # argparse.error exits, but for type narrowing.

    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
