from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pybench.charts.data import partition, payload_hash6
from pybench.charts.figure import BuildFigureInputs, build_figure
from pybench.charts.filenames import chart_filename
from pybench.charts.gates import GateFailure, evaluate_gates
from pybench.charts.meta import RunMetadata
from pybench.charts.results_io import capture_metadata, load_results, save_results
from pybench.recipe import Recipe, RunResult
from pybench.recipes import all_recipes
from pybench.runner import expand_param_grid, run_one

logger = logging.getLogger(__name__)

_ANSI_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark C++ clustering against sklearn.",
    )
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
    return parser


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
        fig.savefig(out_path, dpi=120)
        written.append(out_path)
    return written


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
                    result = run_one(recipe, size, dims=dim, params=params)
                    status = "PASS" if result.ari >= recipe.ari_threshold else "FAIL"
                    print(
                        f"ari={result.ari:.2f} ({status})"
                        f"  {result.speedup:.1f}x"
                        f"  {result.ours_median_ms:.1f}ms vs {result.theirs_median_ms:.1f}ms"
                        f"  mem: {result.ours_peak_mb:.1f}MB vs {result.theirs_peak_mb:.1f}MB"
                    )
                    all_results.append(result)

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_replot_args(parser, args)

    recipes = all_recipes()

    if args.list:
        _list_recipes(recipes)
        return

    if not recipes:
        print("No recipes found. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.replot is not None:
        rc = _run_replot(args, recipes)
    else:
        rc = _run_live(args, recipes)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
