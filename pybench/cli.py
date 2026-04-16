from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pybench.charts.results_io import capture_metadata, save_results
from pybench.recipes import all_recipes
from pybench.runner import expand_param_grid, run_one


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


def main() -> None:
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
        type=int,
        default=None,
        dest="n_runs",
        help="Override n_runs for all recipes",
    )

    args = parser.parse_args()

    recipes = all_recipes()

    if args.list:
        _list_recipes(recipes)
        return

    if not recipes:
        print("No recipes found. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.algo:
        missing = [a for a in args.algo if a not in recipes]
        if missing:
            print(f"Unknown algorithm(s): {', '.join(missing)}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(recipes.keys()))}", file=sys.stderr)
            sys.exit(1)
        recipes = {k: recipes[k] for k in args.algo}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name, recipe in sorted(recipes.items()):
        if args.n_runs is not None:
            from dataclasses import replace

            recipe = replace(recipe, n_runs=args.n_runs)

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
                    # Format varying params for display
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

    json_path = out_dir / "results.json"
    meta = capture_metadata()
    save_results(json_path, meta, all_results)
    print(f"\nResults saved to {json_path}")

    from pybench.plot import plot_results

    plot_results(all_results, out_dir)
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
