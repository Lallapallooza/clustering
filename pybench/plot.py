from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pybench.recipe import RunResult


def _get_n_jobs(r: RunResult) -> int:
    return r.params.get("n_jobs", -1)


def plot_results(results: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (algo, dims)
    groups: dict[tuple[str, int], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.recipe_name, r.dims), []).append(r)

    for (algo, dims), group in groups.items():
        _plot_size_scaling(algo, dims, group, out_dir)
        _plot_thread_scaling(algo, dims, group, out_dir)
        _plot_memory(algo, dims, group, out_dir)
        _plot_ari(algo, dims, group, out_dir)


def _plot_size_scaling(
    algo: str, dims: int, results: list[RunResult], out_dir: Path
) -> None:
    """Performance vs dataset size, one line per thread count."""
    n_jobs_values = sorted(set(_get_n_jobs(r) for r in results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for nj in n_jobs_values:
        subset = sorted(
            [r for r in results if _get_n_jobs(r) == nj],
            key=lambda r: r.size,
        )
        if not subset:
            continue
        xs = [r.size for r in subset]
        label = f"n_jobs={nj}" if nj > 0 else "n_jobs=all"

        ax1.plot(xs, [r.ours_median_ms for r in subset], "o-", label=f"C++ {label}")
    # sklearn (single line — doesn't vary with our n_jobs)
    sklearn_by_size = {}
    for r in results:
        if r.size not in sklearn_by_size:
            sklearn_by_size[r.size] = r.theirs_median_ms
    sk_sizes = sorted(sklearn_by_size.keys())
    ax1.plot(
        sk_sizes,
        [sklearn_by_size[s] for s in sk_sizes],
        "s--",
        color="gray",
        label="sklearn",
    )

    ax1.set_xlabel("Dataset size")
    ax1.set_ylabel("Median time (ms)")
    ax1.set_title(f"{algo} {dims}D — time vs size")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Speedup vs size
    for nj in n_jobs_values:
        subset = sorted(
            [r for r in results if _get_n_jobs(r) == nj],
            key=lambda r: r.size,
        )
        if not subset:
            continue
        label = f"n_jobs={nj}" if nj > 0 else "n_jobs=all"
        ax2.plot(
            [r.size for r in subset],
            [r.speedup for r in subset],
            "o-",
            label=label,
        )
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Dataset size")
    ax2.set_ylabel("Speedup (sklearn / C++)")
    ax2.set_title(f"{algo} {dims}D — speedup vs size")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{algo}_{dims}d_size_scaling.png", dpi=120)
    plt.close(fig)


def _plot_thread_scaling(
    algo: str, dims: int, results: list[RunResult], out_dir: Path
) -> None:
    """Performance vs thread count, one line per dataset size."""
    sizes = sorted(set(r.size for r in results))
    n_jobs_values = sorted(set(_get_n_jobs(r) for r in results))

    if len(n_jobs_values) <= 1:
        return  # nothing to plot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for size in sizes:
        subset = sorted(
            [r for r in results if r.size == size],
            key=lambda r: _get_n_jobs(r),
        )
        if not subset:
            continue
        xs = [_get_n_jobs(r) for r in subset]
        x_labels = [str(x) if x > 0 else "all" for x in xs]

        ax1.plot(
            range(len(xs)), [r.ours_median_ms for r in subset], "o-", label=f"n={size}"
        )
        ax1.set_xticks(range(len(xs)))
        ax1.set_xticklabels(x_labels)

        ax2.plot(range(len(xs)), [r.speedup for r in subset], "o-", label=f"n={size}")
        ax2.set_xticks(range(len(xs)))
        ax2.set_xticklabels(x_labels)

    ax1.set_xlabel("Threads")
    ax1.set_ylabel("C++ median time (ms)")
    ax1.set_title(f"{algo} {dims}D — thread scaling")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Speedup (sklearn / C++)")
    ax2.set_title(f"{algo} {dims}D — speedup vs threads")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{algo}_{dims}d_thread_scaling.png", dpi=120)
    plt.close(fig)


def _plot_memory(algo: str, dims: int, results: list[RunResult], out_dir: Path) -> None:
    """Peak memory vs dataset size, C++ vs sklearn."""
    n_jobs_values = sorted(set(_get_n_jobs(r) for r in results))

    # Memory doesn't vary much with n_jobs for the same algo, pick n_jobs=1 for ours
    # and show sklearn once
    fig, ax = plt.subplots(figsize=(8, 5))

    # C++ at a single thread count (memory is dominated by data, not threads)
    nj = n_jobs_values[0]
    subset = sorted(
        [r for r in results if _get_n_jobs(r) == nj],
        key=lambda r: r.size,
    )
    ax.plot(
        [r.size for r in subset],
        [r.ours_peak_mb for r in subset],
        "o-",
        label="C++ (ours)",
        color="steelblue",
    )
    ax.plot(
        [r.size for r in subset],
        [r.theirs_peak_mb for r in subset],
        "s-",
        label="sklearn",
        color="coral",
    )

    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"{algo} {dims}D — memory usage")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{algo}_{dims}d_memory.png", dpi=120)
    plt.close(fig)


def _plot_ari(algo: str, dims: int, results: list[RunResult], out_dir: Path) -> None:
    """ARI heatmap-style: sizes x param combos."""
    sizes = sorted(set(r.size for r in results))
    n_jobs_values = sorted(set(_get_n_jobs(r) for r in results))
    ari_threshold = 0.85

    if len(n_jobs_values) <= 1:
        # Simple bar chart
        subset = sorted(results, key=lambda r: r.size)
        x = np.arange(len(sizes))
        aris = [next(r.ari for r in subset if r.size == s) for s in sizes]
        colors = ["green" if a >= ari_threshold else "red" for a in aris]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, aris, color=colors)
        ax.axhline(
            ari_threshold,
            color="gray",
            linestyle="--",
            label=f"threshold={ari_threshold}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Dataset size")
        ax.set_ylabel("ARI")
        ax.set_title(f"{algo} {dims}D — correctness")
        ax.legend()
    else:
        # Grouped bars: one group per size, one bar per n_jobs
        x = np.arange(len(sizes))
        width = 0.8 / len(n_jobs_values)

        fig, ax = plt.subplots(figsize=(max(10, len(sizes) * 2), 5))
        for i, nj in enumerate(n_jobs_values):
            aris = []
            for s in sizes:
                r = next(
                    (r for r in results if r.size == s and _get_n_jobs(r) == nj), None
                )
                aris.append(r.ari if r else 0)
            label = f"n_jobs={nj}" if nj > 0 else "n_jobs=all"
            colors = ["green" if a >= ari_threshold else "red" for a in aris]
            ax.bar(x + i * width - 0.4 + width / 2, aris, width, label=label, alpha=0.8)

        ax.axhline(
            ari_threshold,
            color="gray",
            linestyle="--",
            label=f"threshold={ari_threshold}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Dataset size")
        ax.set_ylabel("ARI")
        ax.set_title(f"{algo} {dims}D — correctness across threads")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{algo}_{dims}d_ari.png", dpi=120)
    plt.close(fig)
