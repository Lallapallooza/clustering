"""Generate a reproducible 2D DBSCAN fixture matching the pybench recipe.

Produces a CSV and prints the eps chosen by pybench's knee policy against the fixture. The
output mirrors DatasetSpec(n_features=2, centers=20, cluster_std=3.0) and the runner's
eps policy so the numbers line up with what pybench reports.

Usage:
    uv run python tools/make_dbscan_fixture.py --n 100000 --out /tmp/dbscan_2d_100k.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

# pybench's eps knee derivation uses min_samples to compute k-dist and kneed's KneeLocator.
from pybench.runner import _knee_eps  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--centers", type=int, default=20)
    parser.add_argument("--std", type=float, default=3.0)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-box", type=float, nargs=2, default=(-150.0, 150.0))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x, _ = make_blobs(
        n_samples=args.n,
        n_features=args.d,
        centers=args.centers,
        cluster_std=args.std,
        center_box=tuple(args.center_box),
        random_state=args.seed,
    )
    x = x.astype(np.float32)

    eps = _knee_eps(x, args.min_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, x, delimiter=",", fmt="%.6f")
    sys.stderr.write(
        f"wrote {args.out} shape={x.shape} eps={eps:.6f} min_samples={args.min_samples}\n"
    )
    print(eps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
