from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pybench.recipe import RunResult


@dataclass(frozen=True, slots=True)
class GateFailure:
    recipe_name: str
    size: int
    dims: int
    params: Mapping[str, Any]
    reasons: tuple[str, ...]


def evaluate_gates(
    results: Sequence[RunResult],
    ari_thresholds: Mapping[str, float],
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for r in results:
        reasons: list[str] = []

        if r.recipe_name in ari_thresholds:
            threshold = ari_thresholds[r.recipe_name]
            if not math.isfinite(r.ari):
                reasons.append(f"ari={r.ari!r} not finite (threshold {threshold})")
            elif r.ari < threshold:
                reasons.append(f"ari={r.ari!r} < threshold {threshold}")
        else:
            warnings.warn(
                f"no ARI threshold for recipe {r.recipe_name}; skipping ARI gate",
                UserWarning,
                stacklevel=2,
            )

        if not math.isfinite(r.ours_peak_mb):
            reasons.append(f"ours_peak_mb={r.ours_peak_mb!r} not finite")
        elif r.ours_peak_mb <= 0:
            reasons.append(f"ours_peak_mb={r.ours_peak_mb!r} <= 0")

        if not math.isfinite(r.theirs_peak_mb):
            reasons.append(f"theirs_peak_mb={r.theirs_peak_mb!r} not finite")
        elif r.theirs_peak_mb <= 0:
            reasons.append(f"theirs_peak_mb={r.theirs_peak_mb!r} <= 0")

        if reasons:
            failures.append(
                GateFailure(
                    recipe_name=r.recipe_name,
                    size=r.size,
                    dims=r.dims,
                    params=dict(r.params),
                    reasons=tuple(reasons),
                )
            )
    return failures


__all__ = ["GateFailure", "evaluate_gates"]
