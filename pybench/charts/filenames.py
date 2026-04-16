from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pybench.charts.data import slug_from_params


def chart_filename(algo: str, params: Mapping[str, Any], hash6: str) -> str:
    slug = slug_from_params(params)
    if slug:
        return f"{algo}_{slug}_{hash6}.png"
    return f"{algo}_{hash6}.png"
