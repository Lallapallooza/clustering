from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pybench.charts.data import slug_from_params


def chart_filename(
    algo: str,
    params: Mapping[str, Any],
    hash6: str,
    *,
    suffix: str = "",
) -> str:
    """Build the PNG filename for a partition chart.

    The default ``suffix=""`` preserves the byte-exact filename every current
    caller relies on: ``{algo}_{slug}_{hash6}.png`` or ``{algo}_{hash6}.png``
    when @p params contributes no slug.

    Passing a non-empty @p suffix injects it between the slug and the hash:
    ``{algo}_{slug}_{suffix}_{hash6}.png``; ``{algo}_{suffix}_{hash6}.png``
    when the slug is empty. Used by @c pybench vis to write vis PNGs that do
    not collide with the corresponding @c bench chart (``suffix="vis"``).
    """
    slug = slug_from_params(params)
    if slug and suffix:
        return f"{algo}_{slug}_{suffix}_{hash6}.png"
    if slug:
        return f"{algo}_{slug}_{hash6}.png"
    if suffix:
        return f"{algo}_{suffix}_{hash6}.png"
    return f"{algo}_{hash6}.png"
