from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from pybench.recipe import RunResult

PARTITION_EXCLUDED_PARAM_KEYS: frozenset[str] = frozenset({"n_jobs"})

_JsonScalar = (int, float, str, bool, type(None))

PartitionKey = tuple[str, tuple[tuple[str, Any], ...]]


def _validate_scalar_params(params: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if any param value is not a JSON scalar.

    Booleans are ``int`` subclasses in Python; that is accepted because
    booleans serialize cleanly in JSON.
    """
    for key, value in params.items():
        if not isinstance(value, _JsonScalar):
            raise ValueError(
                f"params[{key!r}] has non-scalar value of type "
                f"{type(value).__name__}: {value!r}. Chart partitioning "
                "requires scalar (int, float, str, bool, None) params."
            )


def partition_key(r: RunResult) -> PartitionKey:
    """Build the partition key for a result.

    Key is ``(recipe_name, sorted-tuple of (k, v) for non-excluded scalar
    params)``. ``n_jobs`` (and anything else in
    ``PARTITION_EXCLUDED_PARAM_KEYS``) is filtered so that results
    differing only by thread count land in the same partition.

    Raises ``ValueError`` if any params value is non-scalar.
    """
    _validate_scalar_params(r.params)
    filtered = tuple(
        sorted(
            (k, v)
            for k, v in r.params.items()
            if k not in PARTITION_EXCLUDED_PARAM_KEYS
        )
    )
    return (r.recipe_name, filtered)


def partition(
    results: Sequence[RunResult],
) -> dict[PartitionKey, list[RunResult]]:
    """Group results by ``partition_key``.

    Raises ``ValueError`` on:
    - A non-scalar params value (via ``partition_key``).
    - Two results sharing the same ``(recipe_name, size, dims, params)``
      tuple; silent deduplication would hide a double-run bug upstream.
    """
    seen: set[tuple[str, int, int, tuple[tuple[str, Any], ...]]] = set()
    groups: dict[PartitionKey, list[RunResult]] = {}
    for r in results:
        _validate_scalar_params(r.params)
        params_tuple = tuple(sorted(r.params.items()))
        dup_key = (r.recipe_name, r.size, r.dims, params_tuple)
        if dup_key in seen:
            raise ValueError(
                "duplicate result for "
                f"recipe_name={r.recipe_name!r}, size={r.size}, "
                f"dims={r.dims}, params={dict(params_tuple)!r}"
            )
        seen.add(dup_key)
        groups.setdefault(partition_key(r), []).append(r)
    return groups


def safe_ratio(num: float, den: float) -> float:
    """Return ``num / den``, or NaN if the inputs are unsafe.

    Returns NaN when ``den`` is non-positive or non-finite, or when
    ``num`` is non-finite. Never returns ``+inf``/``-inf`` and never
    emits a runtime warning.
    """
    if not math.isfinite(num):
        return math.nan
    if not math.isfinite(den) or den <= 0.0:
        return math.nan
    return num / den


CANONICAL_JSON_OPTS = {
    "sort_keys": True,
    "indent": 2,
    "ensure_ascii": True,
    "allow_nan": False,
}


def canonical_dumps(payload: Any, rows_for_error: list[dict[str, Any]] | None) -> str:
    """Dump `payload` with canonical options; on NaN/Inf, name the offender.

    `rows_for_error` is the sequence to scan for a nicer error -- pass the row
    list if the payload itself is or contains one; pass None to just re-raise.
    """
    try:
        return json.dumps(payload, **CANONICAL_JSON_OPTS)
    except ValueError as exc:
        if rows_for_error is None:
            raise
        offender = _find_non_finite(rows_for_error)
        if offender is None:
            raise
        row_idx, field_path, value = offender
        raise ValueError(
            f"non-finite value {value!r} in results[{row_idx}].{field_path}; "
            "canonical payload forbids NaN/Inf"
        ) from exc


def canonical_results_payload(results: Sequence[RunResult]) -> str:
    rows = [asdict(r) for r in results]
    return canonical_dumps(rows, rows_for_error=rows)


def _find_non_finite(
    rows: list[dict[str, Any]],
) -> tuple[int, str, Any] | None:
    for idx, row in enumerate(rows):
        hit = _find_non_finite_in_mapping(row)
        if hit is not None:
            field_path, value = hit
            return (idx, field_path, value)
    return None


def _find_non_finite_in_mapping(
    mapping: dict[str, Any], prefix: str = ""
) -> tuple[str, Any] | None:
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, float) and not math.isfinite(value):
            return (path, value)
        if isinstance(value, dict):
            hit = _find_non_finite_in_mapping(value, prefix=f"{path}.")
            if hit is not None:
                return hit
    return None


def payload_hash6(results: Sequence[RunResult]) -> str:
    """Return the first 6 lowercase hex chars of the payload SHA-256."""
    digest = hashlib.sha256(
        canonical_results_payload(results).encode("utf-8")
    ).hexdigest()
    return digest[:6]


def _escape_slug_value(value: Any) -> str:
    """Escape a param value for inclusion in a filename slug.

    ``+`` is reserved as the pair separator, whitespace and ``/`` would
    confuse filesystems; ``-`` and ``.`` are kept so numeric values read
    naturally.
    """
    s = str(value)
    out_chars: list[str] = []
    for ch in s:
        if ch == "+":
            out_chars.append("plus")
        elif ch.isspace() or ch == "/":
            out_chars.append("_")
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def slug_from_params(params: Mapping[str, Any]) -> str:
    """Build a deterministic ``k=v`` slug from non-excluded params.

    Keys are sorted; values are escaped so the ``+`` separator is
    unambiguous. Returns ``""`` when every key is excluded.
    """
    pairs = sorted(
        (k, v) for k, v in params.items() if k not in PARTITION_EXCLUDED_PARAM_KEYS
    )
    if not pairs:
        return ""
    return "+".join(f"{k}={_escape_slug_value(v)}" for k, v in pairs)
