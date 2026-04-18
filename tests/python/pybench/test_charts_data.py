from __future__ import annotations

import math
import random
import warnings
from typing import Any

import pytest

from pybench.charts.data import (
    canonical_results_payload,
    partition,
    partition_key,
    payload_hash6,
    safe_ratio,
    slug_from_params,
)
from pybench.recipe import RunResult


def _make_result(
    *,
    recipe_name: str = "dbscan",
    size: int = 1000,
    dims: int = 2,
    params: dict[str, Any] | None = None,
    ours_median_ms: float = 10.0,
    theirs_median_ms: float = 20.0,
    ours_peak_mb: float = 50.0,
    theirs_peak_mb: float = 100.0,
    ari: float = 0.99,
    ours_noise_frac: float = 0.01,
    theirs_noise_frac: float = 0.01,
    speedup: float = 2.0,
    timestamp: str = "2026-04-16T00:00:00+00:00",
) -> RunResult:
    return RunResult(
        recipe_name=recipe_name,
        size=size,
        dims=dims,
        params=dict(params) if params is not None else {"eps": 0.5, "min_samples": 5},
        ours_median_ms=ours_median_ms,
        theirs_median_ms=theirs_median_ms,
        ours_peak_mb=ours_peak_mb,
        theirs_peak_mb=theirs_peak_mb,
        ari=ari,
        ours_noise_frac=ours_noise_frac,
        theirs_noise_frac=theirs_noise_frac,
        speedup=speedup,
        timestamp=timestamp,
    )


def test_partition_groups_sizes_dims_and_n_jobs_into_one_bucket() -> None:
    base = {"eps": 0.5, "min_samples": 5}
    results = [
        _make_result(size=1000, dims=2, params={**base, "n_jobs": 1}),
        _make_result(size=5000, dims=2, params={**base, "n_jobs": 4}),
        _make_result(size=10000, dims=8, params={**base, "n_jobs": 8}),
        _make_result(size=50000, dims=32, params={**base, "n_jobs": 1}),
    ]

    groups = partition(results)

    assert len(groups) == 1
    (only_group,) = groups.values()
    assert len(only_group) == 4


def test_partition_splits_on_non_n_jobs_param_difference() -> None:
    a = _make_result(size=1000, params={"eps": 0.5, "min_samples": 5, "n_jobs": 1})
    b = _make_result(size=1000, params={"eps": 1.0, "min_samples": 5, "n_jobs": 1})
    c = _make_result(size=5000, params={"eps": 1.0, "min_samples": 5, "n_jobs": 4})

    groups = partition([a, b, c])

    assert len(groups) == 2
    # Locate groups by eps value.
    eps_values = {dict(k[1])["eps"]: len(v) for k, v in groups.items()}
    assert eps_values == {0.5: 1, 1.0: 2}


@pytest.mark.parametrize(
    ("bad_value", "type_name"),
    [
        ([0.5, 1.0], "list"),
        ({"x": 1}, "dict"),
        ((1, 2), "tuple"),
    ],
    ids=["list", "dict", "tuple"],
)
def test_partition_rejects_non_scalar_param_value(
    bad_value: Any, type_name: str
) -> None:
    bad = _make_result(params={"eps": bad_value})
    with pytest.raises(ValueError) as exc:
        partition([bad])
    assert "eps" in str(exc.value)


def test_partition_scalar_booleans_and_none_are_accepted() -> None:
    r = _make_result(
        params={"flag": True, "tag": None, "name": "x", "eps": 0.5, "n_jobs": 1}
    )
    groups = partition([r])
    assert len(groups) == 1


def test_partition_raises_on_duplicate_recipe_size_dims_params() -> None:
    a = _make_result(size=1000, dims=2, params={"eps": 0.5, "n_jobs": 1})
    b = _make_result(size=1000, dims=2, params={"eps": 0.5, "n_jobs": 1})

    with pytest.raises(ValueError) as exc:
        partition([a, b])

    msg = str(exc.value)
    assert "duplicate" in msg.lower()
    assert "dbscan" in msg
    assert "size=1000" in msg
    assert "dims=2" in msg


def test_partition_allows_same_recipe_size_dims_differing_in_n_jobs_only() -> None:
    a = _make_result(params={"eps": 0.5, "n_jobs": 1})
    b = _make_result(params={"eps": 0.5, "n_jobs": 4})
    groups = partition([a, b])
    assert sum(len(v) for v in groups.values()) == 2


@pytest.mark.parametrize(
    ("num", "den"),
    [
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, math.nan),
        (1.0, math.inf),
        (math.nan, 1.0),
        (math.inf, 1.0),
        (-math.inf, 1.0),
    ],
)
def test_safe_ratio_returns_nan_on_unsafe_inputs(num: float, den: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = safe_ratio(num, den)
    assert math.isnan(result)


def test_safe_ratio_returns_quotient_on_valid_inputs() -> None:
    assert safe_ratio(10.0, 2.0) == 5.0
    assert safe_ratio(1.0, 4.0) == 0.25


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("ari", math.nan), ("speedup", math.inf)],
    ids=["nan", "inf"],
)
def test_canonical_payload_rejects_non_finite_with_named_field(
    field: str, bad_value: float
) -> None:
    r = _make_result(**{field: bad_value})
    with pytest.raises(ValueError) as exc:
        canonical_results_payload([r])
    msg = str(exc.value)
    assert field in msg
    assert "results[0]" in msg


def test_canonical_payload_is_byte_stable_under_params_reordering() -> None:
    a = _make_result(params={"eps": 1.0, "min_samples": 5, "n_jobs": 1})
    b = _make_result(params={"n_jobs": 1, "min_samples": 5, "eps": 1.0})

    assert canonical_results_payload([a]) == canonical_results_payload([b])
    assert payload_hash6([a]) == payload_hash6([b])


def test_payload_hash6_is_six_lowercase_hex_chars() -> None:
    h = payload_hash6([_make_result()])
    assert len(h) == 6
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


def test_payload_hash6_changes_when_data_changes() -> None:
    a = _make_result(ours_median_ms=10.0)
    b = _make_result(ours_median_ms=11.0)
    assert payload_hash6([a]) != payload_hash6([b])


def test_partition_key_is_stable_under_param_insertion_order() -> None:
    rng = random.Random(0xC0DE)
    keys_pool = ["eps", "min_samples", "leaf_size", "algorithm", "metric"]
    values_pool = [0.1, 0.5, 1.0, 2, 5, 10, "auto", "kd_tree", "euclidean", True]

    for _ in range(20):
        size = rng.randint(1, 5)
        pairs = [(rng.choice(keys_pool), rng.choice(values_pool)) for _ in range(size)]
        pairs = list(dict(pairs).items())
        d1 = dict(pairs)
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        d1["n_jobs"] = 1
        d2 = dict(shuffled)
        d2["n_jobs"] = 8

        r1 = _make_result(params=d1)
        r2 = _make_result(params=d2)

        assert partition_key(r1) == partition_key(r2)


def test_slug_excludes_n_jobs_and_sorts_keys() -> None:
    assert (
        slug_from_params({"eps": 0.5, "min_samples": 5, "n_jobs": 4})
        == "eps=0.5+min_samples=5"
    )


def test_slug_escapes_plus_and_whitespace_and_slash() -> None:
    assert slug_from_params({"tag": "a+b"}) == "tag=aplusb"
    assert slug_from_params({"tag": "a b"}) == "tag=a_b"
    assert slug_from_params({"tag": "a/b"}) == "tag=a_b"


def test_slug_preserves_dot_and_hyphen() -> None:
    assert slug_from_params({"eps": 0.5}) == "eps=0.5"
    assert slug_from_params({"label": "pre-release"}) == "label=pre-release"


def test_slug_empty_when_only_excluded_keys() -> None:
    assert slug_from_params({"n_jobs": 8}) == ""
    assert slug_from_params({}) == ""
