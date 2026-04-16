"""Smoke tests for recipe default dim sweeps.

Verifies that both DBSCAN and KMeans recipes advertise the unified default
`(2, 8, 32, 128)` dim sweep and that each dim in that sweep produces a
valid label array at size=1000 -- both from our C++ binding and from the
sklearn reference.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pybench.recipes.dbscan import recipe as dbscan_recipe
from pybench.recipes.kmeans import recipe as kmeans_recipe
from pybench.recipe import Recipe
from pybench.runner import make_data, make_sklearn_reference

EXPECTED_DEFAULT_DIMS = (2, 8, 32, 128)
SMOKE_SIZE = 1000


@pytest.mark.parametrize(
    "recipe", [dbscan_recipe, kmeans_recipe], ids=["dbscan", "kmeans"]
)
def test_default_dims_are_unified(recipe: Recipe) -> None:
    assert recipe.default_dims == EXPECTED_DEFAULT_DIMS


def _resolve_theirs(recipe: Recipe):
    return (
        recipe.theirs
        if recipe.theirs is not None
        else make_sklearn_reference(recipe.name)
    )


def _recipe_dim_params() -> list[pytest.param]:
    cases = []
    for recipe in (dbscan_recipe, kmeans_recipe):
        for dim in recipe.default_dims:
            cases.append(pytest.param(recipe, dim, id=f"{recipe.name}-d{dim}"))
    return cases


@pytest.mark.parametrize(("recipe", "dim"), _recipe_dim_params())
def test_ours_returns_integer_labels_at_each_default_dim(
    recipe: Recipe, dim: int
) -> None:
    dataset = replace(recipe.dataset, n_features=dim)
    data = make_data(SMOKE_SIZE, dataset)

    labels = recipe.ours(data, **recipe.default_params)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (SMOKE_SIZE,)
    assert np.issubdtype(labels.dtype, np.integer)


@pytest.mark.parametrize(("recipe", "dim"), _recipe_dim_params())
def test_sklearn_reference_returns_integer_labels_at_each_default_dim(
    recipe: Recipe, dim: int
) -> None:
    dataset = replace(recipe.dataset, n_features=dim)
    data = make_data(SMOKE_SIZE, dataset)

    theirs = _resolve_theirs(recipe)
    labels = theirs(data, **recipe.default_params)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (SMOKE_SIZE,)
    assert np.issubdtype(labels.dtype, np.integer)
