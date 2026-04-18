"""Smoke tests that each recipe's @c ours and @c theirs produce valid label
arrays at every dim in @c default_dims."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pybench.recipes.dbscan import recipe as dbscan_recipe
from pybench.recipes.kmeans import recipe as kmeans_recipe
from pybench.recipe import Recipe
from pybench.runner import make_data

SMOKE_SIZE = 1000


def _recipe_dim_params() -> list[pytest.param]:
    cases = []
    for recipe in (dbscan_recipe, kmeans_recipe):
        for dim in recipe.default_dims:
            cases.append(pytest.param(recipe, dim, id=f"{recipe.name}-d{dim}"))
    return cases


@pytest.mark.parametrize(("recipe", "dim"), _recipe_dim_params())
@pytest.mark.parametrize("side", ["ours", "theirs"])
def test_recipe_returns_integer_labels(recipe: Recipe, dim: int, side: str) -> None:
    dataset = replace(recipe.dataset, n_features=dim)
    data = make_data(SMOKE_SIZE, dataset)
    fn = recipe.ours if side == "ours" else recipe.theirs

    labels = fn(data, **recipe.default_params)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (SMOKE_SIZE,)
    assert np.issubdtype(labels.dtype, np.integer)
