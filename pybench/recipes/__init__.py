from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

from pybench.recipe import Recipe

logger = logging.getLogger(__name__)

_PACKAGE_PATH = [str(Path(__file__).parent)]
_PACKAGE_NAME = __name__


def all_recipes() -> dict[str, Recipe]:
    """Discover and return all Recipe instances from sibling modules."""
    recipes: dict[str, Recipe] = {}

    for module_info in pkgutil.iter_modules(_PACKAGE_PATH):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{_PACKAGE_NAME}.{module_info.name}"
        try:
            mod = importlib.import_module(full_name)
        except ImportError as exc:
            logger.warning("Could not import recipe module %s: %s", full_name, exc)
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, Recipe):
                recipes[obj.name] = obj

    return recipes
