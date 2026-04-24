"""Sidecar `.npz` archive for cluster labels and 2D projections.

Written alongside `{algo}_{slug}_{hash6}.png` chart files so that a later
``pybench vis`` invocation can render fixture-label visualizations without
re-running the benchmark.

One sidecar per partition (recipe + non-`n_jobs` params). Filename matches the
chart prefix so the association is obvious on disk. A JSON freshness envelope
is stored inside the archive under ``__envelope__`` as a uint8 byte array, so
the filename itself stays identifier-clean.

The archive is written atomically via a sibling temp-file + ``os.replace`` so a
crashed write can never leave the canonical path partially-written.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import threadpoolctl

from pybench.charts.data import slug_from_params
from pybench.charts.results_io import _CANONICAL_ENCODING_VERSION

_ENVELOPE_KEY = "__envelope__"
_CELL_PREFIX = "cell__"
_FIELD_GT = "gt_labels"
_FIELD_OURS = "ours_labels"
_FIELD_THEIRS = "theirs_labels"
_FIELD_PROJ = "projection_2d"


@dataclass(frozen=True, slots=True)
class LabelCell:
    """Per-cell label arrays for one (recipe, params, size, dims, n_jobs) run.

    ``theirs_labels`` is ``None`` when the run was executed under
    ``--ours-only`` and the competitor was not invoked.
    """

    gt_labels: np.ndarray
    ours_labels: np.ndarray
    projection_2d: np.ndarray
    theirs_labels: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FreshnessEnvelope:
    """Metadata stored inside the sidecar archive for staleness detection.

    ``hash6`` and ``recipe_name`` are the authoritative identity of the
    partition; ``sklearn_version``, ``threadpool_info``, and
    ``canonical_encoding_version`` are soft indicators that the cached labels
    were produced against the same dependency stack.
    """

    hash6: str
    recipe_name: str
    sklearn_version: str
    threadpool_info: list[dict[str, Any]]
    canonical_encoding_version: int


class LoadStatus(Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Outcome of a sidecar load attempt.

    - ``OK``: envelope matches exactly; ``cells`` holds the arrays.
    - ``WARN``: soft mismatch (sklearn/threadpool/encoding); ``cells`` still
      populated so the caller can decide whether to regenerate.
    - ``ERROR``: missing file, corrupt archive, or identity mismatch (hash6
      or recipe_name); ``cells`` is an empty dict and ``envelope`` may be
      ``None``.
    """

    status: LoadStatus
    reason: str | None
    envelope: FreshnessEnvelope | None
    cells: dict[str, LabelCell] = field(default_factory=dict)


def labels_sidecar_filename(algo: str, params_slug: str, hash6: str) -> str:
    """Return the sidecar filename matching the chart's prefix.

    Empty ``params_slug`` collapses to ``{algo}_{hash6}.labels.npz`` so it
    matches ``chart_filename`` when every param is excluded from the slug.
    """
    if params_slug:
        return f"{algo}_{params_slug}_{hash6}.labels.npz"
    return f"{algo}_{hash6}.labels.npz"


def labels_sidecar_filename_for_params(
    algo: str, params: dict[str, Any], hash6: str
) -> str:
    """Convenience wrapper that builds the slug from ``params``."""
    return labels_sidecar_filename(algo, slug_from_params(params), hash6)


def _capture_threadpool_info() -> list[dict[str, Any]]:
    raw = threadpoolctl.threadpool_info()
    # Normalize through JSON so non-JSON-safe objects (e.g. enums) become
    # primitives; this also decouples the envelope from threadpoolctl's
    # internal dict-ordering quirks.
    return json.loads(json.dumps(raw))


def _build_envelope(hash6: str, recipe_name: str) -> FreshnessEnvelope:
    return FreshnessEnvelope(
        hash6=hash6,
        recipe_name=recipe_name,
        sklearn_version=sklearn.__version__,
        threadpool_info=_capture_threadpool_info(),
        canonical_encoding_version=_CANONICAL_ENCODING_VERSION,
    )


def _envelope_to_bytes(envelope: FreshnessEnvelope) -> np.ndarray:
    payload = {
        "hash6": envelope.hash6,
        "recipe_name": envelope.recipe_name,
        "sklearn_version": envelope.sklearn_version,
        "threadpool_info": envelope.threadpool_info,
        "canonical_encoding_version": envelope.canonical_encoding_version,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _envelope_from_bytes(blob: np.ndarray) -> FreshnessEnvelope:
    raw = bytes(blob.tobytes())
    data = json.loads(raw.decode("utf-8"))
    return FreshnessEnvelope(
        hash6=data["hash6"],
        recipe_name=data["recipe_name"],
        sklearn_version=data["sklearn_version"],
        threadpool_info=data["threadpool_info"],
        canonical_encoding_version=data["canonical_encoding_version"],
    )


def _validate_cell(key: str, cell: LabelCell) -> None:
    if cell.gt_labels.dtype != np.int32:
        raise ValueError(
            f"cell[{key!r}].gt_labels dtype must be int32, got {cell.gt_labels.dtype}"
        )
    if cell.ours_labels.dtype != np.int32:
        raise ValueError(
            f"cell[{key!r}].ours_labels dtype must be int32, got "
            f"{cell.ours_labels.dtype}"
        )
    if cell.theirs_labels is not None and cell.theirs_labels.dtype != np.int32:
        raise ValueError(
            f"cell[{key!r}].theirs_labels dtype must be int32, got "
            f"{cell.theirs_labels.dtype}"
        )
    if cell.projection_2d.dtype != np.float32:
        raise ValueError(
            f"cell[{key!r}].projection_2d dtype must be float32, got "
            f"{cell.projection_2d.dtype}"
        )
    if cell.gt_labels.ndim != 1:
        raise ValueError(
            f"cell[{key!r}].gt_labels must be 1-D, got shape {cell.gt_labels.shape}"
        )
    if cell.ours_labels.ndim != 1:
        raise ValueError(
            f"cell[{key!r}].ours_labels must be 1-D, got shape {cell.ours_labels.shape}"
        )
    if cell.theirs_labels is not None and cell.theirs_labels.ndim != 1:
        raise ValueError(
            f"cell[{key!r}].theirs_labels must be 1-D, got shape "
            f"{cell.theirs_labels.shape}"
        )
    if cell.projection_2d.ndim != 2 or cell.projection_2d.shape[1] != 2:
        raise ValueError(
            f"cell[{key!r}].projection_2d must have shape (n, 2), got "
            f"{cell.projection_2d.shape}"
        )


def _cell_archive_key(cell_key: str, field_name: str) -> str:
    return f"{_CELL_PREFIX}{cell_key}__{field_name}"


def _parse_archive_key(name: str) -> tuple[str, str] | None:
    if not name.startswith(_CELL_PREFIX):
        return None
    rest = name[len(_CELL_PREFIX) :]
    sep = rest.rfind("__")
    if sep < 0:
        return None
    return rest[:sep], rest[sep + 2 :]


def save_labels(
    path: Path,
    hash6: str,
    recipe_name: str,
    cells: dict[str, LabelCell],
) -> None:
    """Write ``cells`` to ``path`` as a compressed ``.npz`` sidecar.

    The write is atomic: a sibling temp-file is populated first, then moved
    into place with ``os.replace``. If ``np.savez_compressed`` raises, the
    temp-file is removed and the canonical ``path`` is untouched.

    The freshness envelope (``hash6``, ``recipe_name``, current
    ``sklearn.__version__``, ``threadpoolctl.threadpool_info()``, and
    ``_CANONICAL_ENCODING_VERSION``) is captured at call time and stored
    inside the archive under ``__envelope__``.

    Raises ``ValueError`` on dtype/shape mismatches in cells so we fail at the
    producer, not the consumer.
    """
    for key, cell in cells.items():
        _validate_cell(key, cell)

    envelope = _build_envelope(hash6, recipe_name)
    arrays: dict[str, np.ndarray] = {_ENVELOPE_KEY: _envelope_to_bytes(envelope)}
    for cell_key, cell in cells.items():
        arrays[_cell_archive_key(cell_key, _FIELD_GT)] = cell.gt_labels
        arrays[_cell_archive_key(cell_key, _FIELD_OURS)] = cell.ours_labels
        arrays[_cell_archive_key(cell_key, _FIELD_PROJ)] = cell.projection_2d
        if cell.theirs_labels is not None:
            arrays[_cell_archive_key(cell_key, _FIELD_THEIRS)] = cell.theirs_labels

    path.parent.mkdir(parents=True, exist_ok=True)

    # np.savez_compressed unconditionally appends ".npz" when the target
    # path does not already end with ".npz", so the temp suffix must end in
    # ".npz" for numpy to write to the path we preallocated. mkstemp gives
    # us an exclusive path on `path.parent` (same filesystem), which is a
    # precondition for `os.replace` to be atomic on POSIX.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.npz",
        dir=str(path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(str(tmp_path), str(path))
    except BaseException:
        # Clean up the temp file on any failure, including a crash in
        # np.savez_compressed mid-write or in os.replace.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_archive_raw(
    path: Path,
) -> tuple[np.lib.npyio.NpzFile, None] | tuple[None, str]:
    if not path.exists():
        return None, "file does not exist"
    try:
        # allow_pickle=False: labels archives must be primitives-only.
        archive = np.load(path, allow_pickle=False)
    except zipfile.BadZipFile as exc:
        return None, f"corrupt archive: {exc}"
    except (OSError, ValueError, EOFError) as exc:
        return None, f"unreadable archive: {exc}"
    return archive, None


def _extract_cells(archive: np.lib.npyio.NpzFile) -> dict[str, LabelCell]:
    accum: dict[str, dict[str, np.ndarray]] = {}
    for name in archive.files:
        if name == _ENVELOPE_KEY:
            continue
        parsed = _parse_archive_key(name)
        if parsed is None:
            continue
        cell_key, field_name = parsed
        accum.setdefault(cell_key, {})[field_name] = archive[name]

    cells: dict[str, LabelCell] = {}
    for cell_key, fields in accum.items():
        gt = fields.get(_FIELD_GT)
        ours = fields.get(_FIELD_OURS)
        proj = fields.get(_FIELD_PROJ)
        if gt is None or ours is None or proj is None:
            raise ValueError(
                f"cell {cell_key!r} missing required fields; got keys "
                f"{sorted(fields.keys())}"
            )
        theirs = fields.get(_FIELD_THEIRS)
        cells[cell_key] = LabelCell(
            gt_labels=gt,
            ours_labels=ours,
            projection_2d=proj,
            theirs_labels=theirs,
        )
    return cells


def load_labels(
    path: Path,
    expected_hash6: str,
    expected_recipe_name: str,
) -> LoadResult:
    """Load a sidecar archive and classify its freshness.

    Returns a :class:`LoadResult` with:

    - ``OK``: envelope matches exactly, arrays populated.
    - ``WARN``: soft mismatch (``sklearn_version``, ``threadpool_info``, or
      ``canonical_encoding_version``). Arrays are still returned; caller
      decides regen vs proceed.
    - ``ERROR``: missing file, corrupt archive, envelope missing/unparseable,
      or identity mismatch (``hash6`` / ``recipe_name``). ``cells`` is empty.

    Never raises for the expected error conditions above; unexpected array
    corruption (e.g. a broken cell key inside an otherwise-valid envelope)
    is propagated as ``LoadResult(status=ERROR)`` as well.
    """
    archive, err = _load_archive_raw(path)
    if archive is None:
        assert err is not None
        return LoadResult(
            status=LoadStatus.ERROR,
            reason=err,
            envelope=None,
            cells={},
        )

    try:
        if _ENVELOPE_KEY not in archive.files:
            return LoadResult(
                status=LoadStatus.ERROR,
                reason="envelope missing from archive",
                envelope=None,
                cells={},
            )

        try:
            envelope = _envelope_from_bytes(archive[_ENVELOPE_KEY])
        except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return LoadResult(
                status=LoadStatus.ERROR,
                reason=f"envelope unparseable: {exc}",
                envelope=None,
                cells={},
            )

        if envelope.hash6 != expected_hash6:
            return LoadResult(
                status=LoadStatus.ERROR,
                reason=(
                    f"hash6 mismatch: expected {expected_hash6!r}, "
                    f"got {envelope.hash6!r}"
                ),
                envelope=envelope,
                cells={},
            )
        if envelope.recipe_name != expected_recipe_name:
            return LoadResult(
                status=LoadStatus.ERROR,
                reason=(
                    f"recipe_name mismatch: expected "
                    f"{expected_recipe_name!r}, got {envelope.recipe_name!r}"
                ),
                envelope=envelope,
                cells={},
            )

        try:
            cells = _extract_cells(archive)
        except ValueError as exc:
            return LoadResult(
                status=LoadStatus.ERROR,
                reason=f"cell extraction failed: {exc}",
                envelope=envelope,
                cells={},
            )

        warn_reasons: list[str] = []
        current_sklearn = sklearn.__version__
        current_tp = _capture_threadpool_info()
        if envelope.sklearn_version != current_sklearn:
            warn_reasons.append(
                f"sklearn_version mismatch "
                f"(archive={envelope.sklearn_version!r}, "
                f"current={current_sklearn!r})"
            )
        if envelope.threadpool_info != current_tp:
            warn_reasons.append("threadpool_info mismatch")
        if envelope.canonical_encoding_version != _CANONICAL_ENCODING_VERSION:
            warn_reasons.append(
                f"canonical_encoding_version mismatch "
                f"(archive={envelope.canonical_encoding_version}, "
                f"current={_CANONICAL_ENCODING_VERSION})"
            )

        if warn_reasons:
            return LoadResult(
                status=LoadStatus.WARN,
                reason="; ".join(warn_reasons),
                envelope=envelope,
                cells=cells,
            )

        return LoadResult(
            status=LoadStatus.OK,
            reason=None,
            envelope=envelope,
            cells=cells,
        )
    finally:
        archive.close()
