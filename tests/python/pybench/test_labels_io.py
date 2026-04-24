from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pybench.charts.labels_io import (
    FreshnessEnvelope,
    LabelCell,
    LoadResult,
    LoadStatus,
    _ENVELOPE_KEY,
    _envelope_to_bytes,
    labels_sidecar_filename,
    labels_sidecar_filename_for_params,
    load_labels,
    save_labels,
)
from pybench.charts.results_io import _CANONICAL_ENCODING_VERSION


def _make_cell(
    n: int = 64,
    *,
    theirs: bool = True,
    rng_seed: int = 0,
) -> LabelCell:
    rng = np.random.default_rng(rng_seed)
    return LabelCell(
        gt_labels=rng.integers(0, 8, n, dtype=np.int32),
        ours_labels=rng.integers(0, 8, n, dtype=np.int32),
        theirs_labels=(rng.integers(0, 8, n, dtype=np.int32) if theirs else None),
        projection_2d=rng.standard_normal((n, 2)).astype(np.float32),
    )


def _assert_cells_bit_identical(
    saved: dict[str, LabelCell], loaded: dict[str, LabelCell]
) -> None:
    assert sorted(saved.keys()) == sorted(loaded.keys())
    for key in saved:
        src = saved[key]
        got = loaded[key]
        assert np.array_equal(src.gt_labels, got.gt_labels)
        assert np.array_equal(src.ours_labels, got.ours_labels)
        assert np.array_equal(src.projection_2d, got.projection_2d)
        assert src.gt_labels.dtype == got.gt_labels.dtype == np.int32
        assert src.ours_labels.dtype == got.ours_labels.dtype == np.int32
        assert src.projection_2d.dtype == got.projection_2d.dtype == np.float32
        if src.theirs_labels is None:
            assert got.theirs_labels is None
        else:
            assert got.theirs_labels is not None
            assert np.array_equal(src.theirs_labels, got.theirs_labels)
            assert got.theirs_labels.dtype == np.int32


def test_round_trip_single_cell(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=32)}
    path = tmp_path / "kmeans_n_clusters=4_abc123.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.OK
    assert result.reason is None
    assert result.envelope is not None
    assert result.envelope.hash6 == "abc123"
    assert result.envelope.recipe_name == "kmeans_blobs"
    _assert_cells_bit_identical(cells, result.cells)


def test_round_trip_multi_cell(tmp_path: Path) -> None:
    cells = {
        "n=100__d=2__jobs=1": _make_cell(n=50, rng_seed=1),
        "n=100__d=2__jobs=4": _make_cell(n=50, rng_seed=2),
        "n=500__d=8__jobs=32": _make_cell(n=120, rng_seed=3),
    }
    path = tmp_path / "multi.labels.npz"

    save_labels(path, "123456", "dbscan_blobs", cells)
    result = load_labels(path, "123456", "dbscan_blobs")

    assert result.status is LoadStatus.OK
    _assert_cells_bit_identical(cells, result.cells)


def test_round_trip_with_theirs_none(tmp_path: Path) -> None:
    cells = {
        "ours_only_0": _make_cell(n=40, theirs=False, rng_seed=4),
        "ours_only_1": _make_cell(n=40, theirs=False, rng_seed=5),
    }
    path = tmp_path / "ours_only.labels.npz"

    save_labels(path, "ffffff", "kmeans_blobs", cells)
    result = load_labels(path, "ffffff", "kmeans_blobs")

    assert result.status is LoadStatus.OK
    for _, cell in result.cells.items():
        assert cell.theirs_labels is None
    _assert_cells_bit_identical(cells, result.cells)


def test_round_trip_empty_cells(tmp_path: Path) -> None:
    path = tmp_path / "empty.labels.npz"

    save_labels(path, "deadbe", "dbscan_blobs", {})
    result = load_labels(path, "deadbe", "dbscan_blobs")

    assert result.status is LoadStatus.OK
    assert result.cells == {}
    assert result.envelope is not None
    assert result.envelope.hash6 == "deadbe"


def test_atomic_write_never_leaves_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between temp-file creation and rename must leave `path` absent."""
    cells = {"cell_0": _make_cell(n=32)}
    path = tmp_path / "atomic.labels.npz"

    real_replace = os.replace

    def boom_replace(src: str, dst: str) -> None:
        # Verify that the staged temp file did get written before the crash;
        # this exercises the branch we care about (write succeeded, rename
        # failed).
        assert Path(src).exists()
        raise OSError("simulated rename failure")

    monkeypatch.setattr("pybench.charts.labels_io.os.replace", boom_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        save_labels(path, "abc123", "kmeans_blobs", cells)

    assert not path.exists()
    # No orphan temp files should remain alongside `path`.
    siblings = list(tmp_path.iterdir())
    assert siblings == [], f"orphan files left behind: {siblings}"

    # Sanity: a follow-up normal save should succeed (nothing is locked).
    monkeypatch.setattr("pybench.charts.labels_io.os.replace", real_replace)
    save_labels(path, "abc123", "kmeans_blobs", cells)
    assert path.exists()


def test_missing_file_returns_error_status(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.labels.npz"
    result = load_labels(path, "abc123", "kmeans_blobs")
    assert result.status is LoadStatus.ERROR
    assert result.reason is not None
    assert "not exist" in result.reason.lower() or "no such" in result.reason.lower()
    assert result.cells == {}
    assert result.envelope is None


def test_hash6_mismatch_returns_error_status(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=16)}
    path = tmp_path / "hash.labels.npz"

    save_labels(path, "aaaaaa", "kmeans_blobs", cells)
    result = load_labels(path, "bbbbbb", "kmeans_blobs")

    assert result.status is LoadStatus.ERROR
    assert result.reason is not None
    assert "hash6 mismatch" in result.reason
    assert result.envelope is not None
    assert result.envelope.hash6 == "aaaaaa"
    assert result.cells == {}


def test_recipe_name_mismatch_returns_error_status(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=16)}
    path = tmp_path / "recipe.labels.npz"

    save_labels(path, "abc123", "dbscan_blobs", cells)
    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.ERROR
    assert result.reason is not None
    assert "recipe_name mismatch" in result.reason
    assert result.cells == {}


def _rewrite_envelope(path: Path, overrides: dict[str, Any]) -> None:
    """Load a sidecar, patch envelope fields, and re-save with the same arrays."""
    with np.load(path, allow_pickle=False) as arc:
        arrays = {name: arc[name].copy() for name in arc.files}
    envelope_bytes = bytes(arrays[_ENVELOPE_KEY].tobytes())
    envelope = json.loads(envelope_bytes.decode("utf-8"))
    envelope.update(overrides)
    raw = json.dumps(envelope, sort_keys=True).encode("utf-8")
    arrays[_ENVELOPE_KEY] = np.frombuffer(raw, dtype=np.uint8).copy()
    np.savez_compressed(path, **arrays)


def test_sklearn_version_mismatch_returns_warn_status(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=16, rng_seed=6)}
    path = tmp_path / "sklearn_mismatch.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    _rewrite_envelope(path, {"sklearn_version": "0.0.0-not-a-real-version"})

    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.WARN
    assert result.reason is not None
    assert "sklearn_version" in result.reason
    _assert_cells_bit_identical(cells, result.cells)


def test_threadpool_info_mismatch_returns_warn_status(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=16, rng_seed=7)}
    path = tmp_path / "threadpool_mismatch.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    _rewrite_envelope(
        path,
        {
            "threadpool_info": [
                {"user_api": "totally_fake_api", "num_threads": 9999},
            ],
        },
    )

    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.WARN
    assert result.reason is not None
    assert "threadpool_info" in result.reason
    _assert_cells_bit_identical(cells, result.cells)


def test_canonical_encoding_version_mismatch_returns_warn_status(
    tmp_path: Path,
) -> None:
    cells = {"cell_0": _make_cell(n=16, rng_seed=8)}
    path = tmp_path / "encoding_mismatch.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    _rewrite_envelope(
        path, {"canonical_encoding_version": _CANONICAL_ENCODING_VERSION + 99}
    )

    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.WARN
    assert result.reason is not None
    assert "canonical_encoding_version" in result.reason
    _assert_cells_bit_identical(cells, result.cells)


def test_corrupt_archive_returns_error_status(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.labels.npz"
    path.write_bytes(b"\x50\x4b\x03\x04 not_a_valid_zip_entry_blob")

    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.ERROR
    assert result.reason is not None
    # Expect either "corrupt" or "unreadable" in the reason
    assert "corrupt" in result.reason.lower() or "unreadable" in result.reason.lower()
    assert result.cells == {}
    assert result.envelope is None


def test_missing_envelope_returns_error_status(tmp_path: Path) -> None:
    """An archive without the envelope key must be flagged as ERROR."""
    path = tmp_path / "no_envelope.labels.npz"
    np.savez_compressed(path, cell__foo__gt_labels=np.zeros(4, dtype=np.int32))

    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.ERROR
    assert result.reason is not None
    assert "envelope" in result.reason


def test_unparseable_envelope_returns_error_status(tmp_path: Path) -> None:
    path = tmp_path / "bad_envelope.labels.npz"
    bad = np.frombuffer(b"\xff\xfe not utf-8 json \x00", dtype=np.uint8)
    np.savez_compressed(path, **{_ENVELOPE_KEY: bad})

    result = load_labels(path, "abc123", "kmeans_blobs")
    assert result.status is LoadStatus.ERROR
    assert result.reason is not None
    assert "envelope" in result.reason.lower()


def test_int32_arrays_preserve_dtype(tmp_path: Path) -> None:
    """Guard against accidental int64 leakage in label arrays."""
    cells = {"cell_0": _make_cell(n=24, rng_seed=9)}
    path = tmp_path / "dtype_int32.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.OK
    for cell in result.cells.values():
        assert cell.gt_labels.dtype == np.int32
        assert cell.ours_labels.dtype == np.int32
        if cell.theirs_labels is not None:
            assert cell.theirs_labels.dtype == np.int32


def test_float32_projection_preserves_dtype(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=24, rng_seed=10)}
    path = tmp_path / "dtype_float32.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    result = load_labels(path, "abc123", "kmeans_blobs")

    assert result.status is LoadStatus.OK
    for cell in result.cells.values():
        assert cell.projection_2d.dtype == np.float32


def test_save_rejects_wrong_label_dtype(tmp_path: Path) -> None:
    bad = LabelCell(
        gt_labels=np.zeros(8, dtype=np.int64),  # wrong dtype
        ours_labels=np.zeros(8, dtype=np.int32),
        projection_2d=np.zeros((8, 2), dtype=np.float32),
        theirs_labels=None,
    )
    path = tmp_path / "reject.labels.npz"
    with pytest.raises(ValueError, match="int32"):
        save_labels(path, "abc123", "kmeans_blobs", {"bad": bad})
    assert not path.exists()


def test_save_rejects_wrong_projection_dtype(tmp_path: Path) -> None:
    bad = LabelCell(
        gt_labels=np.zeros(8, dtype=np.int32),
        ours_labels=np.zeros(8, dtype=np.int32),
        projection_2d=np.zeros((8, 2), dtype=np.float64),  # wrong dtype
        theirs_labels=None,
    )
    path = tmp_path / "reject.labels.npz"
    with pytest.raises(ValueError, match="float32"):
        save_labels(path, "abc123", "kmeans_blobs", {"bad": bad})
    assert not path.exists()


def test_save_rejects_projection_wrong_shape(tmp_path: Path) -> None:
    bad = LabelCell(
        gt_labels=np.zeros(8, dtype=np.int32),
        ours_labels=np.zeros(8, dtype=np.int32),
        projection_2d=np.zeros((8, 3), dtype=np.float32),  # not (n, 2)
        theirs_labels=None,
    )
    path = tmp_path / "reject_shape.labels.npz"
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        save_labels(path, "abc123", "kmeans_blobs", {"bad": bad})


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    cells = {"cell_0": _make_cell(n=8, rng_seed=11)}
    path = tmp_path / "nested" / "dir" / "sidecar.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)
    assert path.is_file()


def test_save_creates_file_atomic_rename_path(tmp_path: Path) -> None:
    """Happy-path atomic write: only the final file exists after a save."""
    cells = {"cell_0": _make_cell(n=8, rng_seed=12)}
    path = tmp_path / "happy.labels.npz"

    save_labels(path, "abc123", "kmeans_blobs", cells)

    # No `.tmp` siblings left behind.
    siblings = sorted(tmp_path.iterdir())
    assert siblings == [path]


def test_sidecar_filename_matches_chart_prefix() -> None:
    # Chart filename for the same inputs is
    # "kmeans_n_clusters=16_271045.png" (chart_filename), so the sidecar
    # must share the "kmeans_n_clusters=16_271045" prefix.
    assert (
        labels_sidecar_filename("kmeans", "n_clusters=16", "271045")
        == "kmeans_n_clusters=16_271045.labels.npz"
    )


def test_sidecar_filename_with_empty_slug() -> None:
    assert labels_sidecar_filename("dbscan", "", "abc123") == "dbscan_abc123.labels.npz"


def test_sidecar_filename_for_params_matches_chart_filename() -> None:
    """Cross-check against the actual chart_filename builder so the two
    stay locked together if someone changes slug_from_params later."""
    from pybench.charts.filenames import chart_filename

    params = {"eps": 0.5, "min_samples": 5, "n_jobs": 8}
    chart = chart_filename("dbscan", params, "abc123")
    sidecar = labels_sidecar_filename_for_params("dbscan", params, "abc123")

    chart_prefix = chart[: -len(".png")]
    sidecar_prefix = sidecar[: -len(".labels.npz")]
    assert chart_prefix == sidecar_prefix


def test_envelope_is_json_safe() -> None:
    """The envelope payload must survive a JSON round-trip with no fallback
    because `.npz` archives store it as raw bytes."""
    env = FreshnessEnvelope(
        hash6="abc123",
        recipe_name="kmeans_blobs",
        sklearn_version="1.2.3",
        threadpool_info=[{"user_api": "blas", "num_threads": 8}],
        canonical_encoding_version=_CANONICAL_ENCODING_VERSION,
    )
    blob = _envelope_to_bytes(env)
    assert blob.dtype == np.uint8
    decoded = json.loads(bytes(blob.tobytes()).decode("utf-8"))
    assert decoded["hash6"] == "abc123"
    assert decoded["recipe_name"] == "kmeans_blobs"
    assert decoded["sklearn_version"] == "1.2.3"
    assert decoded["canonical_encoding_version"] == _CANONICAL_ENCODING_VERSION


def test_load_result_default_empty_cells() -> None:
    """LoadResult defaults cells to an empty dict when not supplied."""
    r = LoadResult(status=LoadStatus.ERROR, reason="missing", envelope=None)
    assert r.cells == {}


def test_round_trip_across_saves_is_deterministic(tmp_path: Path) -> None:
    """Two saves of the same payload produce identical envelope contents."""
    cells = {"cell_0": _make_cell(n=16, rng_seed=13)}

    path_a = tmp_path / "a.labels.npz"
    path_b = tmp_path / "b.labels.npz"

    save_labels(path_a, "abc123", "kmeans_blobs", cells)
    save_labels(path_b, "abc123", "kmeans_blobs", cells)

    with np.load(path_a, allow_pickle=False) as arc_a:
        env_a = bytes(arc_a[_ENVELOPE_KEY].tobytes())
    with np.load(path_b, allow_pickle=False) as arc_b:
        env_b = bytes(arc_b[_ENVELOPE_KEY].tobytes())

    # The envelope contents are deterministic when the environment (sklearn,
    # threadpool) hasn't changed between the two saves.
    assert env_a == env_b


def test_save_overwrites_existing_file(tmp_path: Path) -> None:
    """A second save at the same path replaces the prior contents."""
    path = tmp_path / "overwrite.labels.npz"
    cells_a = {"cell_0": _make_cell(n=8, rng_seed=14)}
    cells_b = {"cell_0": _make_cell(n=8, rng_seed=15)}

    save_labels(path, "abc123", "kmeans_blobs", cells_a)
    first_bytes = path.read_bytes()

    save_labels(path, "abc123", "kmeans_blobs", cells_b)
    second_bytes = path.read_bytes()

    assert first_bytes != second_bytes

    result = load_labels(path, "abc123", "kmeans_blobs")
    assert result.status is LoadStatus.OK
    _assert_cells_bit_identical(cells_b, result.cells)
