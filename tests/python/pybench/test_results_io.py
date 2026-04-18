from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from pybench.charts.data import payload_hash6
from pybench.charts.meta import RunMetadata
from pybench.charts.results_io import (
    capture_metadata,
    load_results,
    save_results,
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


def _meta(**overrides: Any) -> RunMetadata:
    base: dict[str, Any] = {
        "timestamp_iso": "2026-04-16T12:34:56+00:00",
        "git_sha": "abcdef0",
        "machine": "Test CPU",
        "canonical_encoding_version": 1,
    }
    base.update(overrides)
    return RunMetadata(**base)


def test_capture_metadata_basic_fields() -> None:
    meta = capture_metadata()

    assert isinstance(meta.timestamp_iso, str)
    datetime.fromisoformat(meta.timestamp_iso)

    assert isinstance(meta.git_sha, str)
    assert meta.git_sha == "unknown" or re.fullmatch(r"[0-9a-f]{7}", meta.git_sha)

    assert isinstance(meta.machine, str)
    assert len(meta.machine) > 0

    assert meta.canonical_encoding_version == 1


def test_capture_metadata_handles_git_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_fnf(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("no git")

    monkeypatch.setattr("pybench.charts.results_io.subprocess.run", raise_fnf)
    meta = capture_metadata()
    assert meta.git_sha == "unknown"


def test_capture_metadata_handles_git_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=2)

    monkeypatch.setattr("pybench.charts.results_io.subprocess.run", raise_timeout)
    meta = capture_metadata()
    assert meta.git_sha == "unknown"


def test_capture_metadata_handles_git_nonzero_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr("pybench.charts.results_io.subprocess.run", fake_run)
    meta = capture_metadata()
    assert meta.git_sha == "unknown"


def test_capture_metadata_machine_fallback_when_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pybench.charts.results_io.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "pybench.charts.results_io.platform.processor",
        lambda: "MyFallbackCPU",
    )
    meta = capture_metadata()
    assert meta.machine == "MyFallbackCPU"


def test_capture_metadata_machine_returns_unknown_when_all_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pybench.charts.results_io.platform.system", lambda: "Darwin")
    monkeypatch.setattr("pybench.charts.results_io.platform.processor", lambda: "")
    meta = capture_metadata()
    assert meta.machine == "unknown"


def test_save_load_wrapper_round_trip_preserves_results(tmp_path: Path) -> None:
    results = [
        _make_result(size=1000, dims=2),
        _make_result(
            size=5000,
            dims=8,
            params={"eps": 0.7, "min_samples": 10, "n_jobs": 4},
            ours_median_ms=15.5,
            theirs_median_ms=42.1,
        ),
    ]
    meta = _meta()
    path = tmp_path / "out" / "results.json"

    save_results(path, meta, results)
    loaded_meta, loaded_results = load_results(path)

    assert loaded_meta == meta
    assert len(loaded_results) == len(results)
    for original, loaded in zip(results, loaded_results):
        assert loaded == original


def test_save_results_writes_sort_keys(tmp_path: Path) -> None:
    results = [_make_result()]
    meta = _meta()
    path = tmp_path / "results.json"

    save_results(path, meta, results)
    text = path.read_text(encoding="utf-8")
    # sort_keys=True -> "meta" appears before "results" at the top level.
    assert text.index('"meta"') < text.index('"results"')


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("ari", math.nan), ("speedup", math.inf)],
    ids=["nan", "inf"],
)
def test_save_results_rejects_non_finite(
    tmp_path: Path, field: str, bad_value: float
) -> None:
    results = [_make_result(**{field: bad_value})]
    path = tmp_path / "results.json"
    with pytest.raises(ValueError) as exc:
        save_results(path, _meta(), results)
    assert field in str(exc.value)


def test_load_results_legacy_bare_list_prints_warning_and_fills_unknown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "results.json"
    original = _make_result()
    path.write_text(json.dumps([asdict(original)]), encoding="utf-8")

    meta, loaded = load_results(path)

    err = capsys.readouterr().err
    assert "legacy bare-list" in err
    assert str(path) in err

    assert meta.canonical_encoding_version == 0
    assert meta.timestamp_iso == "unknown"
    assert meta.git_sha == "unknown"
    assert meta.machine == "unknown"

    assert len(loaded) == 1
    assert loaded[0] == original


def test_load_results_legacy_bare_list_rows_hydrate_without_remap(
    tmp_path: Path,
) -> None:
    originals = [
        _make_result(size=1000, params={"eps": 0.5, "min_samples": 5, "n_jobs": 1}),
        _make_result(size=5000, params={"eps": 1.0, "min_samples": 10, "n_jobs": 4}),
    ]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([asdict(r) for r in originals]), encoding="utf-8")

    _, loaded = load_results(path)

    assert loaded == originals


def test_hash_independent_of_meta(tmp_path: Path) -> None:
    results = [
        _make_result(size=1000),
        _make_result(size=5000, ours_median_ms=12.0),
    ]
    meta_a = _meta(git_sha="aaaaaaa", machine="Machine A")
    meta_b = _meta(git_sha="bbbbbbb", machine="Machine B")

    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    save_results(path_a, meta_a, results)
    save_results(path_b, meta_b, results)

    loaded_a = load_results(path_a)[1]
    loaded_b = load_results(path_b)[1]
    assert payload_hash6(loaded_a) == payload_hash6(loaded_b)


def test_load_results_does_not_modify_file_mtime(tmp_path: Path) -> None:
    results = [_make_result()]
    meta = _meta()
    path = tmp_path / "results.json"
    save_results(path, meta, results)

    before = path.stat().st_mtime_ns
    load_results(path)
    after = path.stat().st_mtime_ns

    assert before == after


def test_load_results_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps("just a string"), encoding="utf-8")

    with pytest.raises(ValueError):
        load_results(path)


def test_load_results_rejects_dict_without_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_results(path)


def test_save_results_creates_parent_directory(tmp_path: Path) -> None:
    results = [_make_result()]
    meta = _meta()
    path = tmp_path / "nested" / "dir" / "results.json"

    save_results(path, meta, results)
    assert path.is_file()
