from __future__ import annotations

import json
import math
import platform
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pybench.charts.meta import RunMetadata
from pybench.recipe import RunResult

_CANONICAL_ENCODING_VERSION = 1
_LEGACY_ENCODING_VERSION = 0
_UNKNOWN = "unknown"


def _capture_git_sha(repo_root: Path | None) -> str:
    cwd = repo_root if repo_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _UNKNOWN
    if result.returncode != 0:
        return _UNKNOWN
    sha = result.stdout.strip()
    return sha or _UNKNOWN


def _capture_machine() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if platform.system() == "Linux" and cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition(":")
                if sep and key.strip() == "model name":
                    trimmed = value.strip()
                    if trimmed:
                        return trimmed
        except OSError:
            pass
    fallback = platform.processor()
    if fallback:
        return fallback
    return _UNKNOWN


def capture_metadata(repo_root: Path | None = None) -> RunMetadata:
    return RunMetadata(
        timestamp_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_capture_git_sha(repo_root),
        machine=_capture_machine(),
        canonical_encoding_version=_CANONICAL_ENCODING_VERSION,
    )


def _find_non_finite_field(rows: list[dict]) -> tuple[int, str] | None:
    for idx, row in enumerate(rows):
        for field_name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                return (idx, field_name)
    return None


def save_results(path: Path, meta: RunMetadata, results: Sequence[RunResult]) -> None:
    rows = [asdict(r) for r in results]
    payload = {"meta": asdict(meta), "results": rows}
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    except ValueError as exc:
        offender = _find_non_finite_field(rows)
        if offender is not None:
            idx, field_name = offender
            raise ValueError(
                f"non-finite value in results[{idx}].{field_name}; "
                "results.json forbids NaN/Inf"
            ) from exc
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_results(path: Path) -> tuple[RunMetadata, list[RunResult]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        warnings.warn(
            "legacy bare-list results.json; metadata set to 'unknown' sentinels",
            DeprecationWarning,
            stacklevel=2,
        )
        meta = RunMetadata(
            timestamp_iso=_UNKNOWN,
            git_sha=_UNKNOWN,
            machine=_UNKNOWN,
            canonical_encoding_version=_LEGACY_ENCODING_VERSION,
        )
        return meta, [RunResult(**row) for row in data]
    if isinstance(data, dict) and set(data.keys()) == {"meta", "results"}:
        meta = RunMetadata(**data["meta"])
        rows = [RunResult(**row) for row in data["results"]]
        return meta, rows
    raise ValueError("results.json is neither wrapper schema nor legacy bare-list")
