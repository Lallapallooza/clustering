from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunMetadata:
    timestamp_iso: str
    git_sha: str
    machine: str
    canonical_encoding_version: int
