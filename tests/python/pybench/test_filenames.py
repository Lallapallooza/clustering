from __future__ import annotations

from pybench.charts.filenames import chart_filename


def test_chart_filename_without_slug() -> None:
    assert chart_filename("dbscan", {}, "abc123") == "dbscan_abc123.png"


def test_chart_filename_with_slug() -> None:
    name = chart_filename(
        "dbscan", {"eps": 0.5, "min_samples": 5, "n_jobs": 8}, "abc123"
    )
    assert name == "dbscan_eps=0.5+min_samples=5_abc123.png"


def test_chart_filename_ignores_only_excluded_params() -> None:
    assert chart_filename("dbscan", {"n_jobs": 4}, "abc123") == "dbscan_abc123.png"
