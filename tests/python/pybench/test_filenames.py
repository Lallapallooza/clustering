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


def test_chart_filename_default_suffix_is_byte_identical() -> None:
    """@c suffix="" (default) must produce the byte-exact name every current caller expects."""
    # Empty slug case.
    assert (
        chart_filename("dbscan", {}, "abc123", suffix="")
        == chart_filename("dbscan", {}, "abc123")
        == "dbscan_abc123.png"
    )
    # Non-empty slug case.
    params = {"eps": 0.5, "min_samples": 5, "n_jobs": 8}
    assert (
        chart_filename("dbscan", params, "abc123", suffix="")
        == chart_filename("dbscan", params, "abc123")
        == "dbscan_eps=0.5+min_samples=5_abc123.png"
    )


def test_chart_filename_vis_suffix_with_slug() -> None:
    name = chart_filename(
        "dbscan",
        {"eps": 0.5, "min_samples": 5, "n_jobs": 8},
        "abc123",
        suffix="vis",
    )
    assert name == "dbscan_eps=0.5+min_samples=5_vis_abc123.png"


def test_chart_filename_vis_suffix_without_slug() -> None:
    name = chart_filename("dbscan", {}, "abc123", suffix="vis")
    assert name == "dbscan_vis_abc123.png"


def test_chart_filename_vis_and_default_do_not_collide() -> None:
    """The suffix shift guarantees bench + vis PNGs coexist on disk."""
    params = {"eps": 0.5, "min_samples": 5, "n_jobs": 8}
    default_name = chart_filename("dbscan", params, "abc123")
    vis_name = chart_filename("dbscan", params, "abc123", suffix="vis")
    assert default_name != vis_name
    # Same guarantee in the empty-slug branch.
    assert chart_filename("dbscan", {}, "abc123") != chart_filename(
        "dbscan", {}, "abc123", suffix="vis"
    )
