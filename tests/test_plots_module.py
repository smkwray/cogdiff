"""Tests for plot helper utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nls_pipeline.plots import save_bar_plot, save_forest_plot, save_line_plot


def _assert_png_written(path: Path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_save_forest_plot_writes_non_empty_png(tmp_path: Path) -> None:
    data = pd.DataFrame(
        [
            {"label": "A", "estimate": 0.10, "ci_lower": 0.00, "ci_upper": 0.20},
            {"label": "B", "estimate": -0.05, "ci_lower": -0.20, "ci_upper": 0.10},
        ]
    )
    out_path = tmp_path / "forest_test"
    returned_path = save_forest_plot(data, out_path, xlabel="Std. diff")
    _assert_png_written(returned_path)
    assert returned_path == (tmp_path / "forest_test.png")


def test_save_line_plot_writes_non_empty_png(tmp_path: Path) -> None:
    data = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.15]})
    out_path = tmp_path / "line_test.png"
    returned_path = save_line_plot(data, out_path, xlabel="Age", ylabel="Estimate")
    _assert_png_written(returned_path)
    assert returned_path.name == "line_test.png"


def test_save_bar_plot_writes_non_empty_png(tmp_path: Path) -> None:
    data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1.0, 2.5, 0.5]})
    returned_path = save_bar_plot(data, tmp_path / "bar_plot_output", title="Cohorts")
    _assert_png_written(returned_path)


def test_plot_helpers_reject_empty_dataframe() -> None:
    empty = pd.DataFrame(columns=["x", "y"])
    with pytest.raises(ValueError, match="line plot requires at least 2"):
        save_line_plot(empty, "/tmp/ignore.png")


def test_plot_helpers_reject_insufficient_points() -> None:
    one_row = pd.DataFrame({"x": [1.0], "y": [0.2]})
    with pytest.raises(ValueError, match="line plot requires at least 2"):
        save_line_plot(one_row, "/tmp/ignore.png")
