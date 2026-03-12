"""Plot helpers for pipeline output figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


FOREST_FIGSIZE = (8.0, 4.0)
LINE_FIGSIZE = (8.0, 4.0)
BAR_FIGSIZE = (8.0, 4.0)
DEFAULT_DPI = 300


def _validate_frame(data: pd.DataFrame, required: Iterable[str], min_rows: int, kind: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{kind} plot requires a pandas DataFrame, got {type(data)!r}.")
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"{kind} plot missing required columns: {', '.join(missing)}.")
    cleaned = data.dropna(subset=required).copy()
    if len(cleaned) < min_rows:
        raise ValueError(
            f"{kind} plot requires at least {min_rows} row(s) of complete data, got {len(cleaned)}."
        )
    return cleaned


def _validate_numeric(frame: pd.DataFrame, columns: Iterable[str], kind: str) -> pd.DataFrame:
    converted = frame.copy()
    for col in columns:
        series = pd.to_numeric(converted[col], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"{kind} plot column {col!r} must be numeric with no missing values "
                "(or values castable to numeric)."
            )
        converted[col] = series
    return converted


def _ensure_png(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.suffix.lower() == "":
        path = path.with_suffix(".png")
    elif path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_and_return(fig: Figure, output_path: str | Path, dpi: int) -> Path:
    path = _ensure_png(output_path)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path


def save_forest_plot(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    label_col: str = "label",
    estimate_col: str = "estimate",
    lower_col: str = "ci_lower",
    upper_col: str = "ci_upper",
    title: str = "Forest plot",
    xlabel: str = "Estimate",
    figsize: tuple[float, float] = FOREST_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Save a horizontal forest plot.
    """
    frame = _validate_frame(data, [label_col, estimate_col, lower_col, upper_col], min_rows=1, kind="forest")
    frame = _validate_numeric(frame, [estimate_col, lower_col, upper_col], kind="forest")
    frame = frame.sort_values(estimate_col).reset_index(drop=True)

    y_pos = list(range(len(frame)))
    fig, ax = plt.subplots(figsize=figsize)
    xerr = frame[[lower_col, upper_col]].copy()
    ci_left = frame[estimate_col] - xerr[lower_col]
    ci_right = xerr[upper_col] - frame[estimate_col]
    ax.errorbar(
        frame[estimate_col],
        y_pos,
        xerr=[ci_left, ci_right],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frame[label_col])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    return _write_and_return(fig, output_path, dpi=dpi)


def save_line_plot(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    x_col: str = "x",
    y_col: str = "y",
    title: str = "Line plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: tuple[float, float] = LINE_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Save a line plot from sequential x/y values.
    """
    frame = _validate_frame(data, [x_col, y_col], min_rows=2, kind="line")
    frame = _validate_numeric(frame, [x_col, y_col], kind="line").sort_values(x_col)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame[x_col], frame[y_col], marker="o", linewidth=1.8, color="#2ca02c")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _write_and_return(fig, output_path, dpi=dpi)


def save_bar_plot(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    category_col: str = "category",
    value_col: str = "value",
    title: str = "Bar plot",
    xlabel: str = "Category",
    ylabel: str = "Value",
    figsize: tuple[float, float] = BAR_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    yerr_col: str | None = None,
) -> Path:
    """
    Save a bar plot with optional y-error bars.
    """
    frame = _validate_frame(data, [category_col, value_col], min_rows=1, kind="bar")
    frame = _validate_numeric(frame, [value_col], kind="bar")
    fig, ax = plt.subplots(figsize=figsize)
    if yerr_col is not None:
        if yerr_col not in frame.columns:
            raise ValueError(f"bar plot missing optional error bar column: {yerr_col}.")
        yerr = pd.to_numeric(frame[yerr_col], errors="coerce")
        if yerr.isna().any():
            raise ValueError(f"bar plot column {yerr_col!r} must be numeric with no missing values.")
        ax.bar(frame[category_col], frame[value_col], yerr=yerr, capsize=5, color="#ff7f0e")
    else:
        ax.bar(frame[category_col], frame[value_col], color="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    return _write_and_return(fig, output_path, dpi=dpi)
