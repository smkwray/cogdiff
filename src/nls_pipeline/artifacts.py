"""Helpers for canonical SEM artifact file loading and schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

FitColumns = tuple[str, ...]
CsvPath = str | Path

FIT_INDEX_COLUMNS: FitColumns = (
    "cohort",
    "model_step",
    "cfi",
    "tli",
    "rmsea",
    "srmr",
    "chisq_scaled",
    "df",
    "aic",
    "bic",
)

PARAM_COLUMNS: FitColumns = (
    "cohort",
    "model_step",
    "group",
    "lhs",
    "op",
    "rhs",
    "est",
    "se",
    "z",
    "p",
    "std_all",
)

LATENT_COLUMNS: FitColumns = (
    "cohort",
    "group",
    "factor",
    "mean",
    "var",
    "sd",
)

MODINDEX_COLUMNS: FitColumns = (
    "cohort",
    "model_step",
    "lhs",
    "op",
    "rhs",
    "mi",
    "epc",
)


def assert_sem_csv_schema(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    path: CsvPath | None = None,
    artifact: str | None = None,
) -> None:
    """Raise if required columns are missing from a SEM artifact DataFrame."""

    required = list(required_columns)
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return

    source = artifact or "SEM artifact"
    if path is not None:
        source = f"{source} ({Path(path)})"
    missing_txt = ", ".join(missing)
    raise ValueError(f"{source}: missing required columns: {missing_txt}")


def _load_sem_csv(path: CsvPath, required_columns: Iterable[str], artifact: str) -> pd.DataFrame:
    path_obj = Path(path)
    df = pd.read_csv(path_obj)
    assert_sem_csv_schema(df, required_columns, path=path_obj, artifact=artifact)
    return df


def load_fit_csv(path: CsvPath) -> pd.DataFrame:
    """Load `fit_indices.csv`."""

    return _load_sem_csv(path, FIT_INDEX_COLUMNS, "fit_indices.csv")


def load_param_csv(path: CsvPath) -> pd.DataFrame:
    """Load `params.csv`."""

    return _load_sem_csv(path, PARAM_COLUMNS, "params.csv")


def load_latent_csv(path: CsvPath) -> pd.DataFrame:
    """Load `latent_summary.csv`."""

    return _load_sem_csv(path, LATENT_COLUMNS, "latent_summary.csv")


def load_modindex_csv(path: CsvPath) -> pd.DataFrame:
    """Load `modindices.csv` (alias-friendly `modindex.csv`)."""

    return _load_sem_csv(path, MODINDEX_COLUMNS, "modindices.csv")


load_params_csv = load_param_csv
load_latent_summary_csv = load_latent_csv
load_modindices_csv = load_modindex_csv
