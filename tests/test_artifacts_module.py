from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from nls_pipeline.artifacts import (
    FIT_INDEX_COLUMNS,
    LATENT_COLUMNS,
    MODINDEX_COLUMNS,
    PARAM_COLUMNS,
    load_fit_csv,
    load_latent_csv,
    load_modindex_csv,
    load_param_csv,
)


def _write_csv(path: Path, columns: tuple[str, ...]) -> Path:
    payload = {col: [1] for col in columns}
    pd.DataFrame(payload).to_csv(path, index=False)
    return path


def _write_csv_with_gap(path: Path, columns: tuple[str, ...], missing: str) -> Path:
    data = {col: [1] for col in columns if col != missing}
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_load_fit_param_latent_modindex_happy_path(tmp_path: Path) -> None:
    fit = _write_csv(tmp_path / "fit_indices.csv", FIT_INDEX_COLUMNS)
    param = _write_csv(tmp_path / "params.csv", PARAM_COLUMNS)
    latent = _write_csv(tmp_path / "latent_summary.csv", LATENT_COLUMNS)
    modindex = _write_csv(tmp_path / "modindices.csv", MODINDEX_COLUMNS)

    assert list(load_fit_csv(fit).columns) == list(FIT_INDEX_COLUMNS)
    assert list(load_param_csv(param).columns) == list(PARAM_COLUMNS)
    assert list(load_latent_csv(latent).columns) == list(LATENT_COLUMNS)
    assert list(load_modindex_csv(modindex).columns) == list(MODINDEX_COLUMNS)


@pytest.mark.parametrize(
    "loader, path, required, missing",
    [
        (load_fit_csv, "fit_indices.csv", FIT_INDEX_COLUMNS, "cfi"),
        (load_param_csv, "params.csv", PARAM_COLUMNS, "std_all"),
        (load_latent_csv, "latent_summary.csv", LATENT_COLUMNS, "sd"),
        (load_modindex_csv, "modindices.csv", MODINDEX_COLUMNS, "epc"),
    ],
)
def test_loaders_raise_on_missing_columns(
    tmp_path: Path, loader: Callable[[Path], object], path: str, required: tuple[str, ...], missing: str
) -> None:
    csv = _write_csv_with_gap(tmp_path / path, required, missing)
    with pytest.raises(ValueError, match=f"missing required columns: .*{missing}"):
        loader(csv)
