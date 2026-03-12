from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_variable_map_from_header(csv_path: Path) -> pd.DataFrame:
    """Build a minimal variable map from CSV header names.

    This is a bootstrap implementation for Wave 1. It is upgraded later by
    parsing full cohort codebook assets.
    """
    header = pd.read_csv(csv_path, nrows=0)
    return pd.DataFrame(
        {
            "refnum": header.columns,
            "question_name": header.columns,
            "title": header.columns,
            "survey_year": None,
            "type": "unknown",
        }
    )
