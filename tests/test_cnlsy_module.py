from __future__ import annotations

import numpy as np
import pandas as pd

from nls_pipeline.cnlsy import (
    AGEBIN_SUMMARY_COLUMNS,
    LONGITUDINAL_SUMMARY_COLUMNS,
    build_cnlsy_agebin_summary,
    build_cnlsy_longitudinal_summary,
)


def test_agebin_summary_bin_boundaries_and_columns() -> None:
    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "age": [4, 5, 6, 7, 8, 10, 17, 18, 19],
            "sex": ["F", "F", "M", "M", "F", "M", "F", "M", "M"],
            "g": [100.0, 10.0, 20.0, 30.0, 40.0, 15.0, 50.0, 60.0, 70.0],
        }
    )

    summary = build_cnlsy_agebin_summary(
        df,
        id_col="person_id",
        age_col="age",
        sex_col="sex",
        score_col="g",
        min_age=5,
        max_age=18,
        bin_width=2,
    )

    assert list(summary.columns) == AGEBIN_SUMMARY_COLUMNS
    assert len(summary) == 7
    assert list(summary["age_bin"]) == ["5-6", "7-8", "9-10", "11-12", "13-14", "15-16", "17-18"]

    row_56 = summary.loc[summary["age_bin"] == "5-6"].iloc[0]
    assert row_56["age_min"] == 5
    assert row_56["age_max"] == 6
    assert row_56["n_obs"] == 2
    assert row_56["n_persons"] == 2
    assert row_56["n_male"] == 1
    assert row_56["n_female"] == 1
    assert row_56["male_mean"] == 20.0
    assert row_56["female_mean"] == 10.0

    row_78 = summary.loc[summary["age_bin"] == "7-8"].iloc[0]
    assert row_78["n_obs"] == 2

    row_1718 = summary.loc[summary["age_bin"] == "17-18"].iloc[0]
    assert row_1718["n_obs"] == 2

    row_99 = summary.loc[summary["age_bin"] == "11-12"].iloc[0]
    assert row_99["n_obs"] == 0
    assert np.isnan(row_99["mean_diff"])


def test_longitudinal_summary_handles_repeated_persons() -> None:
    df = pd.DataFrame(
        {
            "person_id": ["a", "a", "a", "b", "b", "b"],
            "age": [5.0, 7.0, 9.0, 5.0, 7.0, 9.0],
            "sex": ["F", "F", "F", "M", "M", "M"],
            "g": [1.0, 3.0, 5.0, 2.0, 5.0, 9.0],
        }
    )

    long_summary = build_cnlsy_longitudinal_summary(
        df,
        id_col="person_id",
        age_col="age",
        sex_col="sex",
        score_col="g",
    )

    assert list(long_summary.columns) == LONGITUDINAL_SUMMARY_COLUMNS
    assert long_summary.loc[0, "n_obs"] == 6
    assert long_summary.loc[0, "n_persons"] == 2
    assert long_summary.loc[0, "n_repeated_persons"] == 2
    assert long_summary.loc[0, "model_type"] == "ols_person_fixed_effect"

    # female slope should be smaller than male slope in this constructed data.
    assert long_summary.loc[0, "female_age_slope"] < long_summary.loc[0, "male_age_slope"]
    assert long_summary.loc[0, "female_age_slope"] > 0.5
    assert long_summary.loc[0, "male_age_slope"] > 1.5

    # should not return placeholder insufficient-data row
    assert long_summary.loc[0, "model_type"] != "insufficient_data"


def test_longitudinal_summary_is_insufficient_when_design_too_large() -> None:
    df = pd.DataFrame(
        {
            "person_id": ["a", "a", "b", "b"],
            "age": [5.0, 7.0, 5.0, 7.0],
            "sex": ["F", "F", "M", "M"],
            "g": [1.0, 3.0, 2.0, 8.0],
        }
    )

    long_summary = build_cnlsy_longitudinal_summary(
        df,
        id_col="person_id",
        age_col="age",
        sex_col="sex",
        score_col="g",
    )

    assert long_summary.loc[0, "model_type"] == "insufficient_data"
    assert long_summary.loc[0, "n_obs"] == 4
    assert long_summary.loc[0, "n_persons"] == 2
    assert long_summary.loc[0, "n_repeated_persons"] == 2
