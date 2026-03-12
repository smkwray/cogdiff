from __future__ import annotations

import numpy as np
import pandas as pd

from nls_pipeline.sampling import (
    build_auto_shop_composite,
    harmonize_pos_neg_pairs,
    require_complete_tests,
    require_min_tests_observed,
    residualize_quadratic,
)


def test_require_complete_and_min_tests() -> None:
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [2.0, 4.0, np.nan],
            "C": [5.0, 6.0, 7.0],
        }
    )
    complete = require_complete_tests(df, ["A", "B", "C"])
    assert len(complete) == 1

    min_two = require_min_tests_observed(df, ["A", "B", "C"], min_tests=2)
    assert len(min_two) == 3


def test_build_auto_shop_composite_creates_as() -> None:
    df = pd.DataFrame({"AUTO": [10.0, 11.0, 12.0], "SHOP": [7.0, 8.0, 9.0]})
    out = build_auto_shop_composite(df, auto_col="AUTO", shop_col="SHOP")
    assert "AS" in out.columns
    assert out["AS"].notna().all()


def test_harmonize_pos_neg_pairs_branch_normalized() -> None:
    df = pd.DataFrame(
        {
            "A_POS": [10.0, 20.0, -4.0, np.nan],
            "A_NEG": [-4.0, -4.0, 110.0, 120.0],
        }
    )
    out = harmonize_pos_neg_pairs(
        df,
        [{"output": "A", "pos_col": "A_POS", "neg_col": "A_NEG"}],
        missing_codes={-4},
        method="zscore_by_branch",
        emit_source_cols=True,
    )
    assert "A" in out.columns
    assert "A_source" in out.columns
    # first two rows sourced from positive branch, last two from negative branch
    assert out.loc[0, "A_source"] == "pos"
    assert out.loc[1, "A_source"] == "pos"
    assert out.loc[2, "A_source"] == "neg"
    assert out.loc[3, "A_source"] == "neg"
    # zscore-by-branch keeps ordering within each branch
    assert float(out.loc[1, "A"]) > float(out.loc[0, "A"])
    assert float(out.loc[3, "A"]) > float(out.loc[2, "A"])


def test_harmonize_pos_neg_pairs_signed_merge() -> None:
    df = pd.DataFrame(
        {
            "A_POS": [10.0, -4.0, -4.0, np.nan, -4.0],
            "A_NEG": [-4.0, 110.0, -120.0, 90.0, -4.0],
        }
    )
    out = harmonize_pos_neg_pairs(
        df,
        [{"output": "A", "pos_col": "A_POS", "neg_col": "A_NEG"}],
        missing_codes={-4},
        method="signed_merge",
        emit_source_cols=True,
    )
    assert out.loc[0, "A"] == 10.0
    assert out.loc[1, "A"] == -110.0
    assert out.loc[2, "A"] == -120.0
    assert out.loc[3, "A"] == -90.0
    assert pd.isna(out.loc[4, "A"])
    assert out.loc[0, "A_source"] == "pos"
    assert out.loc[1, "A_source"] == "neg"
    assert out.loc[2, "A_source"] == "neg"
    assert out.loc[3, "A_source"] == "neg"


def test_harmonize_pos_neg_pairs_signed_merge_with_implied_decimals() -> None:
    df = pd.DataFrame(
        {
            "A_POS": [1234.0, -4.0],
            "A_NEG": [-4.0, 987.0],
            "B_POS": [2500.0, -4.0],
            "B_NEG": [-4.0, 500.0],
        }
    )
    out = harmonize_pos_neg_pairs(
        df,
        [
            {"output": "A", "pos_col": "A_POS", "neg_col": "A_NEG"},
            {"output": "B", "pos_col": "B_POS", "neg_col": "B_NEG", "implied_decimal_places": 2},
        ],
        missing_codes={-4},
        method="signed_merge",
        implied_decimal_places=3,
    )
    assert np.isclose(float(out.loc[0, "A"]), 1.234)
    assert np.isclose(float(out.loc[1, "A"]), -0.987)
    assert np.isclose(float(out.loc[0, "B"]), 25.0)
    assert np.isclose(float(out.loc[1, "B"]), -5.0)


def test_residualize_quadratic_exact_fit() -> None:
    x = pd.Series(np.arange(1.0, 7.0))
    y = 1.0 + 2.0 * x + 0.5 * (x**2)
    resid, diag = residualize_quadratic(y, x)
    assert np.allclose(resid.dropna().to_numpy(), 0.0, atol=1e-10)
    assert np.isclose(diag.r2, 1.0, atol=1e-10)
