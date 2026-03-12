from __future__ import annotations

import pandas as pd


DEFAULT_MISSING_CODES = {-1, -2, -3, -4, -5}


def recode_missing(series: pd.Series, missing_codes: set[int] | None = None) -> pd.Series:
    codes = DEFAULT_MISSING_CODES if missing_codes is None else missing_codes
    return series.mask(series.isin(codes))


def combine_pos_neg(pos: pd.Series, neg: pd.Series, missing_codes: set[int] | None = None) -> pd.Series:
    codes = DEFAULT_MISSING_CODES if missing_codes is None else missing_codes
    pos_valid = ~pos.isin(codes)
    neg_valid = ~neg.isin(codes)

    out = pos.where(pos_valid)
    return out.fillna(-neg.where(neg_valid))
