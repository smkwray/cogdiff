from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return "unknown"


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean = vals.mean(skipna=True)
    sd = vals.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="float64")
    return (vals - mean) / sd


def g_proxy(df: pd.DataFrame, indicators: list[str]) -> pd.Series:
    existing = [col for col in indicators if col in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    z = pd.DataFrame({col: zscore(df[col]) for col in existing}, index=df.index)
    return z.mean(axis=1, skipna=False)


def factor_composites(df: pd.DataFrame, factor_map: dict[str, list[str]]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for factor, indicators in factor_map.items():
        existing = [col for col in indicators if col in df.columns]
        if not existing:
            continue
        z = pd.DataFrame({col: zscore(df[col]) for col in existing}, index=df.index)
        out[factor] = z.mean(axis=1, skipna=False)
    return out


def pick_col(df: pd.DataFrame, candidates: tuple[str, ...] | list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return str(col)
    return None


def safe_corr(x: pd.Series, y: pd.Series, *, method: str = "pearson") -> float | None:
    xnum = pd.to_numeric(x, errors="coerce")
    ynum = pd.to_numeric(y, errors="coerce")
    mask = xnum.notna() & ynum.notna()
    if int(mask.sum()) < 3:
        return None
    xv = xnum[mask]
    yv = ynum[mask]
    if float(xv.std(ddof=1)) <= 0.0 or float(yv.std(ddof=1)) <= 0.0:
        return None
    return float(xv.corr(yv, method=method))


def ols_fit(y: pd.Series, x: pd.DataFrame) -> tuple[dict[str, Any] | None, str | None]:
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = x.apply(pd.to_numeric, errors="coerce")
    mask = y_num.notna() & x_num.notna().all(axis=1)
    if int(mask.sum()) == 0:
        return None, "empty_after_numeric_cast"

    yv = y_num[mask].to_numpy(dtype=float)
    xv = x_num[mask].to_numpy(dtype=float)
    n, p = xv.shape
    if n <= p:
        return None, "insufficient_rows_for_model"

    try:
        beta, _, _, _ = np.linalg.lstsq(xv, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None, "ols_singular_matrix"

    yhat = xv @ beta
    resid = yv - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
    dof = n - p
    if dof <= 0:
        return None, "nonpositive_residual_dof"

    sigma2 = sse / float(dof)
    xtx = xv.T @ xv
    try:
        xtx_inv = np.linalg.pinv(xtx)
    except np.linalg.LinAlgError:
        return None, "ols_xtx_pinv_failed"

    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    for i in range(p):
        if math.isfinite(float(t_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * t.sf(abs(float(t_stats[i])), dof))

    r2 = np.nan
    if sst > 0.0 and math.isfinite(sse):
        r2 = float(1.0 - (sse / sst))

    return {
        "beta": beta,
        "se": se,
        "p": p_vals,
        "r2": r2,
        "n_used": int(n),
    }, None


def build_ses_bins(values: pd.Series) -> tuple[pd.Series, str, int]:
    as_num = pd.to_numeric(values, errors="coerce")
    if int(as_num.notna().sum()) >= 12:
        try:
            bins = pd.qcut(as_num, q=3, labels=["low", "mid", "high"], duplicates="drop")
            bins = bins.astype("string")
            unique_bins = int(bins.dropna().nunique())
            if unique_bins >= 2:
                return bins, "numeric_terciles", unique_bins
        except ValueError:
            pass

    cat = values.astype("string").str.strip()
    cat = cat.mask(cat.isin({"", "NA", "NaN", "nan", "unknown", "Unknown"}))
    counts = cat.value_counts(dropna=True)
    keep = counts.head(3).index.tolist()
    binned = cat.where(cat.isin(keep))
    unique_bins = int(binned.dropna().nunique())
    return binned, "categorical_top_levels", unique_bins
