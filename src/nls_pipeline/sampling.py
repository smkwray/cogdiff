from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .transforms import DEFAULT_MISSING_CODES


def deduplicate_people(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[id_col]).copy()


def recode_missing_in_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    missing_codes: set[int] | None = None,
) -> pd.DataFrame:
    codes = DEFAULT_MISSING_CODES if missing_codes is None else missing_codes
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].mask(out[col].isin(codes))
    return out


def require_complete_tests(df: pd.DataFrame, tests: list[str]) -> pd.DataFrame:
    existing = [col for col in tests if col in df.columns]
    if len(existing) != len(tests):
        missing = sorted(set(tests) - set(existing))
        raise ValueError(f"Missing required test columns: {missing}")
    return df.dropna(subset=tests).copy()


def require_min_tests_observed(df: pd.DataFrame, tests: list[str], min_tests: int) -> pd.DataFrame:
    existing = [col for col in tests if col in df.columns]
    if len(existing) != len(tests):
        missing = sorted(set(tests) - set(existing))
        raise ValueError(f"Missing required test columns: {missing}")
    observed = df[tests].notna().sum(axis=1)
    return df.loc[observed >= min_tests].copy()


def filter_age_range(
    df: pd.DataFrame,
    age_col: str,
    age_min: float | int | None,
    age_max: float | int | None,
) -> pd.DataFrame:
    if age_col not in df.columns:
        return df.copy()
    out = df.copy()
    if age_min is not None:
        out = out.loc[out[age_col] >= age_min]
    if age_max is not None:
        out = out.loc[out[age_col] <= age_max]
    return out


def standardize_series(series: pd.Series) -> pd.Series:
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=1)
    if pd.isna(std) or float(std) == 0.0:
        return series * np.nan
    return (series - mean) / std


def build_auto_shop_composite(df: pd.DataFrame, auto_col: str, shop_col: str, out_col: str = "AS") -> pd.DataFrame:
    if auto_col not in df.columns or shop_col not in df.columns:
        return df.copy()
    out = df.copy()
    auto_z = standardize_series(out[auto_col])
    shop_z = standardize_series(out[shop_col])
    out[out_col] = pd.concat([auto_z, shop_z], axis=1).mean(axis=1)
    return out


def harmonize_pos_neg_pairs(
    df: pd.DataFrame,
    pairs: list[dict[str, object]],
    *,
    method: str = "signed_merge",
    missing_codes: set[int] | None = None,
    emit_source_cols: bool = False,
    implied_decimal_places: int | None = None,
    scale_factor: float | None = None,
) -> pd.DataFrame:
    if not pairs:
        return df.copy()

    codes = DEFAULT_MISSING_CODES if missing_codes is None else missing_codes
    out = df.copy()
    normalized_method = str(method).strip().lower()
    supported = {"signed_merge", "zscore_by_branch", "coalesce_raw"}
    if normalized_method not in supported:
        raise ValueError(f"Unsupported harmonization method: {method}. Expected one of {sorted(supported)}")

    def _scale_from_config(raw_scale: object, raw_places: object) -> float | None:
        if raw_scale is not None:
            numeric_scale = pd.to_numeric(pd.Series([raw_scale]), errors="coerce").iloc[0]
            if pd.isna(numeric_scale) or float(numeric_scale) <= 0.0:
                raise ValueError(f"Invalid harmonization scale_factor: {raw_scale}")
            return float(numeric_scale)
        if raw_places is None:
            return None
        parsed_places = pd.to_numeric(pd.Series([raw_places]), errors="coerce").iloc[0]
        if pd.isna(parsed_places):
            raise ValueError(f"Invalid harmonization implied_decimal_places: {raw_places}")
        places = int(parsed_places)
        if places < 0:
            raise ValueError(f"Invalid harmonization implied_decimal_places: {raw_places}")
        return float(10 ** (-places))

    global_scale = _scale_from_config(scale_factor, implied_decimal_places)

    for spec in pairs:
        output_col = str(spec.get("output", "")).strip()
        pos_col = str(spec.get("pos_col", "")).strip()
        neg_col = str(spec.get("neg_col", "")).strip()
        if not output_col or not pos_col or not neg_col:
            raise ValueError(f"Invalid harmonization pair spec: {spec}")

        pos_raw = (
            pd.to_numeric(out[pos_col], errors="coerce")
            if pos_col in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        neg_raw = (
            pd.to_numeric(out[neg_col], errors="coerce")
            if neg_col in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        pos_valid = pos_raw.notna() & ~pos_raw.isin(codes)
        neg_valid = neg_raw.notna() & ~neg_raw.isin(codes)
        pair_scale = _scale_from_config(spec.get("scale_factor"), spec.get("implied_decimal_places"))
        if pair_scale is None:
            pair_scale = global_scale

        if pair_scale is not None:
            pos = pos_raw.where(~pos_valid, pos_raw * pair_scale)
            neg = neg_raw.where(~neg_valid, neg_raw * pair_scale)
        else:
            pos = pos_raw
            neg = neg_raw

        if normalized_method == "zscore_by_branch":
            pos_norm = standardize_series(pos.where(pos_valid))
            neg_norm = standardize_series(neg.where(neg_valid))
            out[output_col] = pd.concat([pos_norm, neg_norm], axis=1).mean(axis=1, skipna=True)
        elif normalized_method == "signed_merge":
            # NLSY97 stores negative theta values in *_NEG because negative codes are reserved
            # for missing values. Reconstruct the signed score before any downstream scaling.
            neg_signed = neg.where(neg <= 0.0, -neg)
            out[output_col] = np.where(pos_valid, pos, np.where(neg_valid, neg_signed, np.nan))
        else:
            out[output_col] = np.where(pos_valid, pos, np.where(neg_valid, neg, np.nan))

        if emit_source_cols:
            source = np.select(
                [pos_valid & ~neg_valid, ~pos_valid & neg_valid, pos_valid & neg_valid],
                ["pos", "neg", "both"],
                default="none",
            )
            out[f"{output_col}_source"] = pd.Series(source, index=out.index, dtype="string")

    return out


@dataclass
class ResidualDiagnostics:
    n_used: int
    r2: float
    beta0: float
    beta1: float
    beta2: float
    resid_sd: float
    outliers_3sd: int


def residualize_quadratic(y: pd.Series, x: pd.Series) -> tuple[pd.Series, ResidualDiagnostics]:
    work = pd.DataFrame({"y": y, "x": x}).dropna().copy()
    residuals = pd.Series(np.nan, index=y.index, dtype=float)
    if work.empty:
        return residuals, ResidualDiagnostics(0, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    xx = work["x"].astype(float)
    design = np.column_stack([np.ones(len(work)), xx.to_numpy(), (xx**2).to_numpy()])
    target = work["y"].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ beta
    resid = target - fitted

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    outliers_3sd = 0 if resid_sd == 0.0 else int(np.sum(np.abs(resid) > (3.0 * resid_sd)))

    residuals.loc[work.index] = resid
    diag = ResidualDiagnostics(
        n_used=len(work),
        r2=r2,
        beta0=float(beta[0]),
        beta1=float(beta[1]),
        beta2=float(beta[2]),
        resid_sd=resid_sd,
        outliers_3sd=outliers_3sd,
    )
    return residuals, diag
