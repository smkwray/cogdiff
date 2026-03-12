from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_MIN_AGE = 5
DEFAULT_MAX_AGE = 18
DEFAULT_AGE_BIN_WIDTH = 2

AGEBIN_SUMMARY_COLUMNS = [
    "age_bin",
    "age_min",
    "age_max",
    "n_obs",
    "n_persons",
    "n_male",
    "n_female",
    "male_mean",
    "female_mean",
    "mean_diff",
    "male_var",
    "female_var",
    "variance_ratio",
]

LONGITUDINAL_SUMMARY_COLUMNS = [
    "model_type",
    "n_obs",
    "n_persons",
    "n_repeated_persons",
    "age_coef",
    "age_se",
    "age_pvalue",
    "sex_coef",
    "sex_se",
    "sex_pvalue",
    "age_x_sex_coef",
    "age_x_sex_se",
    "age_x_sex_pvalue",
    "male_age_slope",
    "female_age_slope",
    "r2",
    "r2_adj",
]


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_sex_code(series: pd.Series) -> pd.Series:
    def _coerce_one(value: object) -> float | np.nan:
        if pd.isna(value):
            return np.nan

        v = str(value).strip().lower()
        # CNLSY-derived files in this repo use the same sex labeling as the broader
        # processed cohort tables: 1/male and 2/female. Keep legacy 0/male support
        # for older fixtures and hand-built test frames.
        if v in {"0", "1", "m", "male", "boy", "man", "men"}:
            return 0.0
        if v in {"2", "f", "female", "girl", "w", "woman"}:
            return 1.0
        return np.nan

    mapped = series.map(_coerce_one)
    return pd.to_numeric(mapped, errors="coerce")


def _age_bins(min_age: int, max_age: int, width: int) -> pd.DataFrame:
    if width <= 0:
        raise ValueError("width must be positive")
    if min_age > max_age:
        raise ValueError("min_age must be <= max_age")

    starts: list[int] = []
    ends: list[int] = []
    for lo in range(min_age, max_age + 1, width):
        starts.append(lo)
        ends.append(min(lo + width - 1, max_age))

    return pd.DataFrame({"age_min": starts, "age_max": ends})


def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0:
        return float("nan")
    return float(num / den)


def _safe_std_int(v: int) -> int:
    return int(v)


def _safe_float(v: float | np.float64 | np.number | None) -> float:
    return float(v) if pd.notna(v) else float("nan")


def _build_age_bin_frame(min_age: int, max_age: int, width: int) -> pd.DataFrame:
    bins = _age_bins(min_age=min_age, max_age=max_age, width=width)
    bins["age_bin"] = bins["age_min"].astype(int).astype(str) + "-" + bins["age_max"].astype(int).astype(str)
    return bins


def build_cnlsy_agebin_summary(
    df: pd.DataFrame,
    *,
    id_col: str = "person_id",
    age_col: str = "age",
    sex_col: str = "sex",
    score_col: str = "g",
    min_age: int = DEFAULT_MIN_AGE,
    max_age: int = DEFAULT_MAX_AGE,
    bin_width: int = DEFAULT_AGE_BIN_WIDTH,
) -> pd.DataFrame:
    """Return cross-sectional summary rows for each age bin."""

    _validate_columns(df, [id_col, age_col, sex_col, score_col])

    required = df[[id_col, age_col, sex_col, score_col]].copy()
    required[age_col] = pd.to_numeric(required[age_col], errors="coerce")
    required[score_col] = pd.to_numeric(required[score_col], errors="coerce")
    required["sex_code"] = _coerce_sex_code(required[sex_col])
    required = required.dropna(subset=[age_col, score_col, "sex_code", id_col]).copy()

    in_range = required[age_col].between(min_age, max_age)
    work = required.loc[in_range].copy()

    bins = _build_age_bin_frame(min_age=min_age, max_age=max_age, width=bin_width)

    if work.empty:
        out = bins.copy()
        out["n_obs"] = 0
        out["n_persons"] = 0
        out["n_male"] = 0
        out["n_female"] = 0
        out["male_mean"] = np.nan
        out["female_mean"] = np.nan
        out["mean_diff"] = np.nan
        out["male_var"] = np.nan
        out["female_var"] = np.nan
        out["variance_ratio"] = np.nan
        return out[AGEBIN_SUMMARY_COLUMNS]

    work["bin_idx"] = np.floor((work[age_col] - min_age) / bin_width).astype(int)
    work = work.loc[work["bin_idx"] >= 0].copy()
    work["age_bin_start"] = min_age + (work["bin_idx"] * bin_width)
    work["age_bin_end"] = (work["age_bin_start"] + bin_width - 1).clip(upper=max_age)

    def _safe_mean(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else float("nan")

    def _safe_var(series: pd.Series) -> float:
        return float(series.var(ddof=1)) if len(series) >= 2 else float("nan")

    rows: list[dict[str, object]] = []
    for _, row in bins.iterrows():
        lo = int(row["age_min"])
        hi = int(row["age_max"])
        subset = work.loc[(work[age_col] >= lo) & (work[age_col] <= hi)]

        male = subset.loc[subset["sex_code"] == 0.0, score_col]
        female = subset.loc[subset["sex_code"] == 1.0, score_col]

        male_mean = _safe_mean(male)
        female_mean = _safe_mean(female)
        male_var = _safe_var(male)
        female_var = _safe_var(female)

        rows.append(
            {
                "age_bin": f"{lo}-{hi}",
                "age_min": lo,
                "age_max": hi,
                "n_obs": _safe_std_int(len(subset)),
                "n_persons": _safe_std_int(subset[id_col].nunique()),
                "n_male": _safe_std_int(len(male)),
                "n_female": _safe_std_int(len(female)),
                "male_mean": male_mean,
                "female_mean": female_mean,
                "mean_diff": female_mean - male_mean if len(male) and len(female) else float("nan"),
                "male_var": male_var,
                "female_var": female_var,
                "variance_ratio": _safe_ratio(female_var, male_var),
            }
        )

    return pd.DataFrame(rows)[AGEBIN_SUMMARY_COLUMNS]


def build_cnlsy_longitudinal_summary(
    df: pd.DataFrame,
    *,
    id_col: str = "person_id",
    age_col: str = "age",
    sex_col: str = "sex",
    score_col: str = "g",
) -> pd.DataFrame:
    """Estimate repeated-measure developmental trend with optional person fixed effects."""

    _validate_columns(df, [id_col, age_col, sex_col, score_col])

    required = df[[id_col, age_col, sex_col, score_col]].copy()
    required[age_col] = pd.to_numeric(required[age_col], errors="coerce")
    required[score_col] = pd.to_numeric(required[score_col], errors="coerce")
    required["sex_code"] = _coerce_sex_code(required[sex_col])
    required = required.dropna(subset=[id_col, age_col, score_col, "sex_code"]).copy()

    n_obs = len(required)
    n_persons = int(required[id_col].nunique()) if n_obs else 0
    n_repeated_persons = int((required.groupby(id_col).size() >= 2).sum()) if n_obs else 0

    base_row = dict(
        model_type="insufficient_data",
        n_obs=n_obs,
        n_persons=n_persons,
        n_repeated_persons=n_repeated_persons,
        age_coef=np.nan,
        age_se=np.nan,
        age_pvalue=np.nan,
        sex_coef=np.nan,
        sex_se=np.nan,
        sex_pvalue=np.nan,
        age_x_sex_coef=np.nan,
        age_x_sex_se=np.nan,
        age_x_sex_pvalue=np.nan,
        male_age_slope=np.nan,
        female_age_slope=np.nan,
        r2=np.nan,
        r2_adj=np.nan,
    )

    if n_obs < 4 or n_persons < 2:
        return pd.DataFrame([base_row], columns=LONGITUDINAL_SUMMARY_COLUMNS)

    x = pd.DataFrame(
        {
            "age": required[age_col].astype(float),
            "sex": required["sex_code"].astype(float),
        }
    )
    x["age_x_sex"] = x["age"] * x["sex"]

    use_person_fe = n_repeated_persons >= 1 and n_persons > 1
    if use_person_fe:
        dummies = pd.get_dummies(required[id_col].astype(str), drop_first=True, dtype=float)
        x = pd.concat([x, dummies], axis=1)

    x = sm.add_constant(x)
    y = required[score_col].astype(float)
    if len(y) <= x.shape[1]:
        return pd.DataFrame([base_row], columns=LONGITUDINAL_SUMMARY_COLUMNS)

    try:
        fit = sm.OLS(y, x).fit()
    except np.linalg.LinAlgError:
        return pd.DataFrame([base_row], columns=LONGITUDINAL_SUMMARY_COLUMNS)

    coef = fit.params
    se = fit.bse
    p = fit.pvalues

    r2 = fit.rsquared if np.isfinite(fit.rsquared) else float("nan")
    r2_adj = fit.rsquared_adj if np.isfinite(fit.rsquared_adj) else float("nan")

    age_coef = coef.get("age", np.nan)
    age_x_sex_coef = coef.get("age_x_sex", np.nan)

    return pd.DataFrame(
        [
            {
                **base_row,
                "model_type": "ols_person_fixed_effect" if use_person_fe else "ols",
                "age_coef": _safe_float(age_coef),
                "age_se": _safe_float(se.get("age", np.nan)),
                "age_pvalue": _safe_float(p.get("age", np.nan)),
                "sex_coef": _safe_float(coef.get("sex", np.nan)),
                "sex_se": _safe_float(se.get("sex", np.nan)),
                "sex_pvalue": _safe_float(p.get("sex", np.nan)),
                "age_x_sex_coef": _safe_float(age_x_sex_coef),
                "age_x_sex_se": _safe_float(se.get("age_x_sex", np.nan)),
                "age_x_sex_pvalue": _safe_float(p.get("age_x_sex", np.nan)),
                "male_age_slope": _safe_float(age_coef),
                "female_age_slope": _safe_float(age_coef + age_x_sex_coef),
                "r2": _safe_float(r2),
                "r2_adj": _safe_float(r2_adj),
            }
        ],
        columns=LONGITUDINAL_SUMMARY_COLUMNS,
    )
