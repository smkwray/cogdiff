#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root

SIBLING_COLUMNS = [
    "outcome",
    "cohort_a",
    "cohort_b",
    "status",
    "reason",
    "beta_within_a",
    "se_within_a",
    "beta_within_b",
    "se_within_b",
    "diff_b_minus_a",
    "z_diff",
    "p_value_diff",
    "source_table",
]

INTERGEN_COLUMNS = [
    "status",
    "reason",
    "n_pairs_bivariate",
    "n_pairs_ses_controlled",
    "beta_mother_g_bivariate",
    "beta_mother_g_ses_controlled",
    "attenuation_abs",
    "attenuation_pct",
    "r2_bivariate",
    "r2_ses_controlled",
    "delta_r2",
    "beta_parent_ed",
    "source_table",
]


def _empty_sibling(outcome: str, cohort_a: str, cohort_b: str, reason: str, source_table: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome": outcome,
        "cohort_a": cohort_a,
        "cohort_b": cohort_b,
        "status": "not_feasible",
        "reason": reason,
        "source_table": source_table,
    }
    for col in SIBLING_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_intergen(reason: str, source_table: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "not_feasible",
        "reason": reason,
        "source_table": source_table,
    }
    for col in INTERGEN_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_family_and_intergenerational_contrasts(
    *,
    root: Path,
    sibling_output_path: Path = Path("outputs/tables/sibling_fe_cross_cohort_contrasts.csv"),
    intergen_output_path: Path = Path("outputs/tables/intergenerational_g_attenuation.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sibling_path = root / "outputs" / "tables" / "sibling_fe_g_outcome.csv"
    intergen_path = root / "outputs" / "tables" / "intergenerational_g_transmission.csv"

    sibling_rows: list[dict[str, Any]] = []
    sibling_source = str(sibling_path.relative_to(root))
    if not sibling_path.exists():
        sibling_rows.append(_empty_sibling("", "nlsy79", "nlsy97", "missing_source_table", sibling_source))
    else:
        frame = pd.read_csv(sibling_path)
        frame = frame.loc[frame["status"].astype("string").str.lower() == "computed"].copy()
        for outcome in sorted(frame["outcome"].dropna().astype(str).unique().tolist()):
            subset = frame.loc[frame["outcome"].astype(str) == outcome].copy()
            a = subset.loc[subset["cohort"].astype(str) == "nlsy79"]
            b = subset.loc[subset["cohort"].astype(str) == "nlsy97"]
            if a.empty or b.empty:
                sibling_rows.append(_empty_sibling(outcome, "nlsy79", "nlsy97", "missing_cohort_rows", sibling_source))
                continue
            a_row = a.iloc[0]
            b_row = b.iloc[0]
            beta_a = pd.to_numeric(pd.Series([a_row.get("beta_g_within")]), errors="coerce").iloc[0]
            beta_b = pd.to_numeric(pd.Series([b_row.get("beta_g_within")]), errors="coerce").iloc[0]
            se_a = pd.to_numeric(pd.Series([a_row.get("se_within")]), errors="coerce").iloc[0]
            se_b = pd.to_numeric(pd.Series([b_row.get("se_within")]), errors="coerce").iloc[0]
            if pd.isna(beta_a) or pd.isna(beta_b) or pd.isna(se_a) or pd.isna(se_b) or float(se_a) <= 0.0 or float(se_b) <= 0.0:
                sibling_rows.append(_empty_sibling(outcome, "nlsy79", "nlsy97", "invalid_effect_or_se", sibling_source))
                continue
            diff = float(beta_b - beta_a)
            se_diff = math.sqrt(float(se_a) ** 2 + float(se_b) ** 2)
            z = diff / se_diff if se_diff > 0.0 else float("nan")
            p = float(2.0 * norm.sf(abs(z))) if math.isfinite(z) else float("nan")
            row = {
                "outcome": outcome,
                "cohort_a": "nlsy79",
                "cohort_b": "nlsy97",
                "status": "computed",
                "reason": pd.NA,
                "beta_within_a": float(beta_a),
                "se_within_a": float(se_a),
                "beta_within_b": float(beta_b),
                "se_within_b": float(se_b),
                "diff_b_minus_a": diff,
                "z_diff": z if math.isfinite(z) else pd.NA,
                "p_value_diff": p if math.isfinite(p) else pd.NA,
                "source_table": sibling_source,
            }
            for col in SIBLING_COLUMNS:
                row.setdefault(col, pd.NA)
            sibling_rows.append(row)

    sibling_out = pd.DataFrame(sibling_rows)
    if sibling_out.empty:
        sibling_out = pd.DataFrame(columns=SIBLING_COLUMNS)
    for col in SIBLING_COLUMNS:
        if col not in sibling_out.columns:
            sibling_out[col] = pd.NA
    sibling_out = sibling_out[SIBLING_COLUMNS].copy()
    sibling_target = sibling_output_path if sibling_output_path.is_absolute() else root / sibling_output_path
    sibling_target.parent.mkdir(parents=True, exist_ok=True)
    sibling_out.to_csv(sibling_target, index=False)

    intergen_rows: list[dict[str, Any]] = []
    intergen_source = str(intergen_path.relative_to(root))
    if not intergen_path.exists():
        intergen_rows.append(_empty_intergen("missing_source_table", intergen_source))
    else:
        frame = pd.read_csv(intergen_path)
        frame = frame.loc[frame["status"].astype("string").str.lower() == "computed"].copy()
        if frame.empty:
            intergen_rows.append(_empty_intergen("no_computed_rows", intergen_source))
        else:
            bi = frame.loc[frame["model"].astype(str) == "bivariate"]
            ses = frame.loc[frame["model"].astype(str) == "ses_controlled"]
            if bi.empty or ses.empty:
                intergen_rows.append(_empty_intergen("missing_required_models", intergen_source))
            else:
                bi_row = bi.iloc[0]
                ses_row = ses.iloc[0]
                beta_bi = pd.to_numeric(pd.Series([bi_row.get("beta_mother_g")]), errors="coerce").iloc[0]
                beta_ses = pd.to_numeric(pd.Series([ses_row.get("beta_mother_g")]), errors="coerce").iloc[0]
                r2_bi = pd.to_numeric(pd.Series([bi_row.get("r2")]), errors="coerce").iloc[0]
                r2_ses = pd.to_numeric(pd.Series([ses_row.get("r2")]), errors="coerce").iloc[0]
                beta_parent_ed = pd.to_numeric(pd.Series([ses_row.get("beta_parent_ed")]), errors="coerce").iloc[0]
                if pd.isna(beta_bi) or pd.isna(beta_ses):
                    intergen_rows.append(_empty_intergen("invalid_beta_rows", intergen_source))
                else:
                    attenuation_abs = float(beta_bi - beta_ses)
                    attenuation_pct = float((attenuation_abs / beta_bi) * 100.0) if float(beta_bi) != 0.0 else float("nan")
                    row = {
                        "status": "computed",
                        "reason": pd.NA,
                        "n_pairs_bivariate": int(pd.to_numeric(pd.Series([bi_row.get("n_pairs")]), errors="coerce").fillna(0).iloc[0]),
                        "n_pairs_ses_controlled": int(pd.to_numeric(pd.Series([ses_row.get("n_pairs")]), errors="coerce").fillna(0).iloc[0]),
                        "beta_mother_g_bivariate": float(beta_bi),
                        "beta_mother_g_ses_controlled": float(beta_ses),
                        "attenuation_abs": attenuation_abs,
                        "attenuation_pct": attenuation_pct if math.isfinite(attenuation_pct) else pd.NA,
                        "r2_bivariate": float(r2_bi) if pd.notna(r2_bi) and math.isfinite(float(r2_bi)) else pd.NA,
                        "r2_ses_controlled": float(r2_ses) if pd.notna(r2_ses) and math.isfinite(float(r2_ses)) else pd.NA,
                        "delta_r2": float(r2_ses - r2_bi) if pd.notna(r2_bi) and pd.notna(r2_ses) and math.isfinite(float(r2_bi)) and math.isfinite(float(r2_ses)) else pd.NA,
                        "beta_parent_ed": float(beta_parent_ed) if pd.notna(beta_parent_ed) and math.isfinite(float(beta_parent_ed)) else pd.NA,
                        "source_table": intergen_source,
                    }
                    for col in INTERGEN_COLUMNS:
                        row.setdefault(col, pd.NA)
                    intergen_rows.append(row)

    intergen_out = pd.DataFrame(intergen_rows)
    if intergen_out.empty:
        intergen_out = pd.DataFrame(columns=INTERGEN_COLUMNS)
    for col in INTERGEN_COLUMNS:
        if col not in intergen_out.columns:
            intergen_out[col] = pd.NA
    intergen_out = intergen_out[INTERGEN_COLUMNS].copy()
    intergen_target = intergen_output_path if intergen_output_path.is_absolute() else root / intergen_output_path
    intergen_target.parent.mkdir(parents=True, exist_ok=True)
    intergen_out.to_csv(intergen_target, index=False)

    return sibling_out, intergen_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build contrasts from sibling FE and intergenerational transmission tables.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--sibling-output-path", type=Path, default=Path("outputs/tables/sibling_fe_cross_cohort_contrasts.csv"))
    parser.add_argument("--intergen-output-path", type=Path, default=Path("outputs/tables/intergenerational_g_attenuation.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        sibling, intergen = run_family_and_intergenerational_contrasts(
            root=root,
            sibling_output_path=args.sibling_output_path,
            intergen_output_path=args.intergen_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.sibling_output_path if args.sibling_output_path.is_absolute() else root / args.sibling_output_path}")
    print(f"[ok] wrote {args.intergen_output_path if args.intergen_output_path.is_absolute() else root / args.intergen_output_path}")
    print(f"[ok] computed sibling contrast rows: {int((sibling['status'] == 'computed').sum()) if 'status' in sibling.columns else 0}")
    print(f"[ok] computed intergen rows: {int((intergen['status'] == 'computed').sum()) if 'status' in intergen.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
