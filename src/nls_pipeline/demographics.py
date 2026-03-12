from __future__ import annotations

import pandas as pd


RACE_ETHNICITY_3CAT_LABELS: tuple[str, ...] = (
    "HISPANIC",
    "BLACK",
    "NON-BLACK, NON-HISPANIC",
)


def harmonize_race_ethnicity_3cat(cohort: str, raw: pd.Series) -> pd.Series:
    """Map cohort-specific race/ethnicity codes into a comparable 3-category label.

    Output labels:
      - HISPANIC
      - BLACK
      - NON-BLACK, NON-HISPANIC
    """
    cohort_key = str(cohort).strip().lower()
    raw_text = raw.astype("string")
    raw_num = pd.to_numeric(raw_text, errors="coerce")

    if cohort_key in {"nlsy79", "cnlsy"}:
        mapping = {
            1: "HISPANIC",
            2: "BLACK",
            3: "NON-BLACK, NON-HISPANIC",
        }
    elif cohort_key == "nlsy97":
        mapping = {
            1: "BLACK",
            2: "HISPANIC",
            3: "NON-BLACK, NON-HISPANIC",
            4: "NON-BLACK, NON-HISPANIC",
        }
    else:
        raise ValueError(f"Unsupported cohort for race/ethnicity harmonization: {cohort}")

    out = raw_num.map(mapping).astype("string")

    low = raw_text.str.strip().str.lower()
    if low.notna().any():
        non_black = (
            low.str.contains("non-black", na=False)
            | low.str.contains("non black", na=False)
            | low.str.contains("nonblack", na=False)
            | low.str.contains("white", na=False)
        )
        mixed = low.str.contains("mixed", na=False) | low.str.contains("multi", na=False)
        hisp = low.str.contains("hisp", na=False) | low.str.contains("latino", na=False) | low.str.contains("span", na=False)
        black = low.str.contains("black", na=False)

        fill = pd.Series(pd.NA, index=raw.index, dtype="string")
        fill = fill.mask(non_black, "NON-BLACK, NON-HISPANIC")
        fill = fill.mask(mixed, "NON-BLACK, NON-HISPANIC")
        fill = fill.mask(hisp, "HISPANIC")
        fill = fill.mask((~hisp) & (~non_black) & black, "BLACK")

        out = out.fillna(fill)

    return out


def compute_parent_education(mother: pd.Series, father: pd.Series | None = None) -> pd.Series:
    """Compute a parent-education summary (rowwise mean) requiring at least one non-missing parent value."""
    mother_num = pd.to_numeric(mother, errors="coerce")
    if father is None:
        return mother_num
    father_num = pd.to_numeric(father, errors="coerce")
    return pd.concat([mother_num, father_num], axis=1).mean(axis=1, skipna=True)

