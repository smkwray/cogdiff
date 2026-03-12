from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_LINKS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "SubjectTag",
    "PartnerTag",
    "R",
    "RelationshipPath",
)
DEFAULT_FAMILY_ID_COLUMNS: tuple[str, ...] = ("ExtendedID", "MPUBID")


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    required = list(columns)
    return [c for c in required if c not in df.columns]


def validate_links_schema(df: pd.DataFrame, required_columns: Iterable[str] = DEFAULT_LINKS_REQUIRED_COLUMNS) -> None:
    """
    Validate a links dataframe has the minimum required columns.

    Raises
    ------
    ValueError
        If required columns are missing, with a clear list of missing names.
    """
    missing = _missing_columns(df, required_columns)
    if missing:
        raise ValueError(f"Missing required links columns: {missing}")


def load_links_csv(path: str | Path, required_columns: Iterable[str] = DEFAULT_LINKS_REQUIRED_COLUMNS) -> pd.DataFrame:
    links = pd.read_csv(Path(path))
    validate_links_schema(links, required_columns=required_columns)
    return links


def filter_links_by_relationship(
    links: pd.DataFrame,
    relatedness_r: float | None = 0.5,
    relationship_path: str | None = "any",
    r_col: str = "R",
    relationship_col: str = "RelationshipPath",
) -> pd.DataFrame:
    """
    Return links matching relationship criteria.

    - If `relatedness_r` is not None, keep rows with numeric R equal to that value.
    - If `relationship_path` is not "any", keep only rows with that relationship path.
    """
    required_columns = [
        r_col if column == "R" else relationship_col if column == "RelationshipPath" else column
        for column in DEFAULT_LINKS_REQUIRED_COLUMNS
    ]
    if relatedness_r is None:
        required_columns = [column for column in required_columns if column != r_col]
    if relationship_path is None or str(relationship_path).strip().lower() == "any":
        required_columns = [column for column in required_columns if column != relationship_col]
    # Preserve stable validation behavior while avoiding false failures if aliases are used.
    required_columns = list(dict.fromkeys(required_columns))

    validate_links_schema(links, required_columns=required_columns)
    if links.empty:
        return links.copy()

    mask = pd.Series(True, index=links.index)
    if relatedness_r is not None:
        r_series = pd.to_numeric(links[r_col], errors="coerce")
        mask &= pd.Series(
            np.isclose(r_series, relatedness_r, rtol=1e-8, atol=1e-8),
            index=links.index,
        )

    if relationship_path is not None and str(relationship_path).strip().lower() != "any":
        target = str(relationship_path).strip().lower()
        mask &= links[relationship_col].astype("string").str.lower().str.strip().eq(target)

    return links.loc[mask].copy()


def _normalize_subject_partner_pair(subject: object, partner: object, pair_sep: str) -> tuple[str, str, str]:
    subject_txt = "" if pd.isna(subject) else str(subject).strip()
    partner_txt = "" if pd.isna(partner) else str(partner).strip()
    if not subject_txt or not partner_txt:
        return ("", "", "")

    first, second = (subject_txt, partner_txt) if subject_txt <= partner_txt else (partner_txt, subject_txt)
    return (first, second, f"{first}{pair_sep}{second}")


def normalize_family_pairs(
    links: pd.DataFrame,
    subject_col: str = "SubjectTag",
    partner_col: str = "PartnerTag",
    pair_id_col: str = "pair_id",
    family_id_col: str = "family_id",
    family_id_columns: Iterable[str] = DEFAULT_FAMILY_ID_COLUMNS,
    pair_sep: str = "|",
    dedupe_pairs: bool = True,
    drop_self_pairs: bool = True,
) -> pd.DataFrame:
    """
    Normalize subject/partner ordering and provide family-level IDs for links.
    Returns a copy and, when requested, removes duplicate unordered pairs.
    """
    validate_links_schema(links, required_columns=(subject_col, partner_col))

    out = links.copy()
    normalized = [_normalize_subject_partner_pair(a, b, pair_sep) for a, b in zip(out[subject_col], out[partner_col])]
    out[[subject_col, partner_col, pair_id_col]] = pd.DataFrame(
        normalized,
        index=out.index,
        columns=[subject_col, partner_col, pair_id_col],
    )
    out = out.loc[(out[pair_id_col] != "")].copy()

    if drop_self_pairs:
        out = out.loc[out[subject_col] != out[partner_col]].copy()

    family = pd.Series("", index=out.index, dtype="string")
    for family_col in family_id_columns:
        if family_col in out.columns:
            available = out[family_col].astype("string").str.strip()
            family = family.mask((family == "") & (available.notna()) & (available != ""), available)

    out[family_id_col] = family
    if dedupe_pairs:
        drop_keys = [pair_id_col, family_id_col]
        out = out.drop_duplicates(subset=drop_keys).copy()

    return out
