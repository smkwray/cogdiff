from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nls_pipeline.links import (
    DEFAULT_LINKS_REQUIRED_COLUMNS,
    filter_links_by_relationship,
    load_links_csv,
    normalize_family_pairs,
    validate_links_schema,
)


def test_load_links_csv_supports_expanded_schema(tmp_path: Path) -> None:
    links = pd.DataFrame(
        {
            "SubjectTag": [101, 102],
            "PartnerTag": [201, 202],
            "R": [0.5, 0.5],
            "RelationshipPath": ["Gen1Housemates", "Gen2Siblings"],
            "ExtendedID": ["H1", "H2"],
            "ExtraColumn": ["x", "y"],
        }
    )
    path = tmp_path / "links.csv"
    links.to_csv(path, index=False)

    loaded = load_links_csv(path)
    assert set(DEFAULT_LINKS_REQUIRED_COLUMNS).issubset(set(loaded.columns))
    assert "ExtraColumn" in loaded.columns


def test_validate_links_schema_errors_when_required_columns_missing() -> None:
    links = pd.DataFrame(
        {
            "SubjectTag": [101],
            "PartnerTag": [201],
            "R": [0.5],
        }
    )
    with pytest.raises(ValueError, match="Missing required links columns: \\['RelationshipPath'\\]"):
        validate_links_schema(links)


def test_filter_links_by_relationship_path_and_r(tmp_path: Path) -> None:
    links = pd.DataFrame(
        {
            "SubjectTag": [101, 102, 103, 104],
            "PartnerTag": [201, 202, 203, 204],
            "R": [0.5, 0.25, 0.5, 0.5],
            "RelationshipPath": ["Gen1Housemates", "Gen1Housemates", "Gen2Siblings", "Gen1Housemates"],
        }
    )
    path = tmp_path / "links.csv"
    links.to_csv(path, index=False)

    loaded = load_links_csv(path)
    filtered = filter_links_by_relationship(loaded, relatedness_r=0.5, relationship_path="Gen1Housemates")
    assert len(filtered) == 2
    assert set(zip(filtered["SubjectTag"], filtered["PartnerTag"])) == {(101, 201), (104, 204)}


def test_filter_links_by_relationship_any_path_only_restricts_r(tmp_path: Path) -> None:
    links = pd.DataFrame(
        {
            "SubjectTag": [101, 102, 103],
            "PartnerTag": [201, 202, 203],
            "R": [0.5, 0.5, 0.25],
            "RelationshipPath": ["Gen1Housemates", "Gen2Siblings", "Gen1Housemates"],
        }
    )
    path = tmp_path / "links.csv"
    links.to_csv(path, index=False)

    loaded = load_links_csv(path)
    filtered = filter_links_by_relationship(loaded, relatedness_r=0.5, relationship_path="any")
    assert len(filtered) == 2


def test_filter_links_by_relationship_allows_custom_column_names() -> None:
    links = pd.DataFrame(
        {
            "SubjectTag": [101, 102],
            "PartnerTag": [201, 202],
            "sub_id": [101, 102],
            "coef": [0.5, 0.5],
            "rel_path": ["Gen1Housemates", "Gen1Siblings"],
        }
    )

    filtered = filter_links_by_relationship(
        links,
        relatedness_r=0.5,
        relationship_path="Gen1Housemates",
        r_col="coef",
        relationship_col="rel_path",
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["sub_id"] == 101


def test_normalize_family_pairs_orders_and_dedupes_pairs(tmp_path: Path) -> None:
    links = pd.DataFrame(
        {
            "SubjectTag": [2, 1, 4, 4, 6],
            "PartnerTag": [1, 2, 3, 5, 6],
            "R": [0.5, 0.5, 0.5, 0.5, 0.5],
            "RelationshipPath": ["Gen1Housemates"] * 5,
            "ExtendedID": ["E1", "E1", "E2", None, "E3"],
            "MPUBID": [None, None, None, "M1", None],
        }
    )
    path = tmp_path / "links.csv"
    links.to_csv(path, index=False)

    loaded = load_links_csv(path)
    normalized = normalize_family_pairs(loaded)

    assert set(normalized["pair_id"]) == {"1|2", "3|4", "4|5"}
    assert set(normalized["family_id"]) == {"E1", "E2", "M1"}
