from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_export_links_script_normalizes_pairs_and_names_output(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "links_interim_dir: data/interim/links\n")
    _write(
        root / "config/nlsy79.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen1Housemates\n",
    )

    input_path = root / "data/interim/links/links79_pair_expanded.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SubjectTag": 2,
                "PartnerTag": 1,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "ExtendedID": "FAM1",
                "MPUBID": None,
            },
            {
                "SubjectTag": 1,
                "PartnerTag": 2,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "ExtendedID": "FAM1",
                "MPUBID": None,
            },
            {
                "SubjectTag": 10,
                "PartnerTag": 20,
                "R": 0.25,
                "RelationshipPath": "Gen1Housemates",
                "ExtendedID": "FAM2",
                "MPUBID": None,
            },
        ]
    ).to_csv(input_path, index=False)

    script = _repo_root() / "scripts/04_export_links.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=_repo_root(),
    )

    out = pd.read_csv(root / "data/interim/links/links79_links.csv")
    assert len(out) == 1
    assert out.loc[0, "pair_id"] == "1|2"
    assert out.loc[0, "SubjectTag"] == 1
    assert out.loc[0, "PartnerTag"] == 2
    assert out.loc[0, "family_id"] == "FAM1"


def test_export_links_script_supports_all_cohorts_and_output_names(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "links_interim_dir: data/interim/links\n",
    )
    _write(
        root / "config/nlsy79.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen1Housemates\n",
    )
    _write(
        root / "config/nlsy97.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: any\n",
    )
    _write(
        root / "config/cnlsy.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen2Siblings\n",
    )

    nlsy79 = root / "data/interim/links/links79_pair_expanded.csv"
    nlsy97 = root / "data/interim/links/links97_pair_expanded.csv"
    nlsy79.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SubjectTag": 1,
                "PartnerTag": 2,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "ExtendedID": "F1",
                "MPUBID": None,
            },
            {
                "SubjectTag": 3,
                "PartnerTag": 4,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "ExtendedID": "F1",
                "MPUBID": None,
            },
        ]
    ).to_csv(nlsy79, index=False)
    pd.DataFrame(
        [
            {
                "SubjectTag": 11,
                "PartnerTag": 12,
                "R": 0.5,
                "RelationshipPath": "Gen2Siblings",
                "ExtendedID": "X1",
                "MPUBID": None,
            },
        ]
    ).to_csv(nlsy97, index=False)
    pd.DataFrame(
        [
            {
                "SubjectTag": 21,
                "PartnerTag": 22,
                "R": 0.5,
                "RelationshipPath": "Gen2Siblings",
                "ExtendedID": "Y1",
                "MPUBID": "M1",
            },
        ]
    ).to_csv(root / "data/interim/links/links_cnlsy_pair_expanded.csv", index=False)

    script = _repo_root() / "scripts/04_export_links.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all"],
        check=True,
        cwd=_repo_root(),
    )

    expected = {
        root / "data/interim/links/links79_links.csv",
        root / "data/interim/links/links97_links.csv",
        root / "data/interim/links/links_cnlsy_links.csv",
    }
    assert all(path.exists() for path in expected)
    status = pd.read_csv(root / "data/interim/links/link_exports.csv")
    assert set(status["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert set(status["output_path"]) == {
        "data/interim/links/links79_links.csv",
        "data/interim/links/links97_links.csv",
        "data/interim/links/links_cnlsy_links.csv",
    }
    raw_paths = status["raw_path"].dropna().astype(str)
    assert all(raw_paths.str.startswith("data/interim/links/"))
    assert all(not Path(path).is_absolute() for path in raw_paths.tolist())


def test_export_links_script_accepts_package_style_subjecttag_columns(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "links_interim_dir: data/interim/links\n")
    _write(
        root / "config/nlsy97.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: any\n",
    )

    input_path = root / "data/interim/links/links97_pair_expanded.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SubjectTag_S1": 21,
                "SubjectTag_S2": 22,
                "R": 0.5,
                "RelationshipPath": "Gen2Siblings",
                "ExtendedID": "G1",
            }
        ]
    ).to_csv(input_path, index=False)

    script = _repo_root() / "scripts" / "04_export_links.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy97"],
        check=True,
        cwd=_repo_root(),
    )

    out = pd.read_csv(root / "data/interim/links/links97_links.csv")
    assert len(out) == 1
    assert out.loc[0, "SubjectTag"] == 21
    assert out.loc[0, "PartnerTag"] == 22
    assert out.loc[0, "pair_id"] == "21|22"


def test_export_links_script_writes_placeholder_when_source_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "links_interim_dir: data/interim/links\n")
    _write(
        root / "config/nlsy79.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen1Housemates\n",
    )

    script = _repo_root() / "scripts/04_export_links.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=_repo_root(),
    )

    out_file = root / "data/interim/links/links79_links.csv"
    status = pd.read_csv(root / "data/interim/links/link_exports.csv")
    out = pd.read_csv(out_file)
    assert out.empty
    assert set(
        [
            "SubjectTag",
            "PartnerTag",
            "R",
            "RelationshipPath",
            "ExtendedID",
            "MPUBID",
            "pair_id",
            "family_id",
        ]
    ).issubset(out.columns)
    assert status.loc[0, "cohort"] == "nlsy79"
    assert status.loc[0, "status"] == "missing_source"
    assert status.loc[0, "output_path"] == "data/interim/links/links79_links.csv"
