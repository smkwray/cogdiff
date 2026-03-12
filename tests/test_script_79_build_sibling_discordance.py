from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_minimal_project(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)
    (root / "config/paths.yml").write_text("processed_dir: data/processed\n", encoding="utf-8")
    model_text = "\n".join(
        [
            "hierarchical_factors:",
            "  speed: [GS, CS, MC]",
            "  math: [AR, MK]",
            "  verbal: [WK, PC, 'NO']",
            "  technical: [AS, EI]",
            "pair_rules:",
            "  relatedness_r: 0.5",
            "  relationship_path: any",
        ]
    ) + "\n"
    (root / "config/nlsy79.yml").write_text(model_text, encoding="utf-8")
    (root / "config/nlsy97.yml").write_text(model_text, encoding="utf-8")
    (root / "config/models.yml").write_text(
        "\n".join(
            [
                "hierarchical_factors:",
                "  speed: [GS, CS, MC]",
                "  math: [AR, MK]",
                "  verbal: [WK, PC, 'NO']",
                "  technical: [AS, EI]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_person_row(person_id: int, g: float, cohort: str) -> dict[str, float]:
    row = {
        "person_id": person_id,
        "GS": g + 1.0,
        "AR": g + 2.0,
        "WK": g + 3.0,
        "PC": g + 4.0,
        "NO": g + 5.0,
        "CS": g + 6.0,
        "AS": g + 7.0,
        "MK": g + 8.0,
        "MC": g + 9.0,
        "EI": g + 10.0,
        "education_years": 12.0 + (1.3 * g),
        "net_worth": 10000.0 + (5000.0 * g),
    }
    if cohort == "nlsy79":
        row.update(
            {
                "household_income": 40000.0 + (9000.0 * g),
                "annual_earnings": 25000.0 + (7000.0 * g),
                "employment_2000": 1.0 if g > -0.2 else 0.0,
            }
        )
    else:
        row.update(
            {
                "household_income_2021": 50000.0 + (8000.0 * g),
                "annual_earnings_2021": 32000.0 + (6000.0 * g),
                "employment_2021": 1.0 if g > 0.0 else 0.0,
            }
        )
    return row


def test_script_79_builds_discordance_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    nlsy79_rows = []
    nlsy79_links = []
    for fam in range(40):
        left_id = 1000 + (fam * 2)
        right_id = left_id + 1
        g1 = -1.0 + (fam * 0.06)
        g2 = g1 + 0.35 + ((fam % 5) * 0.03)
        nlsy79_rows.append(_make_person_row(left_id, g1, "nlsy79"))
        nlsy79_rows.append(_make_person_row(right_id, g2, "nlsy79"))
        nlsy79_links.append(
            {
                "SubjectTag": left_id,
                "PartnerTag": right_id,
                "R": 0.5,
                "RelationshipPath": "Siblings",
                "pair_id": f"{left_id}|{right_id}",
                "family_id": fam + 1,
            }
        )
    pd.DataFrame(nlsy79_rows).to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(nlsy79_links).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    nlsy97_rows = []
    nlsy97_links = []
    for fam in range(35):
        left_id = 2000 + (fam * 2)
        right_id = left_id + 1
        g1 = -0.9 + (fam * 0.07)
        g2 = g1 + 0.30 + ((fam % 4) * 0.04)
        nlsy97_rows.append(_make_person_row(left_id, g1, "nlsy97"))
        nlsy97_rows.append(_make_person_row(right_id, g2, "nlsy97"))
        nlsy97_links.append(
            {
                "SubjectTag": left_id,
                "PartnerTag": right_id,
                "R": 0.5,
                "RelationshipPath": "Siblings",
                "pair_id": f"{left_id}|{right_id}",
                "family_id": 100 + fam,
            }
        )
    pd.DataFrame(nlsy97_rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame(nlsy97_links).to_csv(root / "data/interim/links/links97_links.csv", index=False)

    script = _repo_root() / "scripts" / "79_build_sibling_discordance.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all", "--min-pairs", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/sibling_discordance.csv")
    computed = out.loc[out["status"] == "computed"].copy()
    assert {"nlsy79", "nlsy97"} <= set(computed["cohort"])
    assert {"education", "household_income", "net_worth", "earnings", "employment"} <= set(computed["outcome"])
    assert (computed["n_pairs"] >= 10).all()
    assert computed["beta_abs_g_diff"].notna().all()


def test_script_79_handles_missing_links(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    pd.DataFrame([_make_person_row(1, 0.1, "nlsy79")]).to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame([_make_person_row(2, 0.2, "nlsy97")]).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "79_build_sibling_discordance.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all", "--min-pairs", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/sibling_discordance.csv")
    assert (out["status"] == "not_feasible").all()
