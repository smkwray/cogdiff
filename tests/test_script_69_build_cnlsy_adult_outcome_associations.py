from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_minimal_project(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "config/paths.yml").write_text("processed_dir: data/processed\n", encoding="utf-8")
    (root / "config/models.yml").write_text("cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n", encoding="utf-8")


def test_script_69_builds_cnlsy_adult_outcome_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    for i in range(140):
        base = rng.normal()
        enrolled = float(rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.2 + 0.4 * base)))))
        rows.append(
            {
                "PPVT": 100 + (base * 10) + rng.normal(),
                "PIAT_RR": 95 + (base * 9) + rng.normal(),
                "PIAT_RC": 96 + (base * 8) + rng.normal(),
                "age_2014": 24 + (i % 4),
                "education_years_2014": 13 + (base * 0.9) + rng.normal(scale=0.5),
                "wage_income_2014": 28000 + (base * 5000) + rng.normal(scale=1000),
                "family_income_2014": 42000 + (base * 6000) + rng.normal(scale=2000),
                "total_hours_2014": 28 + (base * 4) + rng.normal(scale=1.5),
                "num_current_jobs_2014": np.clip(np.round(1 + (base * 0.3) + rng.normal(scale=0.2)), 0, 3),
                "enrolled_2014": enrolled,
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/cnlsy_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "69_build_cnlsy_adult_outcome_associations.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-n-continuous", "40", "--min-class-n-binary", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/cnlsy_adult_outcome_associations.csv")
    computed = out.loc[out["status"] == "computed"].copy()
    assert "education_years_2014" in set(computed["outcome"])
    assert "wage_income_2014" in set(computed["outcome"])
    assert "family_income_2014" in set(computed["outcome"])
    assert "enrolled_2014" in set(computed["outcome"])


def test_script_69_marks_missing_age_field(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    pd.DataFrame([{"PPVT": 1, "PIAT_RR": 1, "PIAT_RC": 1, "education_years_2014": 12}]).to_csv(
        root / "data/processed/cnlsy_cfa_resid.csv", index=False
    )

    script = _repo_root() / "scripts" / "69_build_cnlsy_adult_outcome_associations.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/cnlsy_adult_outcome_associations.csv")
    assert (out["status"] == "not_feasible").all()
    assert set(out["reason"]) == {"missing_age_2014"}
