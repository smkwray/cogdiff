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
    (root / "config/paths.yml").write_text("processed_dir: data/processed\n", encoding="utf-8")
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


def test_script_77_builds_volatility_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    rows: list[dict[str, float]] = []
    for i in range(80):
        g = (i - 40.0) / 12.0
        baseline_income = 40000.0 + (g * 1800.0)
        baseline_earn = 25000.0 + (g * 1300.0)
        noise = ((i % 7) - 3.0) * 0.008
        income_change = 0.10 - (0.012 * g) + noise
        earn_change = 0.14 - (0.015 * g) - (noise * 0.9)
        follow_income = baseline_income * (1.0 + income_change)
        follow_earn = baseline_earn * (1.0 + earn_change)
        rows.append(
            {
                "person_id": i + 1,
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
                "household_income_2019": baseline_income,
                "household_income_2021": follow_income,
                "annual_earnings_2019": baseline_earn,
                "annual_earnings_2021": follow_earn,
                "age_2019": 35.0 + (i % 3),
                "age_2021": 37.0 + (i % 3),
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "77_build_nlsy97_income_earnings_volatility.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-n", "20", "--min-class-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_income_earnings_volatility.csv")
    assert set(out["outcome"]) == {"annual_earnings", "household_income"}
    assert set(out["model"]) == {"abs_annualized_log_change", "high_instability_top_quartile"}
    computed = out.loc[out["status"] == "computed"].copy()
    assert len(computed) == 4
    assert (computed["n_used"] == 80).all()
    ols_rows = computed.loc[computed["model"] == "abs_annualized_log_change"].copy()
    assert (ols_rows["beta_g"] < 0).all()
    logit_rows = computed.loc[computed["model"] == "high_instability_top_quartile"].copy()
    assert (logit_rows["odds_ratio_g"] < 1.0).all()


def test_script_77_marks_insufficient_rows(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    rows: list[dict[str, float]] = []
    for i in range(6):
        g = float(i + 1)
        rows.append(
            {
                "person_id": i + 1,
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
                "household_income_2019": 100.0 + (g * 5.0),
                "household_income_2021": 110.0 + (g * 5.0),
                "annual_earnings_2019": 50.0 + (g * 4.0),
                "annual_earnings_2021": 55.0 + (g * 4.0),
                "age_2019": 35.0,
                "age_2021": 37.0,
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "77_build_nlsy97_income_earnings_volatility.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-n", "20", "--min-class-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_income_earnings_volatility.csv")
    assert (out["status"] == "not_feasible").all()
    assert set(out["reason"]) == {"insufficient_two_wave_rows"}
