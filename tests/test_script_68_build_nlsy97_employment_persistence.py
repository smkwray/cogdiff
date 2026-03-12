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


def test_script_68_builds_persistence_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    for i in range(240):
        base = rng.normal()
        p2011 = 1.0 / (1.0 + np.exp(-(-0.2 + 0.7 * base)))
        e2011 = float(rng.binomial(1, p2011))
        p2019 = 1.0 / (1.0 + np.exp(-(-0.1 + 0.9 * base + 1.0 * e2011)))
        e2019 = float(rng.binomial(1, p2019))
        p2021 = 1.0 / (1.0 + np.exp(-(-0.2 + 1.0 * base + 1.2 * e2019 + 0.4 * e2011)))
        e2021 = float(rng.binomial(1, p2021))
        rows.append(
            {
                "person_id": i + 1,
                "GS": base + rng.normal(scale=0.2),
                "AR": base + rng.normal(scale=0.2),
                "WK": base + rng.normal(scale=0.2),
                "PC": base + rng.normal(scale=0.2),
                "NO": base + rng.normal(scale=0.2),
                "CS": base + rng.normal(scale=0.2),
                "AS": base + rng.normal(scale=0.2),
                "MK": base + rng.normal(scale=0.2),
                "MC": base + rng.normal(scale=0.2),
                "EI": base + rng.normal(scale=0.2),
                "employment_2011": e2011,
                "employment_2019": e2019,
                "employment_2021": e2021,
                "age_2019": 35.0 + float(i % 5),
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "68_build_nlsy97_employment_persistence.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-class-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_employment_persistence.csv")
    computed = out.loc[out["status"] == "computed"].copy()
    assert len(computed) >= 2
    assert "persistent_employment_2019_2021" in set(computed["model"])
    persistent = computed.loc[computed["model"] == "persistent_employment_2019_2021"].iloc[0]
    assert float(persistent["odds_ratio_g"]) > 1.0


def test_script_68_marks_insufficient_class_counts(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    rows = []
    for i in range(20):
        rows.append(
            {
                "person_id": i + 1,
                "GS": float(i + 1),
                "AR": float(i + 2),
                "WK": float(i + 3),
                "PC": float(i + 4),
                "NO": float(i + 5),
                "CS": float(i + 6),
                "AS": float(i + 7),
                "MK": float(i + 8),
                "MC": float(i + 9),
                "EI": float(i + 10),
                "employment_2011": 1.0,
                "employment_2019": 1.0,
                "employment_2021": 1.0,
                "age_2019": 35.0,
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "68_build_nlsy97_employment_persistence.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-class-n", "5"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_employment_persistence.csv")
    assert (out["status"] == "not_feasible").all()
    assert set(out["reason"]).issubset({"insufficient_class_counts", "empty_subset"})
