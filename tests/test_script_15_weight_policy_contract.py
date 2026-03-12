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


def _write_data(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_stage_15_weight_discordance_is_caveat_for_robust_claim(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.20},
                {"cohort": "nlsy79", "inference_method": "family_bootstrap", "estimate_type": "d_g", "status": "computed", "estimate": 0.20},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "weight_mode": "unweighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.20},
                {"cohort": "nlsy79", "weight_mode": "weighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.50},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.20},
                {"cohort": "nlsy79", "sampling_scheme": "full_cohort", "status": "computed", "d_g": 0.20},
                {"cohort": "nlsy79", "sampling_scheme": "one_pair_per_family", "status": "computed", "d_g": 0.20},
            ]
        ),
    )
    _write_data(
        tables_dir / "g_mean_diff.csv",
        pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.20}]),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    row = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert row["robust_claim_sign_eligible"] == True  # noqa: E712
    assert row["robust_claim_primary_eligible"] == True  # noqa: E712
    assert row["robust_claim_warning_eligible"] == True  # noqa: E712
    assert row["robust_claim_weight_eligible"] == False  # noqa: E712
    assert row["weight_concordance_reason"] == "discordant_weight_estimates"
    assert row["robust_claim_eligible"] == True  # noqa: E712
