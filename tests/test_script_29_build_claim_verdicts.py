from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _run_script(root: Path, *argv: str) -> None:
    script = _repo_root() / "scripts" / "29_build_claim_verdicts.py"
    subprocess.run([sys.executable, str(script), "--project-root", str(root), *map(str, argv)], check=True)


def test_script_29_writes_deterministic_verdicts_for_all_present_inputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs/tables/specification_stability_summary.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "estimand": "d_g",
                    "robust_claim_eligible": True,
                    "weight_diff_threshold": 0.1,
                },
                {
                    "cohort": "nlsy79",
                    "estimand": "vr_g",
                    "robust_claim_eligible": False,
                    "weight_diff_threshold": 0.1,
                },
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/analysis_tiers.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "estimand": "d_g", "blocked_confirmatory": False},
                {"cohort": "nlsy79", "estimand": "vr_g", "blocked_confirmatory": False},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/discrepancy_attribution_matrix.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "claim_id": "nlsy79:d_g:unweighted_vs_weighted",
                    "metric": "d_g",
                    "delta": 0.05,
                },
                {
                    "cohort": "nlsy79",
                    "claim_id": "nlsy79:log_vr_g:unweighted_vs_weighted",
                    "metric": "log_vr_g",
                    "delta": 0.20,
                },
            ]
        ),
    )

    _run_script(root)

    output = root / "outputs/tables/claim_verdicts.csv"
    assert output.exists()
    verdicts = pd.read_csv(output)
    assert list(verdicts.columns) == [
        "claim_id",
        "claim_label",
        "verdict",
        "reason",
        "cohorts_evaluated",
        "thresholds_used",
    ]

    mapped = verdicts.set_index("claim_id")["verdict"].to_dict()
    assert mapped["nlsy79:d_g:unweighted_vs_weighted"] == "confirmed"
    assert mapped["nlsy79:log_vr_g:unweighted_vs_weighted"] == "not_confirmed"
    assert set(verdicts["cohorts_evaluated"]) == {"nlsy79"}


def test_script_29_handles_missing_artifact_gracefully_with_inconclusive(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs/tables/specification_stability_summary.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "estimand": "d_g",
                    "robust_claim_eligible": True,
                    "weight_diff_threshold": 0.1,
                }
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/analysis_tiers.csv",
        pd.DataFrame([{"cohort": "nlsy79", "estimand": "d_g", "blocked_confirmatory": False}]),
    )

    # Deliberately omit discrepancy_attribution_matrix.csv
    _run_script(root)

    output = root / "outputs/tables/claim_verdicts.csv"
    verdicts = pd.read_csv(output)
    assert len(verdicts) == 1
    assert verdicts.loc[0, "verdict"] == "inconclusive"
    assert "discrepancy_attribution_matrix" in str(verdicts.loc[0, "reason"]) 


def test_script_29_verdict_mapping_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs/tables/specification_stability_summary.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "estimand": "d_g",
                    "robust_claim_eligible": True,
                    "weight_diff_threshold": 0.1,
                },
                {
                    "cohort": "nlsy97",
                    "estimand": "d_g",
                    "robust_claim_eligible": False,
                    "weight_diff_threshold": 0.1,
                },
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/analysis_tiers.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "estimand": "d_g", "blocked_confirmatory": False},
                {"cohort": "nlsy97", "estimand": "d_g", "blocked_confirmatory": False},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/discrepancy_attribution_matrix.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "claim_id": "nlsy79:d_g:unweighted_vs_weighted", "metric": "d_g", "delta": 0.02},
                {"cohort": "nlsy97", "claim_id": "nlsy97:d_g:unweighted_vs_weighted", "metric": "d_g", "delta": 0.20},
            ]
        ),
    )

    output_path = root / "outputs/tables/claim_verdicts.csv"
    _run_script(root)
    first = pd.read_csv(output_path)
    _run_script(root)
    second = pd.read_csv(output_path)

    assert first.equals(second)
    verdicts = first.set_index("claim_id")["verdict"].to_dict()
    assert verdicts["nlsy79:d_g:unweighted_vs_weighted"] == "confirmed"
    assert verdicts["nlsy97:d_g:unweighted_vs_weighted"] == "not_confirmed"
