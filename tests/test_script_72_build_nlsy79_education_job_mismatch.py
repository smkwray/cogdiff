from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "72_build_nlsy79_education_job_mismatch.py"
    spec = importlib.util.spec_from_file_location("script72_build_nlsy79_education_job_mismatch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_nlsy79_education_job_mismatch_computes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    census_text = root / "fixtures/census_codes.txt"
    _write(
        census_text,
        "\n".join(
            [
                "Management analysts                       0800  13-1111",
                "Computer programmers                     1010  15-1021",
                "Computer software engineers              1020  15-1030",
            ]
        ),
    )
    ete_categories = root / "fixtures/ete_categories.txt"
    _write(
        ete_categories,
        "\n".join(
            [
                "Element ID\tElement Name\tScale ID\tCategory\tCategory Description",
                "2.D.1\tRequired Level of Education\tRL\t2\tHigh School Diploma",
                "2.D.1\tRequired Level of Education\tRL\t6\tBachelor's Degree",
                "2.D.1\tRequired Level of Education\tRL\t8\tMaster's Degree",
            ]
        ),
    )
    ete = root / "fixtures/ete.txt"
    _write(
        ete,
        "\n".join(
            [
                "O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tCategory\tData Value\tN\tStandard Error\tLower CI Bound\tUpper CI Bound\tRecommend Suppress\tDate\tDomain Source",
                "13-1111.00\t2.D.1\tRequired Level of Education\tRL\t6\t100\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "15-1021.00\t2.D.1\tRequired Level of Education\tRL\t2\t100\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "15-1031.00\t2.D.1\tRequired Level of Education\tRL\t8\t100\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
            ]
        ),
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    rows = []
    codes = [80, 101, 102]
    educs = [18.0, 16.0, 12.0]
    for i in range(900):
        g = (i - 450) / 90.0
        idx = i % len(codes)
        rows.append(
            {
                "GS": g + 0.1,
                "AR": g - 0.1,
                "WK": g + 0.2,
                "PC": g - 0.2,
                "NO": g + 0.15,
                "CS": g - 0.15,
                "AS": g + 0.05,
                "MK": g - 0.05,
                "MC": g + 0.03,
                "EI": g - 0.03,
                "occupation_code_2000": codes[idx],
                "education_years": educs[idx] + float(i % 2),
                "age_2000": 36 + (i % 8),
                "annual_earnings": 25000 + (i * 10),
                "household_income": 45000 + (i * 15),
            }
        )
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy79_education_job_mismatch(
        root=root,
        census_text_path=census_text,
        ete_source=str(ete),
        ete_categories_source=str(ete_categories),
        min_n=100,
        min_class_n=50,
    )

    assert (root / "outputs" / "tables" / "nlsy79_education_job_mismatch_summary.csv").exists()
    assert (root / "outputs" / "tables" / "nlsy79_education_job_mismatch_models.csv").exists()
    assert set(summary["status"]) == {"computed"}
    assert set(summary["group"]) == {"overall", "undereducated", "matched_band", "overeducated"}
    assert int(summary.loc[summary["group"] == "overall", "n_used"].iloc[0]) == 900
    assert set(model["outcome"]) == {"mismatch_years", "abs_mismatch_years", "overeducated", "undereducated"}
    assert "computed" in set(model["status"])


def test_run_nlsy79_education_job_mismatch_handles_missing_columns(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"GS": 1.0, "AR": 2.0}]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy79_education_job_mismatch(root=root, min_n=100)
    assert set(summary["status"]) == {"not_feasible"}
    assert set(model["status"]) == {"not_feasible"}
