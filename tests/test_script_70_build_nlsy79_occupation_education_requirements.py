from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "70_build_nlsy79_occupation_education_requirements.py"
    spec = importlib.util.spec_from_file_location("script70_build_nlsy79_occupation_education_requirements", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_nlsy79_occupation_education_requirements_computes_outputs(tmp_path: Path) -> None:
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
                "Writers and authors                      2850  27-3043",
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
                "13-1111.00\t2.D.1\tRequired Level of Education\tRL\t6\t80\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "13-1111.00\t2.D.1\tRequired Level of Education\tRL\t8\t20\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "15-1021.00\t2.D.1\tRequired Level of Education\tRL\t6\t100\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "15-1031.00\t2.D.1\tRequired Level of Education\tRL\t8\t100\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "27-3043.00\t2.D.1\tRequired Level of Education\tRL\t2\t50\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
                "27-3043.00\t2.D.1\tRequired Level of Education\tRL\t6\t50\t10\t0\t0\t0\tN\t08/2023\tIncumbent",
            ]
        ),
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    rows = []
    codes = [80, 101, 102, 285]
    for i in range(800):
        g = (i - 400) / 80.0
        code = codes[i % len(codes)]
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
                "occupation_code_2000": code,
                "age_2000": 36 + (i % 8),
            }
        )
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    mapping, model = module.run_nlsy79_occupation_education_requirements(
        root=root,
        census_text_path=census_text,
        ete_source=str(ete),
        ete_categories_source=str(ete_categories),
        min_n=100,
    )

    assert (root / "outputs" / "tables" / "nlsy79_occupation_education_mapping_quality.csv").exists()
    assert (root / "outputs" / "tables" / "nlsy79_occupation_education_requirement_outcome.csv").exists()
    assert mapping.loc[0, "status"] == "computed"
    assert int(mapping.loc[0, "n_matched_any"]) == 800
    assert int(mapping.loc[0, "n_matched_prefix_only"]) > 0
    assert "Bachelor" in str(mapping.loc[0, "modal_required_education_label"])
    assert set(model["status"]) == {"computed"}
    assert set(model["outcome"]) == {"required_education_years", "bachelor_plus_share"}


def test_run_nlsy79_occupation_education_requirements_handles_missing_columns(tmp_path: Path) -> None:
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
    _write(census_text, "Management analysts                       0800  13-1111\n")
    ete_categories = root / "fixtures/ete_categories.txt"
    _write(
        ete_categories,
        "\n".join(
            [
                "Element ID\tElement Name\tScale ID\tCategory\tCategory Description",
                "2.D.1\tRequired Level of Education\tRL\t6\tBachelor's Degree",
            ]
        ),
    )
    ete = root / "fixtures/ete.txt"
    _write(
        ete,
        "\n".join(
            [
                "O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tCategory\tData Value",
                "13-1111.00\t2.D.1\tRequired Level of Education\tRL\t6\t100",
            ]
        ),
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"GS": 1.0, "AR": 2.0}]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    mapping, model = module.run_nlsy79_occupation_education_requirements(
        root=root,
        census_text_path=census_text,
        ete_source=str(ete),
        ete_categories_source=str(ete_categories),
    )
    assert mapping.loc[0, "status"] == "not_feasible"
    assert set(model["status"]) == {"not_feasible"}
