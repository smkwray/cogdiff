from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "66_build_nlsy79_job_zone_complexity.py"
    spec = importlib.util.spec_from_file_location("script66_build_nlsy79_job_zone_complexity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_census_crosswalk_from_text_fixture() -> None:
    module = _load_module()
    fixture = Path(__file__).resolve().parent / "_tmp_census_jobzone_fixture.txt"
    fixture.write_text(
        "\n".join(
            [
                "Management analysts                       0800  13-1111",
                "Computer programmers                     1010  15-1021",
                "Writers and authors                      2850  27-3043",
                "Something unmapped                       9999  none",
            ]
        ),
        encoding="utf-8",
    )
    try:
        crosswalk = module._load_census_crosswalk(census_pdf_url="unused", census_text_path=fixture)
    finally:
        fixture.unlink(missing_ok=True)

    assert list(crosswalk["census_code"]) == [800, 1010, 2850]
    assert list(crosswalk["soc_code"]) == ["13-1111", "15-1021", "27-3043"]


def test_run_nlsy79_job_zone_complexity_computes_outputs(tmp_path: Path) -> None:
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
    job_zones = root / "fixtures/job_zones.txt"
    _write(
        job_zones,
        "O*NET-SOC Code\tTitle\tJob Zone\n"
        "13-1111.00\tManagement Analysts\t4\n"
        "15-1029.00\tComputer Specialists, All Other\t5\n"
        "15-1031.00\tSoftware Developers, Applications\t5\n"
        "27-3043.00\tWriters and Authors\t4\n",
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

    mapping, model = module.run_nlsy79_job_zone_complexity(
        root=root,
        census_text_path=census_text,
        job_zones_source=str(job_zones),
        min_n=100,
    )

    assert (root / "outputs" / "tables" / "nlsy79_job_zone_mapping_quality.csv").exists()
    assert (root / "outputs" / "tables" / "nlsy79_job_zone_complexity_outcome.csv").exists()
    assert mapping.loc[0, "status"] == "computed"
    assert int(mapping.loc[0, "n_matched_any"]) == 800
    assert int(mapping.loc[0, "n_matched_prefix_only"]) > 0
    assert model.loc[0, "status"] == "computed"
    assert float(model.loc[0, "mean_job_zone"]) >= 4.0


def test_run_nlsy79_job_zone_complexity_handles_missing_columns(tmp_path: Path) -> None:
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
    job_zones = root / "fixtures/job_zones.txt"
    _write(job_zones, "O*NET-SOC Code\tTitle\tJob Zone\n13-1111.00\tManagement Analysts\t4\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"GS": 1.0, "AR": 2.0}]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    mapping, model = module.run_nlsy79_job_zone_complexity(
        root=root,
        census_text_path=census_text,
        job_zones_source=str(job_zones),
    )
    assert mapping.loc[0, "status"] == "not_feasible"
    assert model.loc[0, "status"] == "not_feasible"
