from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_extract_cohort_panel_respects_configured_subset(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    config_dir = project_root / "config"
    config_dir.mkdir()
    (config_dir / "paths.yml").write_text("interim_dir: data/interim\n", encoding="utf-8")
    (config_dir / "nlsy79.yml").write_text(
        """
cohort: nlsy79
panel_extract:
  required_columns:
    - person_id
    - sex
  optional_columns:
    - gpa
    - missing_optional
""",
        encoding="utf-8",
    )

    raw_csv = project_root / "data/interim/nlsy79/raw_files/panel.csv"
    raw_csv.parent.mkdir(parents=True)
    raw_csv.write_text(
        "person_id,sex,gpa,age\n1,F,0.5,16\n2,M,0.7,17\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts/03_extract_cohort_panel.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--project-root",
            str(project_root),
            "--cohort",
            "nlsy79",
            "--source-path",
            str(raw_csv),
            "--force",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    extracted = project_root / "data/interim/nlsy79/panel_extract.csv"
    manifest = project_root / "data/interim/nlsy79/panel_extract.manifest.json"
    assert extracted.exists()
    assert manifest.exists()
    assert extracted.read_text(encoding="utf-8").splitlines()[0] == "person_id,sex,gpa"

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["selected_columns"] == ["person_id", "sex", "gpa"]
    assert manifest_payload["missing_optional_columns"] == ["missing_optional"]
    assert manifest_payload["n_rows"] is None
    assert manifest_payload["source_path"] == "data/interim/nlsy79/raw_files/panel.csv"
    assert manifest_payload["output_path"] == "data/interim/nlsy79/panel_extract.csv"

    # Smoke check that reruns without force skip when artifact is unchanged.
    rerun = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--project-root",
            str(project_root),
            "--cohort",
            "nlsy79",
            "--source-path",
            str(raw_csv),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[skip] " in rerun.stdout


def test_extract_cohort_panel_uses_sample_construct_fallback_subset(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    config_dir = project_root / "config"
    config_dir.mkdir()
    (config_dir / "paths.yml").write_text("interim_dir: data/interim\n", encoding="utf-8")
    (config_dir / "nlsy79.yml").write_text(
        """
cohort: nlsy79
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [GS, AR]
  column_map: {}
""",
        encoding="utf-8",
    )

    raw_csv = project_root / "data/interim/nlsy79/raw_files/panel.csv"
    raw_csv.parent.mkdir(parents=True)
    raw_csv.write_text(
        "person_id,sex,age,birth_year,GS,AR,noise\n1,F,16,1963,0.5,0.7,9\n2,M,17,1962,0.4,0.8,8\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts/03_extract_cohort_panel.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--project-root",
            str(project_root),
            "--cohort",
            "nlsy79",
            "--source-path",
            str(raw_csv),
            "--force",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    extracted = project_root / "data/interim/nlsy79/panel_extract.csv"
    manifest_payload = json.loads((project_root / "data/interim/nlsy79/panel_extract.manifest.json").read_text())
    assert extracted.read_text(encoding="utf-8").splitlines()[0] == "person_id,sex,age,birth_year,GS,AR"
    assert manifest_payload["selected_columns"] == ["person_id", "sex", "age", "birth_year", "GS", "AR"]
    assert manifest_payload["missing_optional_columns"] == []
    assert manifest_payload["source_path"] == "data/interim/nlsy79/raw_files/panel.csv"
    assert manifest_payload["output_path"] == "data/interim/nlsy79/panel_extract.csv"


def test_extract_cohort_panel_fails_without_selector_or_fallback_columns(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    config_dir = project_root / "config"
    config_dir.mkdir()
    (config_dir / "paths.yml").write_text("interim_dir: data/interim\n", encoding="utf-8")
    (config_dir / "nlsy79.yml").write_text(
        """
cohort: nlsy79
sample_construct:
  id_col: person_id
  sex_col: sex
  subtests: [GS]
""",
        encoding="utf-8",
    )

    raw_csv = project_root / "data/interim/nlsy79/raw_files/panel.csv"
    raw_csv.parent.mkdir(parents=True)
    raw_csv.write_text(
        "a,b,c\n1,2,3\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts/03_extract_cohort_panel.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--project-root",
            str(project_root),
            "--cohort",
            "nlsy79",
            "--source-path",
            str(raw_csv),
            "--force",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "No panel_extract selector was configured" in result.stderr
