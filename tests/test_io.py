from pathlib import Path

from nls_pipeline.io import load_yaml, project_root, resolve_from_project


def test_project_root_contains_plan() -> None:
    root = project_root()
    assert (root / "replication_plan.md").exists()


def test_load_yaml_paths() -> None:
    paths = load_yaml(resolve_from_project("config/paths.yml"))
    assert "raw_dir" in paths
    assert "manifest_file" in paths


def test_resolve_from_project_relative() -> None:
    resolved = resolve_from_project("config/nlsy79.yml")
    assert isinstance(resolved, Path)
    assert resolved.exists()


def test_project_root_env_override(monkeypatch) -> None:
    override = Path("/tmp/custom-nls-root")
    monkeypatch.setenv("NLS_PROJECT_ROOT", str(override))
    assert project_root() == override.resolve()
