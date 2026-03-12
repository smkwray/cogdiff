from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module():
    path = _repo_root() / "scripts" / "00_download_raw.py"
    spec = importlib.util.spec_from_file_location("stage00_download_raw", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_stage00_writes_relative_manifest_paths(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "raw_dir: data/raw\nmanifest_file: data/raw/manifest.json\n",
    )
    _write(
        root / "config/nlsy79.yml",
        "download_url: https://example.test/nlsy79.zip\nraw_zip_name: nlsy79.zip\n",
    )

    module = _module()
    monkeypatch.setenv("NLS_PROJECT_ROOT", str(root))

    def _fake_download(url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"downloaded from {url}".encode("utf-8"))

    monkeypatch.setattr(module, "_download", _fake_download)
    monkeypatch.setattr(module.sys, "argv", ["00_download_raw.py", "--cohort", "nlsy79", "--force"])

    result = module.main()
    assert result == 0

    payload = json.loads((root / "data/raw/manifest.json").read_text(encoding="utf-8"))
    assert payload["nlsy79"]["path"] == "data/raw/nlsy79.zip"
    assert not Path(payload["nlsy79"]["path"]).is_absolute()
