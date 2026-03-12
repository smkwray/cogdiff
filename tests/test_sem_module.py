from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nls_pipeline import sem


def test_run_sem_r_script_copies_model_from_request_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_dir = tmp_path / "request"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_file = request_dir / "request.json"
    request_file.write_text("{}", encoding="utf-8")
    model_src = request_dir / "model.lavaan"
    model_src.write_text("g =~ x1 + x2 + x3\n", encoding="utf-8")

    outdir = tmp_path / "outputs"
    captured: dict[str, object] = {}

    def fake_run(*, args, check, text, capture_output):
        captured["args"] = args
        assert check is True
        assert text is True
        assert capture_output is True
        model_dst = outdir / "model.lavaan"
        assert model_dst.exists()
        assert model_dst.read_text(encoding="utf-8") == model_src.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sem, "rscript_path", lambda: "/opt/homebrew/bin/Rscript")
    monkeypatch.setattr(sem.subprocess, "run", lambda args, check, text, capture_output: fake_run(args=args, check=check, text=text, capture_output=capture_output))

    result = sem.run_sem_r_script(
        r_script=tmp_path / "scripts" / "sem_fit.R",
        request_file=request_file,
        outdir=outdir,
    )

    assert result.returncode == 0
    assert captured["args"] == [
        "/opt/homebrew/bin/Rscript",
        str(tmp_path / "scripts" / "sem_fit.R"),
        "--request",
        str(request_file),
        "--outdir",
        str(outdir),
    ]


def test_run_sem_r_script_raises_when_model_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_dir = tmp_path / "request"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_file = request_dir / "request.json"
    request_file.write_text("{}", encoding="utf-8")
    outdir = tmp_path / "outputs"

    monkeypatch.setattr(sem, "rscript_path", lambda: "/opt/homebrew/bin/Rscript")

    with pytest.raises(FileNotFoundError):
        sem.run_sem_r_script(
            r_script=tmp_path / "scripts" / "sem_fit.R",
            request_file=request_file,
            outdir=outdir,
        )
