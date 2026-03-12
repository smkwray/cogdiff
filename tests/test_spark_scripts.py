from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fake_codex_bin(tmp_root: Path, fail_marker: str | None = None) -> Path:
    fake_bin = tmp_root / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_codex = fake_bin / "codex"
    fail_clause = ""
    if fail_marker:
        fail_clause = (
            f'if [[ "${{SPARK_TEST_FAIL_MARKER}}" == "{fail_marker}" ]] && '
            f"[[ \"$input\" == *\"${{SPARK_TEST_FAIL_MARKER}}\"* ]]; then\n"
            '  echo "Simulated codex failure for marker: $SPARK_TEST_FAIL_MARKER" >&2\n'
            "  exit 2\n"
            "fi\n"
        )

    fake_codex.write_text(
        "#!/usr/bin/env bash\n"
        "# Minimal codex shim used by spark runner tests.\n"
        "input=\"$(cat)\"\n"
        f"{fail_clause}"
        "echo '{\"status\":\"ok\"}'\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    return fake_bin


def _run_sparks(
    tmp_root: Path,
    *spark_files: Path,
    fail_marker: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{_fake_codex_bin(tmp_root, fail_marker)}:{env['PATH']}"
    env["SPARK_LOG_DIR"] = str(tmp_root / "logs" / "spark")
    env["SPARK_AUTO_CLEANUP"] = "0"
    if fail_marker:
        env["SPARK_TEST_FAIL_MARKER"] = fail_marker
    else:
        env.pop("SPARK_TEST_FAIL_MARKER", None)

    return subprocess.run(
        [str(_repo_root() / "run_sparks.sh"), *[str(path) for path in spark_files]],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )


def test_run_sparks_writes_manifest_and_warns_on_incomplete_prompt(tmp_path: Path) -> None:
    prompt_incomplete = tmp_path / "spark_incomplete.md"
    prompt_complete = tmp_path / "spark_complete.md"
    _write(prompt_incomplete, "# Spark Task 99 - in progress\n\nStill running")
    _write(prompt_complete, "# Spark Task 100 - Completed\n\nDone")

    result = _run_sparks(tmp_path, prompt_incomplete, prompt_complete)
    assert result.returncode == 0

    combined_output = result.stdout + result.stderr
    assert "Warning: successful run but prompt not rewritten to completed heading" in combined_output

    manifest_match = re.search(r"Wave manifest written: (.+)", combined_output)
    assert manifest_match is not None
    manifest_path = Path(manifest_match.group(1).strip())
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "Wave manifest written" in combined_output
    assert str(prompt_incomplete) in manifest
    assert str(prompt_complete) in manifest
    assert "| success | no_heading |" not in manifest  # explicit sanity check on state handling
    assert "| success | active_or_incomplete |" in manifest
    assert "| success | completed_heading |" in manifest
    assert manifest_path.exists()
    assert manifest.startswith("# Spark Wave Manifest")


def test_run_sparks_propagates_nonzero_codex_exit(tmp_path: Path) -> None:
    fail_prompt = tmp_path / "spark_fail.md"
    ok_prompt = tmp_path / "spark_ok.md"
    marker = "FAIL_MARKER_123"
    _write(fail_prompt, f"# Spark Task 99 - in progress\n\n{marker}\n")
    _write(ok_prompt, "# Spark Task 100 - Completed\n\nDone\n")

    result = _run_sparks(tmp_path, fail_prompt, ok_prompt, fail_marker=marker)
    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "Failed Spark prompt:" in combined_output
    assert "Spark run(s) failed." in combined_output

    manifest_match = re.search(r"Wave manifest written: (.+)", combined_output)
    assert manifest_match is not None
    manifest_path = Path(manifest_match.group(1).strip())
    manifest = manifest_path.read_text(encoding="utf-8")

    fail_row_re = re.compile(
        rf"\|\s*{re.escape(str(fail_prompt))}\s*\|\s*spark_fail\s*\|\s*failed\s*\|\s*.*?\s*\|\s*([^|]+)\|\s*([^|]+)\|"
    )
    fail_row_match = fail_row_re.search(manifest)
    assert fail_row_match is not None
    stderr_log = Path(fail_row_match.group(2).strip())
    assert "Simulated codex failure for marker: FAIL_MARKER_123" in stderr_log.read_text(encoding="utf-8")


def test_run_sparks_manifest_records_failure_summary(tmp_path: Path) -> None:
    fail_prompt = tmp_path / "spark_fail.md"
    ok_prompt = tmp_path / "spark_ok.md"
    marker = "FAIL_MARKER_124"
    _write(fail_prompt, f"# Spark Task 99 - in progress\n\n{marker}\n")
    _write(ok_prompt, "# Spark Task 100 - in progress\n\nStill running\n")

    result = _run_sparks(tmp_path, fail_prompt, ok_prompt, fail_marker=marker)
    assert result.returncode != 0

    combined_output = result.stdout + result.stderr
    manifest_match = re.search(r"Wave manifest written: (.+)", combined_output)
    assert manifest_match is not None
    manifest_path = Path(manifest_match.group(1).strip())
    manifest = manifest_path.read_text(encoding="utf-8")

    assert "- Failures: 1" in manifest
    assert str(fail_prompt) in manifest
    assert str(ok_prompt) in manifest
    fail_row = f"| {fail_prompt} | spark_fail | failed | active_or_incomplete |"
    ok_row = f"| {ok_prompt} | spark_ok | success | active_or_incomplete |"
    assert fail_row in manifest
    assert ok_row in manifest


def test_spark_cleanup_deletes_only_completed_extras_and_keeps_active_files(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    _write(prompts / "spark1.md", "# Spark Task 1 - Completed\n")
    _write(prompts / "spark2.md", "# Spark Task 2 - Completed\n")
    _write(prompts / "spark3.md", "# Spark Task 3 - in progress\n")

    result = subprocess.run(
        [
            "bash",
            str(_repo_root() / "prompts" / "spark_cleanup.sh"),
            "--project-root",
            str(tmp_path),
            "--keep",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Cleanup summary: deleted=1" in output
    assert (prompts / "spark1.md").exists()
    assert (prompts / "spark2.md").exists() is False
    assert (prompts / "spark3.md").exists()


def test_spark_cleanup_dry_run_preserves_files(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    _write(prompts / "spark1.md", "# Spark Task 1 - Completed\n")
    _write(prompts / "spark2.md", "# Spark Task 2 - Completed\n")

    result = subprocess.run(
        [
            "bash",
            str(_repo_root() / "prompts" / "spark_cleanup.sh"),
            "--project-root",
            str(tmp_path),
            "--keep",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "[dry-run] delete completed extra: " in output
    assert (prompts / "spark1.md").exists()
    assert (prompts / "spark2.md").exists()
