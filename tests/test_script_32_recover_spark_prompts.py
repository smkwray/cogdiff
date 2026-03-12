from __future__ import annotations

import importlib.util
import sys
from pathlib import Path



def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "32_recover_spark_prompts.py"
    spec = importlib.util.spec_from_file_location("script32_recover_spark_prompts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prompt(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_manifest(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spark Wave Manifest",
        "## Prompt Runs",
        "| Prompt | Spark | Status | Completion state | JSON log | STDERR log |",
        "|---|---|---|---|---|---|",
    ]
    for prompt_path, spark_name, status, completion_state, json_log, stderr_log in rows:
        lines.append(
            f"| {prompt_path} | {spark_name} | {status} | {completion_state} | {json_log} | {stderr_log} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_dry_run_does_not_modify_prompt_file(tmp_path: Path, monkeypatch: object) -> None:
    module = _module()
    root = tmp_path / "project"
    prompt = root / "prompts" / "spark7.md"
    _write_prompt(prompt, "# Spark Task 7\n\nIn progress\n")

    manifest = root / "logs" / "20260221_140000_wave_manifest.md"
    _write_manifest(
        manifest,
        [
            (
                str(prompt.relative_to(root)),
                "spark7",
                "success",
                "active_or_incomplete",
                "/tmp/spark_run.jsonl",
                "/tmp/spark_run.stderr.log",
            )
        ],
    )

    monkeypatch.setattr(module.sys, "argv", ["32_recover_spark_prompts.py", str(manifest), "--project-root", str(root), "--dry-run"])
    code = module.main()
    assert code == 0

    assert prompt.read_text(encoding="utf-8") == "# Spark Task 7\n\nIn progress\n"


def test_apply_rewrites_only_incomplete_success_rows(tmp_path: Path, monkeypatch: object) -> None:
    module = _module()
    root = tmp_path / "project"
    prompt_to_rewrite = root / "prompts" / "spark8.md"
    prompt_completed = root / "prompts" / "spark9.md"
    prompt_failed = root / "prompts" / "spark10.md"

    completed_content = "# Spark Task 9 - Completed\n\nPreviously done\n"
    _write_prompt(prompt_to_rewrite, "# Spark Task 8\n\nNeed write\n")
    _write_prompt(prompt_completed, completed_content)
    _write_prompt(prompt_failed, "# Spark Task 10\n\nNeed write but failed\n")

    manifest = root / "logs" / "20260221_150000_wave_manifest.md"
    _write_manifest(
        manifest,
        [
            (
                str(prompt_to_rewrite.relative_to(root)),
                "spark8",
                "success",
                "no_heading",
                "/tmp/spark8.jsonl",
                "/tmp/spark8.stderr.log",
            ),
            (
                str(prompt_completed.relative_to(root)),
                "spark9",
                "success",
                "completed_heading",
                "/tmp/spark9.jsonl",
                "/tmp/spark9.stderr.log",
            ),
            (
                str(prompt_failed.relative_to(root)),
                "spark10",
                "failed",
                "active_or_incomplete",
                "/tmp/spark10.jsonl",
                "/tmp/spark10.stderr.log",
            ),
        ],
    )

    monkeypatch.setattr(module.sys, "argv", ["32_recover_spark_prompts.py", str(manifest), "--project-root", str(root)])
    code = module.main()
    assert code == 0

    recovered = prompt_to_rewrite.read_text(encoding="utf-8")
    assert recovered.startswith("# Spark Task 8 - Completed")
    assert "Recovery note: Spark runner reported success, but did not rewrite completed-heading status." in recovered
    assert "- JSON log: `/tmp/spark8.jsonl`" in recovered
    assert "- STDERR log: `/tmp/spark8.stderr.log`" in recovered

    assert prompt_completed.read_text(encoding="utf-8") == completed_content
    assert prompt_failed.read_text(encoding="utf-8") == "# Spark Task 10\n\nNeed write but failed\n"
