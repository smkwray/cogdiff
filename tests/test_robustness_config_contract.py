from nls_pipeline.io import load_yaml, resolve_from_project


def test_robustness_inference_rerun_command_uses_sem_refit_governed_bootstrap_flags() -> None:
    cfg = load_yaml(resolve_from_project("config/robustness.yml"))
    rerun_command = cfg["rerun_commands"]["inference"]["family_bootstrap"]

    assert "20_run_inference_bootstrap.py" in rerun_command
    assert "--engine sem_refit" in rerun_command
    assert "--min-success-share" in rerun_command
    assert "--sem-timeout-seconds" in rerun_command
    assert "--skip-successful" in rerun_command
