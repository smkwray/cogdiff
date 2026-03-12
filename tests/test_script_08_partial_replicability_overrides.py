from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "08_invariance_and_partial.py"
    spec = importlib.util.spec_from_file_location("script08_invariance_and_partial", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_partial_replicability_overrides_sets_and_clamps_values() -> None:
    module = _module()
    base = {
        "enabled": True,
        "apply_when_partial_refit": True,
        "mode": "combined",
        "min_overlap_share": 0.7,
        "min_bootstrap_indicator_share": 0.7,
        "min_bootstrap_success_reps": 100,
        "bootstrap_replicates": 200,
    }

    out = module._apply_partial_replicability_overrides(
        base,
        enabled_override=False,
        apply_when_partial_refit_override=False,
        mode_override="cross_cohort_overlap",
        min_overlap_share_override=2.0,
        min_bootstrap_indicator_share_override=-0.5,
        min_bootstrap_success_reps_override=999,
        bootstrap_replicates_override=50,
    )

    assert out["enabled"] is False
    assert out["apply_when_partial_refit"] is False
    assert out["mode"] == "cross_cohort_overlap"
    assert float(out["min_overlap_share"]) == 1.0
    assert float(out["min_bootstrap_indicator_share"]) == 0.0
    assert int(out["bootstrap_replicates"]) == 50
    assert int(out["min_bootstrap_success_reps"]) == 50


def test_apply_partial_replicability_overrides_preserves_defaults_when_none() -> None:
    module = _module()
    base = {
        "enabled": True,
        "apply_when_partial_refit": True,
        "mode": "combined",
        "min_overlap_share": 0.7,
        "min_bootstrap_indicator_share": 0.7,
        "min_bootstrap_success_reps": 100,
        "bootstrap_replicates": 200,
    }

    out = module._apply_partial_replicability_overrides(base)
    assert out == base
