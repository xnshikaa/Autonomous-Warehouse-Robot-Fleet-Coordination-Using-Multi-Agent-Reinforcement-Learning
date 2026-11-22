"""Automated tests for benchmark metric definitions, table formatting, and CI consistency.

Verifies:
1. Physical collisions, safety conflict events, and blocked intent groups are 3 distinct metrics.
2. Point estimates and bootstrap confidence intervals in markdown reports match raw summary data.
3. No cross-contamination between safety conflict CIs and blocked intent group CIs.
4. Holm-Bonferroni adjustment logic and monotonicity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_confidence_interval,
    holm_adjust,
    paired_algorithm_comparison,
    EpisodeRecord,
)
from src.evaluation.run_benchmark import _write_markdown_report


def test_three_distinct_conflict_metrics_in_episode_record() -> None:
    """Ensure physical collisions, safety conflicts, and blocked intent groups remain distinct."""
    record = EpisodeRecord(
        algorithm="QMIX",
        fleet_size=10,
        episode=1,
        collision_events=[{"type": "robot_robot"}],  # 1 physical collision
        safety_conflict_events=[{"step": 1}, {"step": 2}],  # 2 safety conflict events
        collision_intent_groups=[{"group_id": 1}],  # 1 intent group
        safety_overrides=2,
        steps=100,
    )
    from src.evaluation.metrics import _episode_metric

    phys = _episode_metric(record, "collision_rate_per_episode")
    conflicts = _episode_metric(record, "safety_conflict_events")
    intents = _episode_metric(record, "collision_intent_groups")

    assert phys == 1.0, "Physical collision must be 1.0"
    assert conflicts == 2.0, "Safety conflicts must be 2.0"
    assert intents == 1.0, "Collision intent groups must be 1.0"
    assert phys != conflicts, "Physical collisions and safety conflicts must not be conflated"


def test_holm_adjust_properties() -> None:
    """Verify Holm-Bonferroni step-down properties and edge cases."""
    p_vals = [0.01, 0.04, 0.03, 0.005]
    adjusted = holm_adjust(p_vals)

    # 4 tests: sorted p are 0.005 (x4=0.02), 0.01 (x3=0.03), 0.03 (x2=0.06), 0.04 (x1=0.04 -> max(0.06, 0.04)=0.06)
    assert len(adjusted) == 4
    assert adjusted[3] == pytest.approx(0.02)  # 0.005 -> 0.02
    assert adjusted[0] == pytest.approx(0.03)  # 0.01 -> 0.03
    assert adjusted[2] == pytest.approx(0.06)  # 0.03 -> 0.06
    assert adjusted[1] == pytest.approx(0.06)  # 0.04 -> 0.06 (enforcing monotonicity)

    # All values bounded by 1.0
    large_p = [0.8, 0.9]
    adj_large = holm_adjust(large_p)
    assert all(p <= 1.0 for p in adj_large)


def test_markdown_report_ci_alignment(tmp_path: Path) -> None:
    """Verify that every column in the generated report strictly aligns with its metric and CI."""
    dummy_report = {
        "manifest_id": "test_manifest_123",
        "scenario_manifest_id": None,
        "episodes_per_run": 20,
        "fleet_sizes": [10],
        "algorithms": ["QMIX", "IQL"],
        "summary": [
            {
                "algorithm": "QMIX",
                "fleet_size": 10,
                "completed_tasks": 0,
                "tasks": 200,
                "completion_rate": 0.0,
                "average_completion_time_steps": None,
                "collision_rate_per_episode": 0.0,
                "collision_intent_groups_per_episode": 1.05,
                "safety_conflict_rate_per_episode": 1.05,
                "throughput_tasks_per_100_steps": 0.0,
                "collision_events": 0,
            },
            {
                "algorithm": "IQL",
                "fleet_size": 10,
                "completed_tasks": 2,
                "tasks": 200,
                "completion_rate": 0.01,
                "average_completion_time_steps": 31.0,
                "collision_rate_per_episode": 0.0,
                "collision_intent_groups_per_episode": 43.20,
                "safety_conflict_rate_per_episode": 99.00,
                "throughput_tasks_per_100_steps": 0.1,
                "collision_events": 0,
            },
        ],
        "confidence_intervals": [
            # QMIX CIs
            {"algorithm": "QMIX", "fleet_size": 10, "metric": "completion_rate", "estimate": 0.0, "lower": 0.0, "upper": 0.0},
            {"algorithm": "QMIX", "fleet_size": 10, "metric": "collision_rate_per_episode", "estimate": 0.0, "lower": 0.0, "upper": 0.0},
            {"algorithm": "QMIX", "fleet_size": 10, "metric": "collision_intent_groups_per_episode", "estimate": 1.05, "lower": 0.70, "upper": 1.40},
            {"algorithm": "QMIX", "fleet_size": 10, "metric": "safety_conflict_events", "estimate": 1.05, "lower": 0.70, "upper": 1.40},
            # IQL CIs (Intentionally different between intent groups and safety conflict events)
            {"algorithm": "IQL", "fleet_size": 10, "metric": "completion_rate", "estimate": 0.01, "lower": 0.0, "upper": 0.025},
            {"algorithm": "IQL", "fleet_size": 10, "metric": "average_completion_time_steps", "estimate": 31.0, "lower": 30.0, "upper": 32.0},
            {"algorithm": "IQL", "fleet_size": 10, "metric": "collision_rate_per_episode", "estimate": 0.0, "lower": 0.0, "upper": 0.0},
            {"algorithm": "IQL", "fleet_size": 10, "metric": "collision_intent_groups_per_episode", "estimate": 43.20, "lower": 27.70, "upper": 58.95},
            {"algorithm": "IQL", "fleet_size": 10, "metric": "safety_conflict_events", "estimate": 99.00, "lower": 59.00, "upper": 140.40},
        ],
        "paired_comparisons": [],
        "safety_gate": {
            "attempted_corridor_entries": 20,
            "intrusions": 0,
            "episodes": 20,
            "passed": True,
        },
    }

    report_path = tmp_path / "report.md"
    _write_markdown_report(dummy_report, report_path)
    content = report_path.read_text(encoding="utf-8")

    # Verify table headers
    assert "Blocked intent groups/episode (95% CI)" in content
    assert "Safety conflicts/episode (95% CI)" in content
    assert "Collisions/episode (95% CI)" in content

    # Check IQL row: column 7 (blocked intent groups) must have [27.70, 58.95] and point estimate 43.20
    # Column 8 (safety conflicts) must have [59.00, 140.40] and point estimate 99.00
    iql_match = re.search(r"\|\s*IQL\s*\|\s*10\s*\|.*?\|.*?\|.*?\|.*?\|\s*43\.200\s*\(27\.700-58\.950\)\s*\|\s*99\.00\s*\(59\.00-140\.40\)\s*\|", content)
    assert iql_match is not None, f"IQL row did not match expected distinct CIs:\n{content}"

    # Verify that [27.70, 58.95] is NEVER placed beside 99.00
    assert not re.search(r"99\.00.*?27\.70", content), "Critical Bug: Conflated 99.00 with intent groups CI [27.70, 58.95]"


def test_existing_benchmark_n5_n10_consistency() -> None:
    """Validate that results/benchmark_n5_n10 summary.json and report.md are strictly consistent."""
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "results" / "benchmark_n5_n10" / "summary.json"
    report_path = root / "results" / "benchmark_n5_n10" / "report.md"
    if not summary_path.is_file() or not report_path.is_file():
        pytest.skip("Benchmark files not present")

    with open(summary_path, encoding="utf-8") as f:
        summary_data = json.load(f)

    report_content = report_path.read_text(encoding="utf-8")

    # Check CI map from summary
    ci_map = {
        (row["algorithm"], row["fleet_size"], row["metric"]): row
        for row in summary_data["confidence_intervals"]
    }

    # For IQL N=10, assert intent CI and safety CI are distinct
    iql_intent = ci_map[("IQL", 10, "collision_intent_groups_per_episode")]
    iql_conflict = ci_map[("IQL", 10, "safety_conflict_events")]

    assert iql_intent["estimate"] == pytest.approx(43.2)
    assert iql_conflict["estimate"] == pytest.approx(99.0)
    assert iql_intent["lower"] != iql_conflict["lower"]
    assert iql_intent["upper"] != iql_conflict["upper"]

    # In report, verify IQL row has both distinct CIs
    assert "43.200 (27.700-58.950)" in report_content
    assert "99.00 (59.00-140.40)" in report_content
