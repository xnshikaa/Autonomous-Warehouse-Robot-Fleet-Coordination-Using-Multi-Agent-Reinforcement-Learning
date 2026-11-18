import json

import pytest

from src.evaluation.scenario_manifest import (
    create_evaluation_manifest,
    load_evaluation_manifest,
    write_evaluation_manifest,
)
from src.evaluation.trace_generator import generate_traces


def test_manifest_round_trip_and_exact_tasks_drive_policy_trace(tmp_path):
    manifest = create_evaluation_manifest(
        fleet_sizes=[5], episodes=20, max_steps=100, seed=976304181
    )
    path = write_evaluation_manifest(manifest, tmp_path / "evaluation.json")
    loaded = load_evaluation_manifest(path)
    scenario = loaded["scenarios"]["5"][0]
    traces = generate_traces(
        "Rule-Based", 5, 1, 100, loaded["evaluation_seed"],
        scenarios=[scenario],
    )
    assert traces[0]["initial_tasks"] == scenario["initial_tasks"]
    assert traces[0]["task_seed"] == scenario["task_seed"]


def test_manifest_rejects_tampered_scenario(tmp_path):
    manifest = create_evaluation_manifest(
        fleet_sizes=[5], episodes=20, max_steps=100, seed=976304181
    )
    path = write_evaluation_manifest(manifest, tmp_path / "evaluation.json")
    document = json.loads(path.read_text())
    document["scenarios"]["5"][0]["initial_tasks"][0]["task_id"] = -1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="ID.*contents"):
        load_evaluation_manifest(path)
