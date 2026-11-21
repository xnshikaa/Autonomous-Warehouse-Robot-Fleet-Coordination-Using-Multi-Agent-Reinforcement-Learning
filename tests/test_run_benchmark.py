import json
from pathlib import Path

import torch
import pytest

from src.evaluation.run_benchmark import run_benchmark
from src.evaluation.run_evaluation import load_episode_records
from src.evaluation.scenario_manifest import create_evaluation_manifest, write_evaluation_manifest
from src.marl.iql_trainer import IQLTrainer
from src.marl.qmix_trainer import QMIXTrainer


def test_benchmark_writes_matched_traces_manifest_and_statistics(tmp_path):
    iql_dir = tmp_path / "iql"
    iql = IQLTrainer(
        num_robots=5, episodes=1, max_steps=1, seed=7,
        include_static_map=True, include_path_costs=True,
    )
    iql.save_agents(str(iql_dir))

    qmix = QMIXTrainer(num_agents=5, episodes=1, max_steps=1, seed=7)
    qmix_path = tmp_path / "qmix.pt"
    torch.save({
        "agent_network": qmix.learner.agent_network.state_dict(),
        "mixer": qmix.learner.mixer.state_dict(),
        "config": {"num_agents": 5, "observation_dim": 456, "state_dim": 55},
        "episode": 1,
        "seed": 7,
    }, qmix_path)

    report = run_benchmark(
        fleet_sizes=[5],
        algorithms=["QMIX", "IQL", "Rule-Based"],
        episodes=1,
        max_steps=1,
        seed=42,
        qmix_checkpoints={5: qmix_path},
        iql_checkpoints={5: iql_dir},
        output_dir=tmp_path / "evaluation",
        bootstrap_resamples=20,
        permutation_samples=20,
    )

    assert len(report["summary"]) == 3
    assert len(report["manifest"]["runs"]) == 3
    assert all("trace_sha256" in run for run in report["manifest"]["runs"])
    assert Path(report["manifest"]["runs"][0]["checkpoint"]["path"]).is_absolute()
    assert report["manifest"]["scalability_plot"]["file"] == "scalability.png"
    assert (tmp_path / "evaluation" / "scalability.png").stat().st_size > 0
    assert report["safety_gate"]["passed"] is True
    assert report["safety_gate"]["episodes"] == 20
    assert report["safety_gate"]["intrusions"] == 0
    report_text = (tmp_path / "evaluation" / "report.md").read_text()
    assert report_text.startswith(
        "# Warehouse Fleet Scalability Evaluation"
    )
    assert "1 matched episodes" in report_text
    assert "N=5 QMIX completion target (>=80%): FAIL" in report_text
    assert "No pre-created scenario manifest was supplied" in report_text
    assert "N=10 physical QMIX/IQL collision-rate comparison was not included" in report_text
    assert "point estimate meets the 80%" not in report_text
    trace_path = tmp_path / "evaluation" / "traces" / "qmix_n5.json.gz"
    assert trace_path.is_file()
    assert len(load_episode_records([trace_path])) == 1
    for name in ("manifest.json", "summary.json"):
        json.loads((tmp_path / "evaluation" / name).read_text())


def test_benchmark_consumes_and_copies_frozen_scenario_manifest(tmp_path):
    scenario_manifest = create_evaluation_manifest(
        fleet_sizes=[5], episodes=20, max_steps=1, seed=976304181
    )
    scenario_path = write_evaluation_manifest(
        scenario_manifest, tmp_path / "sealed_scenarios.json"
    )
    report = run_benchmark(
        fleet_sizes=[5],
        algorithms=["Rule-Based"],
        episodes=20,
        max_steps=1,
        seed=0,
        qmix_checkpoints={},
        iql_checkpoints={},
        output_dir=tmp_path / "scenario_evaluation",
        scenario_manifest_path=scenario_path,
        bootstrap_resamples=20,
        permutation_samples=20,
    )
    assert report["manifest"]["scenario_manifest_id"] == scenario_manifest["manifest_id"]
    assert report["scenario_manifest_id"] == scenario_manifest["manifest_id"]
    assert (tmp_path / "scenario_evaluation" / "scenario_manifest.json").is_file()
    trace = load_episode_records([
        tmp_path / "scenario_evaluation" / "traces" / "rule_based_n5.json.gz"
    ])[0]
    assert trace.task_seed == scenario_manifest["scenarios"]["5"][0]["task_seed"]
    assert report["summary"][0]["episodes"] == 20
    assert f"Used the supplied frozen scenario manifest `{scenario_manifest['manifest_id']}`" in (
        (tmp_path / "scenario_evaluation" / "report.md").read_text()
    )


def test_benchmark_rejects_mismatched_learned_observation_dimensions(tmp_path):
    iql_dir = tmp_path / "iql"
    iql = IQLTrainer(num_robots=5, episodes=1, max_steps=1, seed=7)
    iql.save_agents(str(iql_dir))
    qmix = QMIXTrainer(num_agents=5, episodes=1, max_steps=1, seed=7)
    qmix_path = tmp_path / "qmix.pt"
    torch.save({
        "agent_network": qmix.learner.agent_network.state_dict(),
        "mixer": qmix.learner.mixer.state_dict(),
        "config": {"num_agents": 5, "observation_dim": 456},
        "episode": 1,
        "seed": 7,
    }, qmix_path)

    with pytest.raises(ValueError, match="observation dimensions differ"):
        run_benchmark(
            fleet_sizes=[5], algorithms=["QMIX", "IQL"], episodes=1,
            max_steps=1, seed=42, qmix_checkpoints={5: qmix_path},
            iql_checkpoints={5: iql_dir}, output_dir=tmp_path / "mismatch",
            bootstrap_resamples=20, permutation_samples=20,
        )
