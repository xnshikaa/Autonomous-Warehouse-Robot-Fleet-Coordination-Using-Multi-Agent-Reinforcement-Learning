import torch

from src.evaluation.metrics import evaluate_episodes
from src.evaluation.run_evaluation import load_episode_records
from src.evaluation.trace_generator import generate_rule_based, generate_traces, write_traces
from src.marl.iql_trainer import IQLTrainer
from src.marl.qmix_trainer import QMIXTrainer


def test_rule_based_traces_use_reproducible_fleet_task_sets():
    first = generate_rule_based(5, episodes=1, max_steps=1, seed=42)[0]
    repeated = generate_rule_based(5, episodes=1, max_steps=1, seed=42)[0]
    other_seed = generate_rule_based(5, episodes=1, max_steps=1, seed=43)[0]

    assert first["task_seed"] == 42
    assert first["initial_tasks"] == repeated["initial_tasks"]
    assert first["initial_tasks"] != other_seed["initial_tasks"]
    assert len(first["initial_tasks"]) == 5
    assert {task["assigned_robot"] for task in first["initial_tasks"]} == set(range(5))


def test_qmix_iql_and_rule_traces_share_tasks_and_execute_actions(tmp_path):
    iql = IQLTrainer(num_robots=5, episodes=1, max_steps=1, seed=7)
    iql.train()
    iql_checkpoint = tmp_path / "iql"
    iql.save_agents(str(iql_checkpoint))

    qmix = QMIXTrainer(num_agents=5, episodes=1, max_steps=1, seed=7)
    qmix_checkpoint = tmp_path / "qmix.pt"
    torch.save({
        "agent_network": qmix.learner.agent_network.state_dict(),
        "mixer": qmix.learner.mixer.state_dict(),
    }, qmix_checkpoint)

    traces = {
        "Rule-Based": generate_traces("rule", 5, 1, 2, 42)[0],
        "QMIX": generate_traces("QMIX", 5, 1, 2, 42, qmix_checkpoint)[0],
        "IQL": generate_traces("IQL", 5, 1, 2, 42, iql_checkpoint)[0],
    }

    task_sets = [trace["initial_tasks"] for trace in traces.values()]
    assert task_sets[0] == task_sets[1] == task_sets[2]
    for algorithm, trace in traces.items():
        assert trace["algorithm"] == algorithm
        assert len(trace["action_trace"]) > 0
        assert set(trace["action_trace"][0]["requested_actions"]) == set(range(5))

    assert traces["QMIX"]["collision_events"] == []
    trace_file = tmp_path / "all_algorithms.json"
    write_traces(list(traces.values()), trace_file)
    records = load_episode_records([trace_file])
    rows = evaluate_episodes(records)
    assert {row["algorithm"] for row in rows} == {"QMIX", "IQL", "Rule-Based"}


def test_learned_policy_trace_requires_checkpoint():
    import pytest

    with pytest.raises(ValueError, match="checkpoint is required"):
        generate_traces("QMIX", 5, 1, 1, 42)
