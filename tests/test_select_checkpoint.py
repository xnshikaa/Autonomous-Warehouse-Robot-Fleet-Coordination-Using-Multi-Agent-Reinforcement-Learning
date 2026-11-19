import json

from src.evaluation import select_checkpoint as selector


def _trace(episode, completed, candidate):
    return {
        "algorithm": "QMIX",
        "fleet_size": 5,
        "episode": episode,
        "task_seed": 100 + episode,
        "task_snapshots": [
            [{"task_id": 1, "assigned_robot": 0, "completed": False}],
            [{"task_id": 1, "assigned_robot": 0, "completed": completed}],
        ],
        "collision_events": [],
        "safety_overrides": 0,
        "safety_conflict_events": [],
        "collision_intent_groups": [],
        "steps": 1,
        "team_reward": 1.0,
        "candidate": candidate,
    }


def test_checkpoint_selector_uses_identical_validation_and_ranks_completion(tmp_path, monkeypatch):
    high = tmp_path / "high.pt"
    low = tmp_path / "low.pt"
    high.write_bytes(b"high")
    low.write_bytes(b"low")

    def fake_generate(algorithm, fleet_size, episodes, max_steps, seed, checkpoint):
        complete = checkpoint.name == "high.pt"
        return [_trace(i, complete, checkpoint.name) for i in range(1, episodes + 1)]

    monkeypatch.setattr(selector, "generate_traces", fake_generate)
    output = tmp_path / "selection.json"
    result = selector.select_checkpoint(
        algorithm="QMIX", fleet_size=5, candidates=[low, high],
        episodes=2, max_steps=2, seed=321, output=output,
    )

    assert result["selected_checkpoint"] == str(high)
    assert result["purpose"].endswith("not final holdout")
    assert result["results"][0]["completion_rate"] == 1.0
    assert json.loads(output.read_text())["seed"] == 321


def test_selector_expands_snapshot_directory(tmp_path):
    root = tmp_path / "snapshots"
    first = root / "episode_00100" / "checkpoint_state.pt"
    second = root / "episode_00200" / "checkpoint_state.pt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert selector._expand_candidates("QMIX", [root]) == [first, second]
