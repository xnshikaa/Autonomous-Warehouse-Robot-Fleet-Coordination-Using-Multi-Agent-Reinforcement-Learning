from src.evaluation.metrics import (
    EpisodeRecord,
    evaluate_episodes,
    infer_robot_robot_collisions,
)


def _episode(episode=1, collisions=(), overrides=0):
    return EpisodeRecord(
        algorithm="QMIX",
        fleet_size=5,
        episode=episode,
        task_snapshots=[
            [
                {"task_id": "a", "assigned_robot": 0, "completed": False},
                {"task_id": "b", "assigned_robot": 1, "completed": False},
            ],
            [
                {"task_id": "a", "assigned_robot": 0, "completed": True},
                {"task_id": "b", "assigned_robot": 1, "completed": False},
            ],
            [
                {"task_id": "a", "assigned_robot": 0, "completed": True},
                {"task_id": "b", "assigned_robot": 1, "completed": True},
            ],
        ],
        collision_events=collisions,
        safety_overrides=overrides,
    )


def test_completion_time_and_unfinished_tasks():
    row = evaluate_episodes([_episode()])[0]
    assert row["completed_tasks"] == 2
    assert row["completion_rate"] == 1.0
    assert row["average_completion_time_steps"] == 1.5
    assert row["unfinished_tasks"] == 0


def test_multiple_episodes_and_collision_aggregation():
    rows = evaluate_episodes([
        _episode(collisions=[{"type": "robot_robot"}], overrides=2),
        _episode(
            episode=2,
            collisions=[{"type": "robot_robot"}, {"type": "obstacle"}],
            overrides=2,
        ),
    ])
    row = rows[0]
    assert row["episodes"] == 2
    assert row["collision_events"] == 3
    assert row["robot_robot_collision_events"] == 2
    assert row["collision_rate_per_episode"] == 1.5
    assert row["safety_overrides"] == 4


def test_empty_episode_is_deterministic():
    row = evaluate_episodes([
        EpisodeRecord("IQL", 10, 1),
    ])[0]
    assert row["tasks"] == 0
    assert row["average_completion_time_steps"] is None
    assert row["collision_events"] == 0


def test_collision_inference_matches_environment_rules():
    events = infer_robot_robot_collisions(
        {0: (0, 0), 1: (2, 0), 2: (0, 1)},
        {0: (1, 0), 1: (1, 0), 2: (0, 1)},
    )
    assert events == [{"type": "robot_robot", "robots": [0, 1]}]


def test_safety_override_is_not_a_collision():
    row = evaluate_episodes([_episode(overrides=3)])[0]
    assert row["safety_overrides"] == 3
    assert row["collision_events"] == 0
