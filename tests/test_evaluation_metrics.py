import pytest

from src.evaluation.metrics import (
    EpisodeRecord,
    bootstrap_confidence_interval,
    evaluate_episodes,
    holm_adjust,
    infer_robot_robot_collisions,
    paired_algorithm_comparison,
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
    assert row["completion_rate"] == 0.0
    assert row["collision_events"] == 0


def test_collision_inference_matches_environment_rules():
    events = infer_robot_robot_collisions(
        {0: (0, 0), 1: (2, 0), 2: (0, 1)},
        {0: (1, 0), 1: (1, 0), 2: (0, 1)},
    )
    assert events == [{"type": "robot_robot", "robots": [0, 1]}]


def test_delivery_and_throughput_summary_use_episode_level_estimands():
    short = EpisodeRecord(
        algorithm="QMIX", fleet_size=5, episode=1, steps=3,
        task_snapshots=[
            [{"task_id": "a", "completed": False}],
            [{"task_id": "a", "completed": True}],
        ],
    )
    long = EpisodeRecord(
        algorithm="QMIX", fleet_size=5, episode=2, steps=3,
        task_snapshots=[
            [{"task_id": task, "completed": False} for task in "bcd"],
            [{"task_id": task, "completed": False} for task in "bcd"],
            [{"task_id": task, "completed": True} for task in "bcd"],
        ],
    )
    row = evaluate_episodes([short, long])[0]
    assert row["average_completion_time_steps"] == 1.5
    assert row["throughput_tasks_per_100_steps"] == pytest.approx(100 * (1 / 3 + 1) / 2)
    assert row["task_weighted_completion_rate"] == 1.0


def test_safety_override_is_not_a_collision():
    row = evaluate_episodes([_episode(overrides=3)])[0]
    assert row["safety_overrides"] == 3
    assert row["collision_events"] == 0


def test_safety_prevented_robot_conflicts_are_separate_from_collisions():
    record = _episode()
    record = EpisodeRecord(
        algorithm=record.algorithm,
        fleet_size=record.fleet_size,
        episode=record.episode,
        task_snapshots=record.task_snapshots,
        safety_conflict_events=[{"robot_id": 0}, {"robot_id": 1}],
    )
    row = evaluate_episodes([record])[0]
    assert row["collision_events"] == 0
    assert row["safety_conflict_events"] == 2
    assert row["safety_conflict_rate_per_episode"] == 2.0


def _paired_record(algorithm, episode, task_seed, completed):
    return EpisodeRecord(
        algorithm=algorithm,
        fleet_size=5,
        episode=episode,
        task_seed=task_seed,
        steps=2,
        task_snapshots=[
            [{"task_id": 1, "assigned_robot": 0, "completed": False}],
            [{"task_id": 1, "assigned_robot": 0, "completed": bool(completed)}],
        ],
    )


def test_bootstrap_mean_ci_is_deterministic_and_validates_inputs():
    interval = bootstrap_confidence_interval([0.0, 1.0], resamples=100, seed=7)
    assert interval["estimate"] == 0.5
    assert interval["lower"] <= 0.5 <= interval["upper"]
    assert interval == bootstrap_confidence_interval([0.0, 1.0], resamples=100, seed=7)


def test_paired_comparison_matches_common_task_seeds():
    records = [
        _paired_record("QMIX", 1, 101, True),
        _paired_record("QMIX", 2, 102, True),
        _paired_record("IQL", 1, 101, False),
        _paired_record("IQL", 2, 102, False),
    ]
    result = paired_algorithm_comparison(
        records, "QMIX", "IQL", 5, resamples=100, permutation_samples=200, seed=8
    )
    assert result["paired_episodes"] == 2
    assert result["difference_a_minus_b"] == 1.0
    assert result["paired_bootstrap_ci"]["lower"] == 1.0
    assert result["paired_bootstrap_ci"]["upper"] == 1.0


def test_paired_comparison_rejects_unmatched_task_seeds():
    records = [_paired_record("QMIX", 1, 101, True), _paired_record("IQL", 1, 202, False)]
    with pytest.raises(ValueError, match="same non-empty task-seed set"):
        paired_algorithm_comparison(records, "QMIX", "IQL", 5, resamples=10)


def test_holm_adjustment_preserves_order_and_controls_familywise_error():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
