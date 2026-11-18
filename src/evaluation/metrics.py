"""Deterministic Week 5 completion-time and collision metrics.

The evaluator works on episode traces rather than a particular controller.
This keeps QMIX, IQL, and Rule-Based runs on the same metric definitions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskTiming:
    """Timing for one task, measured in environment steps."""

    task_id: str | int
    start_step: int
    completion_step: int | None
    completed: bool

    @property
    def duration_steps(self) -> int | None:
        if not self.completed or self.completion_step is None:
            return None
        return self.completion_step - self.start_step


@dataclass(frozen=True)
class EpisodeRecord:
    """One completed or truncated evaluation episode.

    ``task_snapshots`` contains one snapshot per environment step. Each
    snapshot is a mapping with ``task_id``, ``assigned_robot`` and
    ``completed`` fields, or an object exposing those attributes.
    ``collision_events`` may contain one item per event, or a mapping with
    ``type``/``robots`` fields. Safety overrides are intentionally separate.
    """

    algorithm: str
    fleet_size: int
    episode: int
    task_snapshots: Sequence[Sequence[Any]] = field(default_factory=tuple)
    collision_events: Sequence[Any] = field(default_factory=tuple)
    safety_overrides: int = 0
    safety_conflict_events: Sequence[Any] = field(default_factory=tuple)
    task_seed: int | str | None = None
    steps: int | None = None
    team_reward: float | None = None
    manifest_id: str | None = None
    layout_id: str | None = None
    policy_checkpoint_id: str | None = None
    environment_config_hash: str | None = None
    max_steps: int | None = None


@dataclass(frozen=True)
class CompletionMetrics:
    algorithm: str
    fleet_size: int
    episodes: int
    tasks: int
    completed_tasks: int
    completion_rate: float
    average_completion_time_steps: float | None
    unfinished_tasks: int


@dataclass(frozen=True)
class CollisionMetrics:
    algorithm: str
    fleet_size: int
    episodes: int
    collision_events: int
    collision_rate_per_episode: float
    robot_robot_collision_events: int
    safety_overrides: int
    safety_conflict_events: int = 0


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _task_key(task: Any) -> str | int | None:
    return _value(task, "task_id", _value(task, "id"))


def _task_timing(snapshot_steps: Sequence[Sequence[Any]]) -> list[TaskTiming]:
    starts: dict[str | int, int] = {}
    completions: dict[str | int, int] = {}
    task_ids: set[str | int] = set()

    for step, snapshot in enumerate(snapshot_steps, start=1):
        for task in snapshot:
            task_id = _task_key(task)
            if task_id is None:
                continue
            task_ids.add(task_id)
            assigned = _value(task, "assigned_robot", _value(task, "assigned_robot_id"))
            if task_id not in starts:
                starts[task_id] = step
            if bool(_value(task, "completed", False)) and task_id not in completions:
                completions[task_id] = step

    return [
        TaskTiming(
            task_id=task_id,
            start_step=starts[task_id],
            completion_step=completions.get(task_id),
            completed=task_id in completions,
        )
        for task_id in sorted(task_ids, key=str)
        if task_id in starts
    ]


def _collision_type(event: Any) -> str:
    return str(_value(event, "type", _value(event, "collision_type", "unknown"))).lower()


def _episode_metric(record: EpisodeRecord, metric: str) -> float | None:
    timings = _task_timing(record.task_snapshots)
    completed = [timing.duration_steps for timing in timings if timing.completed]
    completed = [duration for duration in completed if duration is not None]
    steps = record.steps
    if steps is None:
        steps = max(0, len(record.task_snapshots) - 1)
    values = {
        "completion_rate": sum(timing.completed for timing in timings) / len(timings) if timings else None,
        "average_completion_time_steps": mean(completed) if completed else None,
        "completed_task_makespan_steps": float(max(completed)) if completed else None,
        "throughput_tasks_per_100_steps": 100.0 * len(completed) / steps if steps else 0.0,
        "collision_rate_per_episode": float(len(record.collision_events)),
        "safety_conflict_events": float(len(record.safety_conflict_events)),
        "safety_overrides": float(record.safety_overrides),
        "team_reward": record.team_reward,
    }
    if metric not in values:
        raise ValueError(f"Unsupported comparison metric: {metric}")
    return values[metric]


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Percentile bootstrap CI for an episode-level mean."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("values must be a non-empty finite 1-D sequence")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    means = data[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(data.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "confidence_level": confidence_level,
        "samples": int(data.size),
        "resamples": resamples,
    }


def paired_algorithm_comparison(
    episodes: Iterable[EpisodeRecord],
    algorithm_a: str,
    algorithm_b: str,
    fleet_size: int,
    *,
    metric: str = "completion_rate",
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    permutation_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare episode-level metrics on identical task seeds using paired resampling."""
    if resamples <= 0 or permutation_samples <= 0:
        raise ValueError("resamples and permutation_samples must be positive")
    records = [r for r in episodes if r.fleet_size == fleet_size and r.algorithm in {algorithm_a, algorithm_b}]
    grouped: dict[str, dict[Any, EpisodeRecord]] = {algorithm_a: {}, algorithm_b: {}}
    for record in records:
        scenario_key = record.task_seed if record.task_seed is not None else record.episode
        key = (record.manifest_id, scenario_key) if record.manifest_id is not None else scenario_key
        if key in grouped[record.algorithm]:
            raise ValueError(f"Duplicate paired episode key {key!r} for {record.algorithm}")
        grouped[record.algorithm][key] = record
    if set(grouped[algorithm_a]) != set(grouped[algorithm_b]) or not grouped[algorithm_a]:
        raise ValueError("Both algorithms must have the same non-empty task-seed set")

    keys = sorted(grouped[algorithm_a], key=str)
    differences = []
    for key in keys:
        a = _episode_metric(grouped[algorithm_a][key], metric)
        b = _episode_metric(grouped[algorithm_b][key], metric)
        if a is None or b is None:
            continue
        differences.append(a - b)
    if not differences:
        raise ValueError(f"No paired observations are available for metric {metric!r}")

    diff = np.asarray(differences, dtype=np.float64)
    interval = bootstrap_confidence_interval(
        diff, confidence_level=confidence_level, resamples=resamples, seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    observed = abs(float(diff.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutation_samples, diff.size))
    null_means = np.abs((signs * diff).mean(axis=1))
    p_value = (1 + int(np.count_nonzero(null_means >= observed))) / (permutation_samples + 1)
    return {
        "algorithm_a": algorithm_a,
        "algorithm_b": algorithm_b,
        "fleet_size": fleet_size,
        "metric": metric,
        "difference_a_minus_b": float(diff.mean()),
        "paired_bootstrap_ci": interval,
        "paired_sign_flip_p_value": float(p_value),
        "paired_episodes": int(diff.size),
        "excluded_undefined_pairs": int(len(keys) - diff.size),
        "permutation_samples": permutation_samples,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("p_values must be a non-empty finite 1-D sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be between 0 and 1")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def evaluate_episodes(episodes: Iterable[EpisodeRecord]) -> list[dict[str, Any]]:
    """Return one structured metric row per algorithm/fleet-size group."""

    records = list(episodes)
    groups: dict[tuple[str, int], list[EpisodeRecord]] = {}
    for record in records:
        if not record.algorithm:
            raise ValueError("algorithm must not be empty")
        if record.fleet_size <= 0:
            raise ValueError("fleet_size must be positive")
        groups.setdefault((record.algorithm, record.fleet_size), []).append(record)

    rows: list[dict[str, Any]] = []
    for (algorithm, fleet_size), group in sorted(groups.items()):
        timings = [
            timing
            for record in group
            for timing in _task_timing(record.task_snapshots)
        ]
        durations = [
            timing.duration_steps
            for timing in timings
            if timing.duration_steps is not None
        ]
        events = [
            event
            for record in group
            for event in record.collision_events
        ]
        robot_events = [
            event
            for event in events
            if _collision_type(event) in {"robot_robot", "robot-robot", "robot_robot_collision"}
        ]
        conflict_count = sum(len(record.safety_conflict_events) for record in group)
        total_steps = sum(
            record.steps if record.steps is not None
            else max(0, len(record.task_snapshots) - 1)
            for record in group
        )
        makespans = [
            value for record in group
            if (value := _episode_metric(record, "completed_task_makespan_steps")) is not None
        ]
        episode_rewards = [record.team_reward for record in group if record.team_reward is not None]
        episode_completion_rates = [
            value for record in group
            if (value := _episode_metric(record, "completion_rate")) is not None
        ]
        episode_delivery_times = [
            value for record in group
            if (value := _episode_metric(record, "average_completion_time_steps")) is not None
        ]
        episode_throughputs = [
            value for record in group
            if (value := _episode_metric(record, "throughput_tasks_per_100_steps")) is not None
        ]
        completion = CompletionMetrics(
            algorithm=algorithm,
            fleet_size=fleet_size,
            episodes=len(group),
            tasks=len(timings),
            completed_tasks=sum(timing.completed for timing in timings),
            completion_rate=mean(episode_completion_rates) if episode_completion_rates else 0.0,
            average_completion_time_steps=mean(episode_delivery_times) if episode_delivery_times else None,
            unfinished_tasks=sum(not timing.completed for timing in timings),
        )
        collision = CollisionMetrics(
            algorithm=algorithm,
            fleet_size=fleet_size,
            episodes=len(group),
            collision_events=len(events),
            collision_rate_per_episode=len(events) / len(group),
            robot_robot_collision_events=len(robot_events),
            safety_overrides=sum(record.safety_overrides for record in group),
            safety_conflict_events=conflict_count,
        )
        rows.append(
            {
                "algorithm": algorithm,
                "fleet_size": fleet_size,
                "episodes": completion.episodes,
                "tasks": completion.tasks,
                "completed_tasks": completion.completed_tasks,
                "completion_rate": completion.completion_rate,
                "average_completion_time_steps": completion.average_completion_time_steps,
                "mean_completed_task_makespan_steps": mean(makespans) if makespans else None,
                "unfinished_tasks": completion.unfinished_tasks,
                "collision_events": collision.collision_events,
                "collision_rate_per_episode": collision.collision_rate_per_episode,
                "robot_robot_collision_events": collision.robot_robot_collision_events,
                "safety_overrides": collision.safety_overrides,
                "safety_conflict_events": collision.safety_conflict_events,
                "safety_conflict_rate_per_episode": collision.safety_conflict_events / len(group),
                "throughput_tasks_per_100_steps": mean(episode_throughputs) if episode_throughputs else 0.0,
                "task_weighted_completion_rate": (
                    completion.completed_tasks / len(timings) if timings else None
                ),
                "mean_episode_reward": mean(episode_rewards) if episode_rewards else None,
            }
        )
    return rows


def infer_robot_robot_collisions(
    previous_positions: Mapping[int, tuple[int, int]],
    proposed_positions: Mapping[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Apply the environment's same-target/direct-swap collision semantics.

    Each timestep produces one event per distinct conflict, not one event per
    robot. This helper is for adapters recording raw environment positions.
    """

    events: set[tuple[int, ...]] = set()
    by_target: dict[tuple[int, int], list[int]] = {}
    for robot_id, position in proposed_positions.items():
        by_target.setdefault(position, []).append(robot_id)
    for robot_ids in by_target.values():
        if len(robot_ids) > 1:
            events.add(tuple(sorted(robot_ids)))

    robot_ids = sorted(proposed_positions)
    for index, first in enumerate(robot_ids):
        for second in robot_ids[index + 1 :]:
            if (
                proposed_positions[first] == previous_positions[second]
                and proposed_positions[second] == previous_positions[first]
                and proposed_positions[first] != previous_positions[first]
            ):
                events.add((first, second))

    return [{"type": "robot_robot", "robots": list(conflict)} for conflict in sorted(events)]
