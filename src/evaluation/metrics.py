"""Deterministic Week 5 completion-time and collision metrics.

The evaluator works on episode traces rather than a particular controller.
This keeps QMIX, IQL, and Rule-Based runs on the same metric definitions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


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
            if assigned is not None and task_id not in starts:
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
    return str(_value(event, "type", _value(event, "collision_type", "robot_robot"))).lower()


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
        completion = CompletionMetrics(
            algorithm=algorithm,
            fleet_size=fleet_size,
            episodes=len(group),
            tasks=len(timings),
            completed_tasks=sum(timing.completed for timing in timings),
            completion_rate=(
                sum(timing.completed for timing in timings) / len(timings)
                if timings else 0.0
            ),
            average_completion_time_steps=mean(durations) if durations else None,
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
                "unfinished_tasks": completion.unfinished_tasks,
                "collision_events": collision.collision_events,
                "collision_rate_per_episode": collision.collision_rate_per_episode,
                "robot_robot_collision_events": collision.robot_robot_collision_events,
                "safety_overrides": collision.safety_overrides,
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

    return [{"type": "robot_robot", "robots": list(robot_ids)} for robot_ids in sorted(events)]
