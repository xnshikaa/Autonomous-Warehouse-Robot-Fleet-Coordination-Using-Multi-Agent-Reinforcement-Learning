"""Generate reproducible episode traces for Week 5/Week 6 evaluation.

The generator records environment state after every step. It intentionally
does not alter the warehouse environment or algorithm implementations.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.ai.environment import WarehouseEnvironment
from src.ai.baselines.rule_based import RuleBasedController
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.iql_learner import IQLLearner
from src.marl.observation_encoder import ObservationEncoder
from src.marl.qmix_controller import QMIXController
from src.evaluation.metrics import infer_robot_robot_collisions
from src.warehouse.task_generation import generate_task_assignments


def _task_snapshot(environment: WarehouseEnvironment) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "assigned_robot": task.assigned_robot,
            "completed": task.completed,
        }
        for task in environment.tasks
    ]


def _add_seeded_tasks(environment: WarehouseEnvironment, seed: int) -> list[dict[str, Any]]:
    """Create safe tasks reproducibly so policies can share episode inputs."""
    tasks = generate_task_assignments(environment, random.Random(seed))
    for task in tasks:
        environment.add_task(
            task.task_id,
            task.pickup_position,
            task.delivery_position,
            task.assigned_robot,
        )
    return [
        {
            "task_id": task.task_id,
            "assigned_robot": task.assigned_robot,
            "pickup_position": list(task.pickup_position),
            "delivery_position": list(task.delivery_position),
        }
        for task in tasks
    ]


def _record_episode(
    algorithm: str,
    fleet_size: int,
    episode_number: int,
    max_steps: int,
    task_seed: int,
    action_provider: Callable[[WarehouseEnvironment], dict[int, int]],
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    environment = WarehouseEnvironment(
        num_robots=fleet_size,
        max_steps=max_steps,
    )
    environment.logger.setLevel(logging.ERROR)
    environment.reset()
    if scenario is None:
        initial_tasks = _add_seeded_tasks(environment, task_seed)
    else:
        if scenario.get("layout_id") != environment.get_active_layout().layout_id:
            raise ValueError("Scenario layout does not match the active warehouse layout")
        initial_tasks = scenario.get("initial_tasks")
        if not isinstance(initial_tasks, list) or not initial_tasks:
            raise ValueError("Scenario must contain a non-empty initial_tasks list")
        for task in initial_tasks:
            environment.add_task(
                task["task_id"], task["pickup_position"], task["delivery_position"],
                task.get("assigned_robot"),
            )

    task_snapshots: list[list[dict[str, Any]]] = [_task_snapshot(environment)]
    collision_events: list[dict[str, Any]] = []
    safety_conflict_events: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    steps = 0
    terminated = False
    episode_team_reward = 0.0

    while not terminated and steps < max_steps:
        actions = action_provider(environment)
        previous_positions = {
            robot.robot_id: tuple(robot.position) for robot in environment.robots
        }
        _, rewards, terminated, _ = environment.step(actions)
        steps += 1
        episode_team_reward += float(np.mean(list(rewards.values()))) if rewards else 0.0
        action_trace.append({
            "step": steps,
            "requested_actions": actions,
            "executed_actions": {
                robot.robot_id: robot.last_executed_action
                for robot in environment.robots
            },
            "positions": {
                robot.robot_id: list(robot.position)
                for robot in environment.robots
            },
            "agent_rewards": rewards,
        })
        actual_positions = {
            robot.robot_id: tuple(robot.position) for robot in environment.robots
        }
        collision_events.extend(
            {**event, "step": steps}
            for event in infer_robot_robot_collisions(previous_positions, actual_positions)
        )
        safety_conflict_events.extend(
            {
                "type": "robot_robot_conflict",
                "status": "prevented_by_safety",
                "robot_id": event.robot_id,
                "step": steps,
                "current_cell": list(event.current_cell),
                "target_cell": None if event.target_cell is None else list(event.target_cell),
            }
            for event in environment.get_safety_events()
            if event.reason == "robot_collision"
        )
        task_snapshots.append(_task_snapshot(environment))

    return {
        "algorithm": algorithm,
        "fleet_size": fleet_size,
        "episode": episode_number,
        "task_seed": task_seed,
        "layout_id": environment.get_active_layout().layout_id,
        "initial_tasks": initial_tasks,
        "action_trace": action_trace,
        "task_snapshots": task_snapshots,
        "collision_events": collision_events,
        "safety_conflict_events": safety_conflict_events,
        "safety_overrides": environment.safety_overrides,
        "steps": steps,
        "max_steps": max_steps,
        "team_reward": episode_team_reward,
    }


def _rule_based_provider(environment: WarehouseEnvironment) -> dict[int, int]:
    return RuleBasedController(environment).get_actions()


def _qmix_provider(checkpoint: str | Path, fleet_size: int):
    path = Path(checkpoint)
    if path.is_dir():
        candidates = [path / "checkpoint_state.pt", path / "checkpoint_state.pkl"]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"QMIX checkpoint file not found: {path}")
    holder: dict[str, QMIXController] = {}

    def provide(environment: WarehouseEnvironment) -> dict[int, int]:
        if "controller" not in holder:
            adapter = MARLEnvironmentAdapter(
                num_agents=fleet_size,
                max_steps=environment.max_steps,
                environment=environment,
            )
            holder["controller"] = QMIXController(adapter, checkpoint_path=path)
        else:
            holder["controller"].adapter.environment = environment
        return holder["controller"].select_actions()

    return provide


def _iql_provider(checkpoint: str | Path, fleet_size: int):
    directory = Path(checkpoint)
    if not directory.is_dir():
        raise FileNotFoundError(f"IQL checkpoint directory not found: {directory}")
    learners = [IQLLearner(observation_dim=50, action_dim=4) for _ in range(fleet_size)]
    for robot_id, learner in enumerate(learners):
        learner.load(str(directory / f"iql_robot_{robot_id}.pt"))

    def provide(environment: WarehouseEnvironment) -> dict[int, int]:
        layout = environment.get_active_layout()
        encoder = ObservationEncoder(layout.width, layout.height)
        actions = {}
        for robot in environment.robots:
            observation = torch.as_tensor(
                encoder.encode(robot, environment.robots, environment.tasks, layout=layout),
                dtype=torch.float32,
            )
            actions[robot.robot_id] = learners[robot.robot_id].select_action(
                observation, epsilon=0.0
            )
        return actions

    return provide


def generate_rule_based(
    fleet_size: int,
    episodes: int,
    max_steps: int,
    seed: int,
    scenarios: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if scenarios is not None and len(scenarios) != episodes:
        raise ValueError("scenario count must match episodes")
    random.seed(seed)
    np.random.seed(seed)
    traces = [
        _record_episode(
            "Rule-Based",
            fleet_size,
            episode,
            max_steps,
            scenarios[episode - 1]["task_seed"] if scenarios else seed + episode - 1,
            _rule_based_provider,
            scenarios[episode - 1] if scenarios else None,
        )
        for episode in range(1, episodes + 1)
    ]
    return traces


def generate_traces(
    algorithm: str,
    fleet_size: int,
    episodes: int,
    max_steps: int,
    seed: int,
    checkpoint: str | Path | None = None,
    scenarios: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate seeded traces by running the requested policy checkpoint."""
    if scenarios is not None and len(scenarios) != episodes:
        raise ValueError("scenario count must match episodes")
    normalized = algorithm.lower().replace("_", "-")
    if normalized in {"rule-based", "rule"}:
        return generate_rule_based(fleet_size, episodes, max_steps, seed, scenarios)
    if checkpoint is None:
        raise ValueError(f"A trained {algorithm} checkpoint is required for evaluation.")
    if normalized == "qmix":
        provider = _qmix_provider(checkpoint, fleet_size)
        canonical = "QMIX"
    elif normalized == "iql":
        provider = _iql_provider(checkpoint, fleet_size)
        canonical = "IQL"
    else:
        raise ValueError(f"Unsupported evaluation algorithm: {algorithm}")

    traces = [
        _record_episode(
            canonical,
            fleet_size,
            episode,
            max_steps,
            scenarios[episode - 1]["task_seed"] if scenarios else seed + episode - 1,
            provider,
            scenarios[episode - 1] if scenarios else None,
        )
        for episode in range(1, episodes + 1)
    ]
    for trace in traces:
        trace["policy_checkpoint_id"] = str(checkpoint)
    return traces


def write_traces(
    traces: list[dict[str, Any]],
    output_path: str | Path,
    manifest: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps({"manifest": manifest or {}, "episodes": traces}, indent=2)
    if destination.suffix == ".gz":
        with gzip.open(destination, "wt", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        destination.write_text(serialized, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--fleet-size", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")

    traces = generate_traces(
        args.algorithm,
        args.fleet_size,
        args.episodes,
        args.max_steps,
        args.seed,
        args.checkpoint,
    )
    write_traces(traces, args.output)
    print(f"Wrote {len(traces)} episode traces to {args.output}")


if __name__ == "__main__":
    main()
