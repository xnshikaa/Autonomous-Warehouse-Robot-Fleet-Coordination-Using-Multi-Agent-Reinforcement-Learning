"""Generate reproducible episode traces for Week 5/Week 6 evaluation.

The generator records environment state after every step. It intentionally
does not alter the warehouse environment or algorithm implementations.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.ai.environment import WarehouseEnvironment
from src.ai.baselines.rule_based import RuleBasedController
from src.evaluation.metrics import infer_robot_robot_collisions


def _task_snapshot(environment: WarehouseEnvironment) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "assigned_robot": task.assigned_robot,
            "completed": task.completed,
        }
        for task in environment.tasks
    ]


def _add_default_tasks(environment: WarehouseEnvironment) -> None:
    """Create the deterministic task layout used by the existing baseline."""

    for robot in environment.robots:
        row = robot.robot_id % environment.warehouse_height
        environment.add_task(
            task_id=robot.robot_id,
            pickup_position=(5, row),
            delivery_position=(10, row),
            assigned_robot=robot.robot_id,
        )


def _record_episode(
    algorithm: str,
    fleet_size: int,
    episode_number: int,
    max_steps: int,
    action_provider: Callable[[WarehouseEnvironment], dict[int, int]],
) -> dict[str, Any]:
    environment = WarehouseEnvironment(
        num_robots=fleet_size,
        max_steps=max_steps,
    )
    environment.reset()
    _add_default_tasks(environment)

    task_snapshots: list[list[dict[str, Any]]] = [_task_snapshot(environment)]
    collision_events: list[dict[str, Any]] = []
    previous_positions = {
        robot.robot_id: robot.position for robot in environment.robots
    }
    steps = 0
    terminated = False

    while not terminated and steps < max_steps:
        actions = action_provider(environment)
        proposed_positions = {
            robot.robot_id: environment._get_next_position(
                robot, actions[robot.robot_id]
            )
            for robot in environment.robots
        }
        collision_events.extend(
            {
                "step": steps + 1,
                **event,
            }
            for event in infer_robot_robot_collisions(
                previous_positions, proposed_positions
            )
        )
        _, _, terminated, _ = environment.step(actions)
        steps += 1
        task_snapshots.append(_task_snapshot(environment))
        previous_positions = {
            robot.robot_id: robot.position for robot in environment.robots
        }

    return {
        "algorithm": algorithm,
        "fleet_size": fleet_size,
        "episode": episode_number,
        "task_snapshots": task_snapshots,
        "collision_events": collision_events,
        "safety_overrides": environment.safety_overrides,
        "steps": steps,
    }


def _rule_based_provider(environment: WarehouseEnvironment) -> dict[int, int]:
    return RuleBasedController(environment).get_actions()


def generate_rule_based(
    fleet_size: int,
    episodes: int,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    return [
        _record_episode(
            "Rule-Based",
            fleet_size,
            episode,
            max_steps,
            _rule_based_provider,
        )
        for episode in range(1, episodes + 1)
    ]


def generate_traces(
    algorithm: str,
    fleet_size: int,
    episodes: int,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate traces using a merged algorithm implementation.

    QMIX and IQL integrations are intentionally lazy imports. This keeps the
    Week 5 branch runnable with the existing baseline and allows the same
    command to work after the team merges the MARL branches.
    """

    if algorithm.lower() in {"rule-based", "rule_based", "rule"}:
        return generate_rule_based(fleet_size, episodes, max_steps, seed)

    raise RuntimeError(
        f"{algorithm} trace generation requires the merged MARL policy adapter. "
        "Run the Rule-Based generator now, or add the merged QMIX/IQL "
        "action provider in this module after integration."
    )


def write_traces(
    traces: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"episodes": traces}, indent=2),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--fleet-size", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
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
    )
    write_traces(traces, args.output)
    print(f"Wrote {len(traces)} episode traces to {args.output}")


if __name__ == "__main__":
    main()
