"""Run reproducible greedy QMIX episodes and save per-step diagnostics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from src.marl.qmix_trainer import QMIXTrainer
from src.warehouse.task_generation import generate_task_assignments


def _add_seeded_tasks(environment, seed: int) -> list[dict[str, Any]]:
    """Generate one held-out task set from an independent episode seed."""
    task_set = generate_task_assignments(environment, random.Random(seed))
    for task in task_set:
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
        for task in task_set
    ]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    episodes: int = 20,
    max_steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a five-robot checkpoint and retain policy/safety trajectories.

    Each episode gets a seeded, reachable task layout. Use a seed disjoint
    from training for held-out generalization checks.
    """
    torch.set_num_threads(1)
    trainer = QMIXTrainer(num_agents=5, episodes=1, max_steps=max_steps, seed=seed)
    trainer.load_checkpoint(str(checkpoint_path))
    trainer.learner.agent_network.eval()
    environment = trainer.env.adapter.environment
    environment.logger.setLevel("ERROR")

    episode_rows = []
    for episode_number in range(1, episodes + 1):
        observation, _ = trainer.env.reset()
        task_set = _add_seeded_tasks(environment, seed + episode_number)
        observation = trainer.env._encode_observations()
        trajectory = []
        episode_reward = 0.0
        completion_steps: dict[int, int] = {}

        for step in range(1, max_steps + 1):
            before = [robot.position for robot in environment.robots]
            observation_before = observation.copy()
            with torch.no_grad():
                q_values = trainer.learner.agent_network(
                    torch.as_tensor(observation, dtype=torch.float32)
                )
                action_masks = torch.as_tensor(
                    environment.get_action_masks(), dtype=torch.bool
                )
                actions = trainer.learner.select_actions(
                    torch.as_tensor(observation, dtype=torch.float32),
                    epsilon=0.0,
                    action_masks=action_masks,
                )

            observation, reward, terminated, truncated, info = trainer.env.step(actions.tolist())
            episode_reward += reward
            for task in environment.tasks:
                if task.completed and task.task_id not in completion_steps:
                    completion_steps[task.task_id] = step

            trajectory.append({
                "step": step,
                "observation_before": np_to_list(observation_before),
                "observation_after": np_to_list(observation),
                "q_values": np_to_list(q_values),
                "actions": actions.tolist(),
                "robot_states": [
                    {
                        "robot_id": robot.robot_id,
                        "position_before": list(before[robot.robot_id]),
                        "position": list(robot.position),
                        "carrying_item": robot.carrying_item,
                        "task_id": (
                            None if environment._get_robot_task(robot.robot_id) is None
                            else environment._get_robot_task(robot.robot_id).task_id
                        ),
                        "blocked_reason": robot.blocked_reason,
                    }
                    for robot in environment.robots
                ],
                "agent_rewards": {str(key): float(value) for key, value in info["agent_rewards"].items()},
                "safety_events": [event.to_dict() for event in environment.get_safety_events()],
            })
            if terminated or truncated:
                break

        completed = sum(task.completed for task in environment.tasks)
        episode_rows.append({
            "episode": episode_number,
            "tasks": task_set,
            "steps": len(trajectory),
            "completed_tasks": completed,
            "total_tasks": len(environment.tasks),
            "completion_rate": completed / len(environment.tasks),
            "completion_steps": completion_steps,
            "mean_team_reward_per_step": episode_reward / max(1, len(trajectory)),
            "safety_overrides": environment.safety_overrides,
            "trajectory": trajectory,
        })

    total_tasks = sum(row["total_tasks"] for row in episode_rows)
    completed_tasks = sum(row["completed_tasks"] for row in episode_rows)
    return {
        "algorithm": "QMIX",
        "checkpoint": str(checkpoint_path),
        "fleet_size": 5,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "checkpoint_loaded": True,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "completion_rate": completed_tasks / max(1, total_tasks),
        "mean_safety_overrides": sum(row["safety_overrides"] for row in episode_rows) / max(1, episodes),
        "episode_results": episode_rows,
    }


def np_to_list(value):
    """Convert tensors/arrays to JSON-compatible Python values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory or state file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")

    results = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print({key: value for key, value in results.items() if key != "episode_results"})
    print(f"Wrote per-step diagnostics to {args.output}")


if __name__ == "__main__":
    main()
