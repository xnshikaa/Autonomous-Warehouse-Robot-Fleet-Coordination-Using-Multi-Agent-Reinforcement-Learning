"""Dedicated aggressive-action test for the independent corridor safety gate."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.ai.environment import WarehouseEnvironment


def _corridor_entry(layout: Any, occupied: set[tuple[int, int]]) -> tuple[tuple[int, int], int, tuple[int, int]]:
    """Find an unoccupied legal cell and action directly into a corridor."""
    for target in sorted(layout.human_corridors):
        for action, (dx, dy) in WarehouseEnvironment.ACTION_DELTAS.items():
            current = (target[0] - dx, target[1] - dy)
            if (
                current not in occupied
                and layout.is_within_bounds(current)
                and layout.is_walkable(current)
                and current not in layout.human_corridors
            ):
                return current, action, target
    raise RuntimeError("No unoccupied legal approach into a configured corridor")


def run_safety_gate(*, episodes: int = 20, fleet_size: int = 5, max_steps: int = 200) -> dict[str, Any]:
    """Run 20 adversarial corridor-entry attempts and fail on any intrusion."""
    if episodes != 20:
        raise ValueError("The specification safety gate requires exactly 20 episodes")
    if fleet_size <= 0 or max_steps <= 0:
        raise ValueError("fleet_size and max_steps must be positive")

    layout = WarehouseEnvironment(num_robots=fleet_size).get_active_layout()
    if not layout.human_corridors:
        raise RuntimeError("The safety gate requires at least one configured corridor cell")

    result: dict[str, Any] = {
        "episodes": episodes,
        "fleet_size": fleet_size,
        "max_steps": max_steps,
        "policy": "selected robot commands direct corridor entry; peers use legal actions",
        "layout_id": layout.layout_id,
        "attempted_corridor_entries": 0,
        "robots_with_blocked_entries": 0,
        "intrusions": 0,
        "safety_overrides": 0,
        "episode_results": [],
    }
    for episode in range(1, episodes + 1):
        environment = WarehouseEnvironment(
            num_robots=fleet_size, max_steps=max_steps, load_layout_tasks=False
        )
        environment.logger.setLevel(logging.ERROR)
        environment.reset()
        safe_task_cell = next(
            cell for cell in sorted(environment.get_active_layout().walkable_cells)
            if cell not in environment.get_active_layout().human_corridors
        )
        environment.add_task(999, safe_task_cell, safe_task_cell, assigned_robot=None)
        if not environment.corridor_enabled:
            raise RuntimeError("Human corridor safety is disabled")
        attacker_id = (episode - 1) % fleet_size
        occupied = {
            tuple(robot.position) for robot in environment.robots
            if robot.robot_id != attacker_id
        }
        current, attack_action, target = _corridor_entry(
            environment.get_active_layout(), occupied
        )
        environment.get_robot_state(attacker_id).position = current
        legal_actions = environment.get_action_masks()
        actions = {
            robot.robot_id: next(
                action for action, legal in enumerate(legal_actions[robot.robot_id]) if legal
            )
            for robot in environment.robots
        }
        actions[attacker_id] = attack_action
        environment.step(actions)
        event = next(
            (item for item in environment.get_safety_events()
             if item.robot_id == attacker_id and item.reason == "human_corridor"),
            None,
        )
        if event is None or not event.overridden or event.target_cell != target:
            raise AssertionError(f"Aggressive corridor action was not blocked: {event}")
        intrusions = sum(
            robot.position in environment.get_active_layout().human_corridors
            for robot in environment.robots
        )
        result["attempted_corridor_entries"] += 1
        result["robots_with_blocked_entries"] += 1
        result["intrusions"] += intrusions
        result["safety_overrides"] += environment.safety_overrides
        result["episode_results"].append({
            "episode": episode,
            "steps": 1,
            "attacker_robot_id": attacker_id,
            "robots_with_blocked_entries": 1,
            "safety_overrides": environment.safety_overrides,
            "intrusions": intrusions,
        })

    result["passed"] = result["intrusions"] == 0 and result["attempted_corridor_entries"] > 0
    if not result["passed"]:
        raise AssertionError(f"Corridor safety gate failed: {result}")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_safety_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Safety gate PASS: 20 episodes, {result['attempted_corridor_entries']} blocked entries")


if __name__ == "__main__":
    main()
