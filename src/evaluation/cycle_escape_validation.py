"""Run a paired, validation-only test of an inference-time cycle escape at N=20."""
from __future__ import annotations

import argparse
from collections import deque
from hashlib import file_digest
import json
from pathlib import Path

from src.evaluation.metrics import evaluate_episodes, paired_algorithm_comparison
from src.evaluation.select_checkpoint import load_episode_records_from_traces
from src.evaluation.trace_generator import _iql_provider, _qmix_provider, _record_episode


def alternating(positions):
    return (
        len(positions) >= 4
        and positions[0] != positions[1]
        and all(cell == positions[index % 2] for index, cell in enumerate(positions))
    )


class CycleEscape:
    def __init__(self, policy, *, avoid_contended=False):
        self.policy = policy
        self.avoid_contended = avoid_contended
        self.replacements = 0
        self.triggers = 0

    def __call__(self, environment):
        actions = self.policy(environment)
        if environment.episode_manager.current_step == 0:
            self.history = {}
        occupied = {robot.position for robot in environment.robots}
        requested = {
            robot.robot_id: environment.get_target_cell(robot, actions[robot.robot_id])
            for robot in environment.robots
        }
        masks = environment.get_action_masks()
        for robot in environment.robots:
            robot_id = robot.robot_id
            task = environment._get_robot_task(robot_id)
            phase = (task.task_id if task else None, robot.carrying_item)
            history = self.history.setdefault((robot_id, phase), deque(maxlen=4))
            history.append(robot.position)
            proposed = requested[robot_id]
            if task is None or not alternating(history) or proposed != history[-2]:
                continue
            self.triggers += 1
            target = environment._task_target(robot, task)
            contended = {cell for other, cell in requested.items() if other != robot_id}
            alternatives = []
            for move, allowed in enumerate(masks[robot_id]):
                if not allowed:
                    continue
                cell = environment.get_target_cell(robot, move)
                if cell != proposed and cell not in occupied and (
                    not self.avoid_contended or cell not in contended
                ):
                    alternatives.append((environment._path_distance(cell, target), move))
            if alternatives:
                actions[robot_id] = min(alternatives)[1]
                requested[robot_id] = environment.get_target_cell(robot, actions[robot_id])
                self.replacements += 1
        return actions


def trace_diagnostics(traces):
    return {
        "timeouts": sum(
            trace["steps"] == trace["max_steps"]
            and any(not task["completed"] for task in trace["task_snapshots"][-1])
            for trace in traces
        ),
        "assignments_changed": sum(
            first["assigned_robot"] != last["assigned_robot"]
            for trace in traces
            for first, last in zip(trace["task_snapshots"][0], trace["task_snapshots"][-1], strict=True)
        ),
        "robot_episodes": sum(trace["fleet_size"] for trace in traces),
        "alternating_final_20": sum(
            len(trace["action_trace"]) >= 20 and alternating(
                [step["positions"][robot] for step in trace["action_trace"][-20:]]
            )
            for trace in traces for robot in trace["action_trace"][0]["positions"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=251720)
    parser.add_argument("--compare-contention", action="store_true")
    args = parser.parse_args()
    seed = args.seed
    episodes = 20
    root = Path("results/project25_validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for algorithm, build_policy in (("QMIX", _qmix_provider), ("IQL", _iql_provider)):
        selection = json.loads((root / f"{algorithm.lower()}_n20.json").read_text(encoding="utf-8"))
        for name, digest in selection["results"][0]["checkpoint_sha256"].items():
            with Path(name).open("rb") as file:
                if file_digest(file, "sha256").hexdigest() != digest:
                    raise ValueError(f"Checkpoint checksum mismatch: {name}")
        checkpoint = selection["selected_checkpoint"]
        policy = build_policy(checkpoint, 20)
        modified = CycleEscape(policy)
        variants = [("baseline", policy), ("cycle_escape", modified)]
        if args.compare_contention:
            variants.append(("contention_aware", CycleEscape(policy, avoid_contended=True)))
        by_variant = {}
        paired_records = []
        for variant, provider in variants:
            traces = [
                _record_episode(variant, 20, number, 100, seed + number - 1, provider)
                for number in range(1, episodes + 1)
            ]
            queues = [row["initial_tasks"] for row in traces]
            if variant == "baseline":
                baseline_queues = queues
            elif queues != baseline_queues:
                raise RuntimeError("Validation scenarios differ across variants")
            records = load_episode_records_from_traces(traces)
            paired_records.extend(records)
            by_variant[variant] = {
                "metrics": evaluate_episodes(records)[0],
                "diagnostics": trace_diagnostics(traces),
                "deliveries_by_episode": [
                    sum(task["completed"] for task in row["task_snapshots"][-1]) for row in traces
                ],
            }
        contention_comparison = (
            paired_algorithm_comparison(paired_records, "contention_aware", "cycle_escape", 20,
                                        metric="completion_rate", seed=seed)
            if args.compare_contention else {}
        )
        results.append({
            "algorithm": algorithm,
            "checkpoint": checkpoint,
            "checkpoint_sha256": selection["results"][0]["checkpoint_sha256"],
            "variants": by_variant,
            "replacements": modified.replacements,
            "triggers": modified.triggers,
            "paired_completion": paired_algorithm_comparison(
                paired_records, "cycle_escape", "baseline", 20, metric="completion_rate", seed=seed
            ),
            **({"contention_comparison": contention_comparison} if args.compare_contention else {}),
        })
        print(f"Completed {algorithm} paired validation", flush=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump({"seed": seed, "episodes_per_variant": episodes, "max_steps": 100, "results": results},
                  file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
