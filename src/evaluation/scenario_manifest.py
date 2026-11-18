"""Create and validate evaluation task queues independently of policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ai.environment import WarehouseEnvironment
from src.warehouse.task_generation import generate_task_assignments


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_evaluation_manifest(
    *, fleet_sizes: list[int], episodes: int, max_steps: int, seed: int
) -> dict[str, Any]:
    """Materialize immutable, explicit task queues for later held-out runs."""
    if not fleet_sizes or any(size not in (5, 10, 20) for size in fleet_sizes):
        raise ValueError("fleet_sizes must be a non-empty selection of 5, 10, and 20")
    if len(set(fleet_sizes)) != len(fleet_sizes):
        raise ValueError("fleet_sizes must be unique")
    if episodes != 20:
        raise ValueError("The specification requires exactly 20 scenarios per fleet size")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    root = Path(__file__).resolve().parents[2]
    config_path = root / "config" / "environment.json"
    scenario_sets: dict[str, list[dict[str, Any]]] = {}
    for size in fleet_sizes:
        environment = WarehouseEnvironment(num_robots=size, max_steps=max_steps)
        environment.logger.setLevel(logging.ERROR)
        environment.reset()
        layout = environment.get_active_layout()
        rows = []
        for episode in range(1, episodes + 1):
            task_seed = seed + episode - 1
            tasks = generate_task_assignments(environment, random.Random(task_seed))
            rows.append({
                "episode": episode,
                "task_seed": task_seed,
                "layout_id": layout.layout_id,
                "initial_tasks": [
                    {
                        "task_id": task.task_id,
                        "assigned_robot": task.assigned_robot,
                        "pickup_position": list(task.pickup_position),
                        "delivery_position": list(task.delivery_position),
                    }
                    for task in tasks
                ],
            })
        scenario_sets[str(size)] = rows

    body: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "held_out_evaluation; commit this file before training",
        "episodes_per_fleet": episodes,
        "max_steps": max_steps,
        "evaluation_seed": seed,
        "task_seed_rule": "evaluation_seed + episode_index - 1",
        "fleet_sizes": fleet_sizes,
        "environment_config_sha256": {"config/environment.json": _sha256(config_path)},
        "source_sha256": {
            name: _sha256(root / name)
            for name in (
                "src/evaluation/scenario_manifest.py",
                "src/evaluation/trace_generator.py",
                "src/warehouse/task_generation.py",
                "src/warehouse/layouts.py",
            )
        },
        "runtime": {"python": sys.version},
        "scenarios": scenario_sets,
    }
    return {"manifest_id": _canonical_hash(body), **body}


def load_evaluation_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a task manifest, including its tamper-evident ID."""
    document = json.loads(path.read_text(encoding="utf-8"))
    manifest_id = document.pop("manifest_id", None)
    if not isinstance(manifest_id, str) or manifest_id != _canonical_hash(document):
        raise ValueError("Evaluation manifest ID is missing or does not match its contents")
    if document.get("episodes_per_fleet") != 20:
        raise ValueError("Evaluation manifest must contain 20 episodes per fleet size")
    if not isinstance(document.get("scenarios"), dict):
        raise ValueError("Evaluation manifest has no scenario mapping")
    return {"manifest_id": manifest_id, **document}


def write_evaluation_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-sizes", nargs="+", type=int, choices=(5, 10, 20), default=[5, 10, 20])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_evaluation_manifest(
        fleet_sizes=args.fleet_sizes,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    write_evaluation_manifest(manifest, args.output)
    print(f"Wrote evaluation manifest {manifest['manifest_id']} to {args.output}")
    print("Commit this manifest before starting any training run.")


if __name__ == "__main__":
    main()
