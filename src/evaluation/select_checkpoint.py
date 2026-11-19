"""Select a policy snapshot on a fixed, exposed validation set (never the holdout)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import bootstrap_confidence_interval, evaluate_episodes
from src.evaluation.trace_generator import generate_traces


def _checkpoint_hashes(algorithm: str, candidate: Path) -> dict[str, str]:
    if algorithm == "QMIX":
        paths = [candidate / "checkpoint_state.pt"] if candidate.is_dir() else [candidate]
    else:
        paths = sorted(candidate.glob("iql_robot_*.pt"))
        if (candidate / "iql_manifest.json").is_file():
            paths.append(candidate / "iql_manifest.json")
    files = [path for path in paths if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No {algorithm} checkpoint files found at {candidate}")
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _expand_candidates(algorithm: str, paths: list[Path]) -> list[Path]:
    """Accept one training snapshots directory or explicit checkpoint paths."""
    expanded = []
    for path in paths:
        if not path.is_dir():
            expanded.append(path)
        elif algorithm == "QMIX" and (path / "checkpoint_state.pt").is_file():
            expanded.append(path)
        elif algorithm == "IQL" and list(path.glob("iql_robot_*.pt")):
            expanded.append(path)
        elif algorithm == "QMIX":
            expanded.extend(sorted(path.glob("episode_*/checkpoint_state.pt")))
        else:
            expanded.extend(sorted(
                candidate.parent for candidate in path.glob("episode_*/iql_robot_*.pt")
            ))
    unique = list(dict.fromkeys(expanded))
    if not unique:
        raise FileNotFoundError("No checkpoint candidates were found in the supplied paths")
    return unique


def _episode_completion(trace: dict[str, Any]) -> float:
    final = trace["task_snapshots"][-1]
    return sum(bool(task["completed"]) for task in final) / len(final) if final else 0.0


def select_checkpoint(
    *, algorithm: str, fleet_size: int, candidates: list[Path],
    episodes: int = 20, max_steps: int = 100, seed: int = 20261120,
    output: Path | None = None,
) -> dict[str, Any]:
    """Evaluate candidate snapshots on identical seeded scenarios and rank them."""
    if algorithm not in {"QMIX", "IQL"}:
        raise ValueError("algorithm must be QMIX or IQL")
    if not candidates:
        raise ValueError("at least one candidate checkpoint is required")
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")

    candidates = _expand_candidates(algorithm, candidates)
    results = []
    for candidate in candidates:
        traces = generate_traces(
            algorithm, fleet_size, episodes, max_steps, seed, candidate
        )
        rows = evaluate_episodes(load_episode_records_from_traces(traces))
        if len(rows) != 1:
            raise RuntimeError("Expected one metric row per candidate")
        row = rows[0]
        completion = [_episode_completion(trace) for trace in traces]
        delivery_times = []
        for trace in traces:
            # Use the standard evaluator summary for aggregate delivery time;
            # completion CI remains episode-level and task-weighted per episode.
            per_record = evaluate_episodes(load_episode_records_from_traces([trace]))[0]
            if per_record["average_completion_time_steps"] is not None:
                delivery_times.append(per_record["average_completion_time_steps"])
        results.append({
            "candidate": str(candidate),
            "checkpoint_sha256": _checkpoint_hashes(algorithm, candidate),
            "completion_rate": row["completion_rate"],
            "completion_rate_95ci": bootstrap_confidence_interval(completion, seed=seed),
            "average_completion_time_steps": row["average_completion_time_steps"],
            "mean_episode_delivery_time_steps": (
                sum(delivery_times) / len(delivery_times) if delivery_times else None
            ),
            "safety_overrides_per_episode": row["safety_conflict_rate_per_episode"],
            "collision_intent_groups_per_episode": row["collision_intent_groups_per_episode"],
            "physical_collision_rate_per_episode": row["collision_rate_per_episode"],
        })

    results.sort(key=lambda row: (
        -row["completion_rate"],
        row["mean_episode_delivery_time_steps"]
        if row["mean_episode_delivery_time_steps"] is not None else float("inf"),
        row["collision_intent_groups_per_episode"],
        row["candidate"],
    ))
    report = {
        "schema_version": 1,
        "purpose": "exposed validation checkpoint selection; not final holdout",
        "algorithm": algorithm,
        "fleet_size": fleet_size,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "selection_order": [
            "highest completion rate", "lowest mean completed-task delivery time",
            "fewest blocked conflict groups", "candidate path lexical tie-break",
        ],
        "selected_checkpoint": results[0]["candidate"],
        "results": results,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def load_episode_records_from_traces(traces):
    """Adapt in-memory traces through the same schema normalization as files."""
    from dataclasses import fields
    from src.evaluation.metrics import EpisodeRecord

    allowed = {field.name for field in fields(EpisodeRecord)}
    return [EpisodeRecord(**{key: value for key, value in trace.items() if key in allowed})
            for trace in traces]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("QMIX", "IQL"), required=True)
    parser.add_argument("--fleet-size", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_checkpoint(
        algorithm=args.algorithm, fleet_size=args.fleet_size,
        candidates=args.candidate, episodes=args.episodes,
        max_steps=args.max_steps, seed=args.seed, output=args.output,
    )
    print(f"Selected {result['selected_checkpoint']}; report: {args.output}")


if __name__ == "__main__":
    main()
