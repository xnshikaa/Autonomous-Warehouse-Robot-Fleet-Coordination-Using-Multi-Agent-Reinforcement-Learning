"""Reproducible command-line training entry point for P2 policy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from src.marl.iql_trainer import IQLTrainer
from src.marl.qmix_trainer import QMIXTrainer

if TYPE_CHECKING:
    from python_infra.experiment_tracker import ExperimentTracker


def _save_checkpoint_snapshot(
    algorithm: str, trainer, metrics: dict, snapshots: Path,
    checkpoint_interval: int, episodes: int,
) -> Path | None:
    episode = int(metrics["episode"])
    if episode % checkpoint_interval and episode != episodes:
        return None
    snapshot_dir = snapshots / f"episode_{episode:05d}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    if algorithm == "QMIX":
        payload = trainer._get_checkpoint_payload(episode, float(metrics["epsilon"]))
        # These snapshots are for inference/validation; replay memory is
        # intentionally excluded to keep periodic artifacts compact.
        payload.pop("replay_buffer", None)
        torch.save(payload, snapshot_dir / "checkpoint_state.pt")
    elif algorithm == "IQL":
        trainer.save_agents(str(snapshot_dir))
    else:
        raise ValueError("algorithm must be QMIX or IQL")
    (snapshot_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return snapshot_dir


def train_policy(
    algorithm: str, fleet_size: int, episodes: int, max_steps: int,
    seed: int, output: Path, collision_penalty: float | None = None,
    learning_rate: float = 1e-4, target_update_interval: int = 200,
    checkpoint_interval: int = 100, resume_from: Path | None = None,
    *, tracker: ExperimentTracker | None = None, warm_start_from: Path | None = None,
    epsilon_start: float = 1.0,
) -> dict:
    """Train one fixed configuration and save periodic inference snapshots."""
    if resume_from is not None and not resume_from.is_file():
        raise FileNotFoundError(f"Resume checkpoint file does not exist: {resume_from}")
    if warm_start_from is not None:
        if algorithm != "QMIX" or resume_from is not None:
            raise ValueError("Warm start is supported for QMIX and cannot be combined with resume")
        if not warm_start_from.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint file does not exist: {warm_start_from}")
    if not 0.05 <= epsilon_start <= 1.0 or (algorithm != "QMIX" and epsilon_start != 1.0):
        raise ValueError("A custom epsilon start must be between 0.05 and 1.0 and requires QMIX")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    overrides = {} if collision_penalty is None else {"collision_penalty": collision_penalty}
    snapshots = output / "snapshots"

    def save_snapshot(trainer, metrics):
        # IQLTrainer does not support resume so episode_offset is always 0
        episode_offset = getattr(trainer, "episode_offset", 0)
        total_episodes = episode_offset + trainer.episodes
        _save_checkpoint_snapshot(
            algorithm, trainer, metrics, snapshots, checkpoint_interval, total_episodes
        )
        if tracker is not None:
            tracker.log_metrics(
                {key: metrics[key] for key in ("reward", "loss", "steps", "epsilon")},
                step=metrics["episode"],
            )

    if algorithm == "QMIX":
        trainer = QMIXTrainer(
            num_agents=fleet_size, episodes=episodes, max_steps=max_steps,
            seed=seed, reward_overrides=overrides,
            learning_rate=learning_rate,
            epsilon_start=epsilon_start,
            target_update_interval=target_update_interval,
            episode_callback=save_snapshot,
        )
        if resume_from is not None:
            trainer.load_checkpoint(str(resume_from))
        if warm_start_from is not None:
            trainer.load_checkpoint(str(warm_start_from), warm_start=True)
        initial_episode_offset = trainer.episode_offset
        history = trainer.train()
        checkpoint = output / "qmix.pt"
        torch.save(
            trainer._get_checkpoint_payload(initial_episode_offset + episodes, trainer.epsilon_end),
            checkpoint,
        )
        episodes_completed_total = initial_episode_offset + episodes
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    elif algorithm == "IQL":
        if resume_from is not None:
            raise ValueError("IQL resume is not supported by this training entry point")
        trainer = IQLTrainer(
            num_robots=fleet_size, episodes=episodes, max_steps=max_steps,
            seed=seed, include_static_map=True, include_path_costs=True,
            reward_overrides=overrides, learning_rate=learning_rate,
            target_update_frequency=target_update_interval,
            episode_callback=save_snapshot,
        )
        history = trainer.train()
        episodes_completed_total = episodes
        checkpoint = output / "agents"
        trainer.save_agents(str(checkpoint))
        config = json.loads((checkpoint / "iql_manifest.json").read_text())
    else:
        raise ValueError("algorithm must be QMIX or IQL")

    history_path = output / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "algorithm": algorithm,
        "fleet_size": fleet_size,
        "episodes": episodes,
        "episodes_completed_total": episodes_completed_total,
        "training_duration_seconds": perf_counter() - started,
        "max_steps": max_steps,
        "seed": seed,
        "learning_rate": learning_rate,
        "target_update_interval": target_update_interval,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_selection": "final; select among snapshots using validation only",
        "reward_overrides": overrides,
        "config": config,
        "checkpoint": checkpoint.name,
        "history": history_path.name,
    }
    if resume_from is not None:
        summary["resumed_from"] = str(resume_from)
        summary["resumed_from_sha256"] = hashlib.sha256(resume_from.read_bytes()).hexdigest()
    if warm_start_from is not None:
        summary["warm_started_from"] = warm_start_from.as_posix()
        summary["warm_started_from_sha256"] = hashlib.sha256(warm_start_from.read_bytes()).hexdigest()
        summary["epsilon_start"] = epsilon_start
    if tracker is not None:
        summary["mlflow_run_id"] = tracker.run_id
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.suffix in {".pt", ".json"}
        and path.name not in {"training_history.json", "training_manifest.json"}
    )
    summary["sha256"] = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    manifest_path = output / "training_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if tracker is not None:
        tracker.log_metric("training_duration_seconds", summary["training_duration_seconds"])
        tracker.log_artifact(str(manifest_path))
        tracker.log_artifact(str(history_path))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("QMIX", "IQL"), required=True)
    parser.add_argument("--fleet-size", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--collision-penalty", type=float)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--warm-start-from", type=Path, help="Initialize QMIX weights with fresh replay and optimizer")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from python_infra.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(experiment_name="project25_policy_training")
    tracker.start_run(
        run_name=args.output.name,
        tags={"algorithm": args.algorithm, "fleet_size": args.fleet_size, "observation_vector": "456-D"},
    )
    try:
        tracker.log_parameters(vars(args))
        result = train_policy(
            args.algorithm, args.fleet_size, args.episodes, args.max_steps,
            args.seed, args.output, args.collision_penalty,
            args.learning_rate, args.target_update_interval, args.checkpoint_interval,
            args.resume_from, tracker=tracker, warm_start_from=args.warm_start_from,
            epsilon_start=args.epsilon_start,
        )
    except BaseException:
        tracker.end_run(status="FAILED")
        raise
    tracker.end_run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
