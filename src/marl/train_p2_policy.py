"""Reproducible command-line training entry point for P2 policy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.marl.iql_trainer import IQLTrainer
from src.marl.qmix_trainer import QMIXTrainer


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
) -> dict:
    """Train one fixed configuration and save periodic inference snapshots."""
    if resume_from is not None and not resume_from.is_file():
        raise FileNotFoundError(f"Resume checkpoint file does not exist: {resume_from}")
    output.mkdir(parents=True, exist_ok=False)
    overrides = {} if collision_penalty is None else {"collision_penalty": collision_penalty}
    snapshots = output / "snapshots"

    def save_snapshot(trainer, metrics):
        # IQLTrainer does not support resume so episode_offset is always 0
        episode_offset = getattr(trainer, "episode_offset", 0)
        total_episodes = episode_offset + trainer.episodes
        _save_checkpoint_snapshot(
            algorithm, trainer, metrics, snapshots, checkpoint_interval, total_episodes
        )

    if algorithm == "QMIX":
        trainer = QMIXTrainer(
            num_agents=fleet_size, episodes=episodes, max_steps=max_steps,
            seed=seed, reward_overrides=overrides,
            learning_rate=learning_rate,
            target_update_interval=target_update_interval,
            episode_callback=save_snapshot,
        )
        if resume_from is not None:
            trainer.load_checkpoint(str(resume_from))
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
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.suffix in {".pt", ".json"}
        and path.name not in {"training_history.json", "training_manifest.json"}
    )
    summary["sha256"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    (output / "training_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train_policy(
        args.algorithm, args.fleet_size, args.episodes, args.max_steps,
        args.seed, args.output, args.collision_penalty,
        args.learning_rate, args.target_update_interval, args.checkpoint_interval,
        args.resume_from,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
