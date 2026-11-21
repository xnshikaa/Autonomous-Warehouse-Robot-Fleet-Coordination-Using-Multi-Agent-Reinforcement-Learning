"""Optuna Hyperparameter Optimization Study for QMIX Warehouse Fleet Coordination."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna

from src.marl.qmix_trainer import QMIXTrainer

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    num_agents: int = 5,
    episodes: int = 10,
    max_steps: int = 50,
    seed: int = 100,
) -> float:
    """Optuna objective function maximizing task completion while minimizing conflicts."""
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99])
    mixing_hidden_dim = trial.suggest_categorical("mixing_hidden_dim", [16, 32, 64])
    collision_penalty = trial.suggest_float("collision_penalty", 0.5, 5.0)
    target_update_interval = trial.suggest_categorical("target_update_interval", [10, 20, 50])

    reward_overrides = {"collision_penalty": -abs(collision_penalty)}

    trial_seed = seed + trial.number
    trainer = QMIXTrainer(
        num_agents=num_agents,
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=lr,
        gamma=gamma,
        target_update_interval=target_update_interval,
        seed=trial_seed,
        reward_overrides=reward_overrides,
        save_checkpoints=False,
    )

    history = trainer.train()

    final_episodes = history[-max(1, len(history) // 2):]
    avg_completion = float(np.mean([ep.get("task_completion_rate", 0.0) for ep in final_episodes]))
    avg_conflicts = float(np.mean([ep.get("collision_or_conflict_count", 0.0) for ep in final_episodes]))
    avg_reward = float(np.mean([ep.get("reward", 0.0) for ep in final_episodes]))

    trial.set_user_attr("task_completion_rate", avg_completion)
    trial.set_user_attr("collision_or_conflict_count", avg_conflicts)
    trial.set_user_attr("average_reward", avg_reward)

    score = avg_completion * 100.0 - avg_conflicts * 2.0
    return score


def run_optuna_study(
    num_agents: int = 5,
    n_trials: int = 10,
    episodes: int = 5,
    max_steps: int = 40,
    seed: int = 42,
    output_dir: str | Path = "artifacts/optuna",
) -> Dict[str, Any]:
    """Run Optuna HPO study and save SQLite db, Pareto analysis, and summary."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "qmix_study.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    study = optuna.create_study(
        study_name=f"qmix_n{num_agents}_study",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            num_agents=num_agents,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
        ),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "trial_number": t.number,
                "score": float(t.value) if t.value is not None else 0.0,
                "params": t.params,
                "completion_rate": t.user_attrs.get("task_completion_rate", 0.0),
                "collision_count": t.user_attrs.get("collision_or_conflict_count", 0.0),
                "average_reward": t.user_attrs.get("average_reward", 0.0),
            })

    # Pareto analysis: identify non-dominated trials (high completion, low collisions)
    pareto_trials = []
    for t1 in trials_data:
        dominated = False
        for t2 in trials_data:
            if (t2["completion_rate"] >= t1["completion_rate"] and
                t2["collision_count"] <= t1["collision_count"] and
                (t2["completion_rate"] > t1["completion_rate"] or t2["collision_count"] < t1["collision_count"])):
                dominated = True
                break
        if not dominated:
            pareto_trials.append(t1)

    summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial": {
            "trial_number": best_trial.number,
            "best_score": float(best_trial.value) if best_trial.value is not None else 0.0,
            "best_params": best_trial.params,
            "completion_rate": best_trial.user_attrs.get("task_completion_rate", 0.0),
            "collision_count": best_trial.user_attrs.get("collision_or_conflict_count", 0.0),
            "average_reward": best_trial.user_attrs.get("average_reward", 0.0),
        },
        "pareto_optimal_trials": pareto_trials,
        "trials": trials_data,
        "database": str(db_path),
    }

    summary_file = out_dir / "study_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Optuna study complete. Best score: {best_trial.value:.2f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QMIX Optuna HPO Study")
    parser.add_argument("--fleet-size", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", type=str, default="artifacts/optuna")
    args = parser.parse_args()

    run_optuna_study(
        num_agents=args.fleet_size,
        n_trials=args.trials,
        episodes=args.episodes,
        max_steps=args.steps,
        output_dir=args.output,
    )
