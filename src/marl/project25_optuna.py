"""Run the persistent Project 25 QMIX warm-start validation study."""

import argparse
import csv
import gzip
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
from filelock import FileLock

from src.evaluation.metrics import EpisodeRecord, evaluate_episodes
from src.evaluation.scenario_manifest import (
    _canonical_hash, create_evaluation_manifest, load_evaluation_manifest,
)
from src.evaluation.trace_generator import _qmix_provider, _record_episode
from src.marl.qmix_trainer import QMIXTrainer


NAME = "project25_qmix_n5_validation_v1"
INCUMBENT = Path("artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt")
RANGES = {
    "learning_rate": [0.00003, 0.001],
    "mixing_hidden_dim": [16, 32, 64],
    "gamma": [0.95, 0.999],
    "collision_penalty": [-20.0, -2.0],
    "delivery_reward": [5.0, 20.0],
}
TRAIN_SEED = 254005
VALIDATION_SEED = 254105
SAMPLER_SEED = 254205
FLEET = 5
HORIZON = 100
SCREEN_EPISODES = 6
TRAIN_EPISODES = 4
STATE = optuna.trial.TrialState


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def scenarios(path, seed, purpose):
    if path.exists():
        return load_evaluation_manifest(path)
    manifest = create_evaluation_manifest(
        fleet_sizes=[FLEET], episodes=20, max_steps=HORIZON, seed=seed,
    )
    manifest.pop("manifest_id")
    manifest["purpose"] = purpose
    manifest["manifest_id"] = _canonical_hash(manifest)
    write_json(path, manifest)
    return manifest


def evaluate(checkpoint, manifest, episodes, trace_path):
    provider = _qmix_provider(checkpoint, FLEET)
    traces = [
        _record_episode("qmix", FLEET, row["episode"], HORIZON, row["task_seed"], provider, row)
        for row in manifest["scenarios"][str(FLEET)][:episodes]
    ]
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        json.dump(traces, stream, separators=(",", ":"))
    record_fields = EpisodeRecord.__dataclass_fields__
    metrics = evaluate_episodes([
        EpisodeRecord(**{key: val for key, val in trace.items() if key in record_fields})
        for trace in traces
    ])[0]
    delivery = metrics["average_completion_time_steps"]
    conflicts = metrics["safety_conflict_events"]
    score = metrics["completed_tasks"] - 0.49 * conflicts / (1 + conflicts)
    score -= 0.49 * (HORIZON if delivery is None else delivery) / HORIZON
    return {**metrics, "score": score}


def run_trial(trial, out, manifest, agent_weights):
    started = time.perf_counter()
    trial_dir = out / "trials" / f"{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    params = {
        name: trial.suggest_categorical(name, bounds) if name == "mixing_hidden_dim"
        else trial.suggest_float(name, *bounds, log=name == "learning_rate")
        for name, bounds in RANGES.items()
    }
    trainer = QMIXTrainer(
        num_agents=FLEET, episodes=TRAIN_EPISODES, max_steps=HORIZON, seed=TRAIN_SEED,
        epsilon_start=0.20, epsilon_end=0.05,
        learning_rate=params["learning_rate"], gamma=params["gamma"],
        mixing_hidden_dim=params["mixing_hidden_dim"],
        reward_overrides={key: params[key] for key in ("collision_penalty", "delivery_reward")},
    )
    trainer.learner.agent_network.load_state_dict(agent_weights)
    trainer.learner.update_target_networks()
    trainer.run_id = f"{NAME}_trial_{trial.number}"
    history = trainer.train()
    if not trainer.learner.update_count or not np.isfinite([row["loss"] for row in history]).all():
        raise ValueError("Trial did not produce finite optimizer updates")
    checkpoint = trial_dir / "checkpoint_state.pt"
    payload = trainer._get_checkpoint_payload(TRAIN_EPISODES, 0.05)
    payload.pop("replay_buffer")  # Trial-level restart does not resume an individual optimizer.
    torch.save(payload, checkpoint)
    metrics = evaluate(checkpoint, manifest, SCREEN_EPISODES, trial_dir / "validation.json.gz")
    attrs = {
        **metrics, "training_seed": TRAIN_SEED, "sampler_seed": SAMPLER_SEED + trial.number,
        "validation_seed": manifest["evaluation_seed"], "manifest_id": manifest["manifest_id"],
        "training_mean_reward": float(np.mean([row["reward"] for row in history])),
        "training_mean_loss": float(np.mean([row["loss"] for row in history])),
        "optimizer_updates": trainer.learner.update_count,
        "duration_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint.relative_to(out).as_posix(), "checkpoint_sha256": digest(checkpoint),
    }
    write_json(trial_dir / "record.json", {"params": params, "metrics": attrs, "history": history})
    for name, value in attrs.items():
        trial.set_user_attr(name, value)
    return metrics["score"]


def export_study(study, out):
    rows = [
        {"trial": t.number, "status": t.state.name, "objective": t.value, **t.params, **t.user_attrs}
        for t in study.trials
    ]
    counts = Counter(row["status"] for row in rows)
    best = rows[study.best_trial.number]
    write_json(out / "study_summary.json", {"study": NAME, "counts": dict(counts), "best": best, "trials": rows})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "best_trial.md").write_text(
        f"# QMIX validation trial {best['trial']}\n\n"
        "| Hyperparameter | Value |\n|---|---:|\n" +
        "".join(f"| {key} | {best[key]} |\n" for key in RANGES) +
        f"\nCompleted {best['completed_tasks']}/{best['tasks']} assigned tasks. "
        f"Objective {best['objective']:.4f}.\n", encoding="utf-8",
    )
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.figure.savefig(out / "optimization_history.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    ax = optuna.visualization.matplotlib.plot_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SAMPLER_SEED),
    )
    ax.figure.savefig(out / "hyperparameter_importance.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/project25_optuna"))
    parser.add_argument("--trials", type=int, default=100, help="Target total of completed trials")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2 to estimate importance")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    logging.getLogger().setLevel(logging.ERROR)
    with FileLock(out / "study.lock", timeout=0):
        manifest = scenarios(out / "screening_manifest.json", VALIDATION_SEED, "Optuna screening validation")
        confirmation = scenarios(out / "confirmation_manifest.json", VALIDATION_SEED + 1000, "Confirmation validation")
        db = f"sqlite:///{(out / 'qmix_study.db').as_posix()}"
        study = optuna.create_study(
            study_name=NAME, storage=db, direction="maximize",
            pruner=optuna.pruners.NopPruner(), load_if_exists=True,
        )
        protocol = {
            "incumbent": INCUMBENT.as_posix(), "incumbent_sha256": digest(INCUMBENT),
            "screening_manifest": manifest["manifest_id"], "confirmation_manifest": confirmation["manifest_id"],
            "screening_episodes": SCREEN_EPISODES, "training_episodes": TRAIN_EPISODES,
            "training_seed": TRAIN_SEED, "sampler_seed_base": SAMPLER_SEED, "max_steps": HORIZON,
            "search": RANGES, "code_sha256": {
                path: digest(Path(path)) for path in (
                    __file__, "src/marl/qmix_trainer.py", "src/marl/qmix_controller.py",
                    "src/marl/qmix_learner.py", "config/reward.json",
                )
            },
            "confirmation_rule": "Compare screening winner and incumbent on 20 separate scenarios. "
                                 "Recommend further testing only with more tasks and zero physical collisions.",
        }
        if study.user_attrs.get("protocol", protocol) != protocol:
            raise ValueError("Study protocol changed. Use a separate output directory.")
        study.set_user_attr("protocol", protocol)
        for trial in study.get_trials(states=(STATE.RUNNING,)):
            study.tell(trial.number, state=STATE.FAIL)
        weights = torch.load(INCUMBENT, weights_only=False, map_location="cpu")["agent_network"]
        mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(NAME)
        with mlflow.start_run(run_name="qmix_screening_and_confirmation") as run:
            mlflow.log_params({**protocol, "target_completed_trials": args.trials})
            while (completed := len(study.get_trials(states=(STATE.COMPLETE,)))) < args.trials:
                number = len(study.get_trials(deepcopy=False))
                study.sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED + number, n_startup_trials=15)
                study.optimize(lambda trial: run_trial(trial, out, manifest, weights), n_trials=1)
                row = study.get_trials(deepcopy=False)[-1]
                mlflow.log_metrics({
                    key: val for key, val in row.user_attrs.items() if isinstance(val, (float, int))
                }, step=number)

            export_study(study, out)
            best = study.best_trial
            final_rows = {}
            for label, checkpoint in (
                ("incumbent", INCUMBENT), ("screening_winner", out / best.user_attrs["checkpoint"])
            ):
                final_rows[label] = evaluate(checkpoint, confirmation, 20, out / f"{label}_confirmation.json.gz")
            write_json(out / "confirmation.json", {
                "manifest_id": confirmation["manifest_id"], "trial": best.number,
                "mlflow_run_id": run.info.run_id, **final_rows,
            })
            for name in ("study_summary.json", "trials.csv", "best_trial.md", "screening_manifest.json",
                         "confirmation_manifest.json", "confirmation.json", "optimization_history.png",
                         "hyperparameter_importance.png", "incumbent_confirmation.json.gz",
                         "screening_winner_confirmation.json.gz"):
                mlflow.log_artifact(str(out / name))
            mlflow.log_artifact(str(out / best.user_attrs["checkpoint"]), "winner")


if __name__ == "__main__":
    main()
