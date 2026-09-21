"""
Infrastructure Smoke Test for MLflow & Checkpoint Management System.

==============================================================================
IMPORTANT NOTE / DISCLAIMER:
This script performs a pure INFRASTRUCTURE SMOKE TEST.
All logged metrics below are SYNTHETIC INFRASTRUCTURE VERIFICATION DATA.
They DO NOT represent real QMIX / IQL / EPyMARL training results.
==============================================================================
"""

import sys
import os
import torch
import torch.nn as nn

from python_infra.config import ExperimentConfig, TrainingMetrics
from python_infra.experiment_tracker import ExperimentTracker
from python_infra.checkpoint_manager import CheckpointManager


class DummyAgentNetwork(nn.Module):
    """Simple PyTorch module used ONLY to test model state checkpointing in the smoke test."""
    def __init__(self, obs_dim: int = 50, action_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def run_infrastructure_smoke_test():
    print("=" * 80)
    print("      AUTONOMOUS WAREHOUSE MARL — INFRASTRUCTURE SMOKE TEST")
    print("  (Validating MLflow Experiment Tracking & Algorithm-Agnostic Checkpoints)")
    print("=" * 80)
    print("[DISCLAIMER] All metrics in this run are SYNTHETIC DATA for infrastructure validation.")
    print("             No real QMIX or IQL training is being performed.")
    print("-" * 80)

    # 1. Initialize Configuration
    config = ExperimentConfig(
        project_name="Autonomous-Warehouse-MARL",
        experiment_name="infrastructure_smoke_tests",
        algorithm="QMIX",
        number_of_agents=10,
        environment_name="Warehouse-20x20-v1",
        environment_version="1.0.0",
        observation_dim=50,      # Preserving project terminology: 50-D observation vector
        action_space="Discrete(4)", # Preserving project terminology: Discrete(4) action space
        action_dim=4,
        random_seed=42,
        learning_rate=0.0005,
        gamma=0.99,
        batch_size=32,
        episode_count=5,
        max_episode_steps=50,
        experiment_type="smoke_test"
    )

    # 2. Initialize Experiment Tracker
    tracker = ExperimentTracker(experiment_name=config.experiment_name)
    run = tracker.start_run(
        run_name="smoke_test_run",
        tags={
            "description": "Validation of MLflow tracking and checkpoint restoration infrastructure",
            "is_smoke_test": "true"
        }
    )

    run_id = tracker.run_id
    print(f"[MLflow] Started MLflow Run ID: {run_id}")
    print(f"[MLflow] Tracking URI: {tracker.tracking_uri}")

    # 3. Log Hyperparameters & Config
    tracker.log_config(config)
    print(f"[MLflow] Logged experiment config parameters & tags successfully.")

    # 4. Initialize Checkpoint Manager
    checkpoint_mgr = CheckpointManager(
        base_dir="artifacts/checkpoints",
        algorithm=config.algorithm,
        run_id=run_id
    )
    print(f"[CheckpointManager] Initialized checkpoint directory at: {checkpoint_mgr.run_dir}")

    # Instantiate dummy PyTorch model & optimizer for state verification
    net = DummyAgentNetwork(obs_dim=config.observation_dim, action_dim=config.action_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    print("\n--- Simulating 5 Steps of Synthetic Metric Logging & Checkpoint Creation ---")

    for step in range(1, 6):
        # Generate synthetic test metrics clearly labelled as infrastructure test data
        synthetic_metrics = TrainingMetrics(
            episode_reward=-10.0 + step * 2.5,
            task_completion_rate=0.2 * step,
            completed_tasks=step * 3,
            collision_or_conflict_count=max(0, 5 - step),
            average_delivery_time=15.0 - step * 1.0,
            throughput=20.0 + step * 5.0,
            idle_time=5.0 - step * 0.8,
            episode_length=40 + step * 2,
            extra_metrics={
                "smoke_test_synthetic_loss": 1.0 / step,
                "smoke_test_data_flag": 1.0
            }
        )

        tracker.log_metrics(synthetic_metrics, step=step)
        print(f"  Step {step}/5 logged to MLflow | Synthetic Reward: {synthetic_metrics.episode_reward:.2f} | Task Completion: {synthetic_metrics.task_completion_rate * 100:.0f}%")

        # Save checkpoint payload at step 5
        if step == 5:
            checkpoint_payload = {
                "step": step,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "algorithm": config.algorithm,
                "config": config.to_dict(),
                "random_seed": config.random_seed,
                "metadata": {
                    "smoke_test_version": "1.0",
                    "note": "Synthetic checkpoint for infrastructure validation"
                }
            }

            # Save regular step checkpoint and best checkpoint
            step_path = checkpoint_mgr.save_checkpoint(
                checkpoint_data=checkpoint_payload,
                step=step,
                is_best=True,
                best_metric_val=synthetic_metrics.task_completion_rate,
                tracker=tracker
            )
            print(f"\n[CheckpointManager] Saved step {step} & best checkpoint to: {step_path}")

    # 5. Verify Checkpoint Loading
    print("\n--- Verifying Checkpoint Restoration ---")
    loaded = checkpoint_mgr.load_checkpoint("best")
    loaded_data = loaded["checkpoint_data"]
    loaded_meta = loaded["metadata"]

    print(f"[Verification] Successfully loaded checkpoint from: {loaded['checkpoint_dir']}")
    print(f"[Verification] Metadata step: {loaded_meta.get('step')}")
    print(f"[Verification] Observation vector: {loaded_meta.get('observation_vector')}")
    print(f"[Verification] Action space: {loaded_meta.get('action_space')}")

    # Load weights into fresh network instance to prove state restoration works
    new_net = DummyAgentNetwork(obs_dim=config.observation_dim, action_dim=config.action_dim)
    new_net.load_state_dict(loaded_data["model_state_dict"])
    print("[Verification] PyTorch state_dict restored successfully into new_net instance!")

    # 6. Complete MLflow Run
    tracker.end_run(status="FINISHED")
    print(f"\n[MLflow] Run {run_id} completed with status FINISHED.")
    print("=" * 80)
    print("  INFRASTRUCTURE SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 80)
    print(f"MLflow Run Data: file:///{os.path.abspath('mlruns').replace('\\', '/')}")
    print(f"Checkpoints:    file:///{os.path.abspath(checkpoint_mgr.run_dir).replace('\\', '/')}")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = run_infrastructure_smoke_test()
    sys.exit(0 if success else 1)
