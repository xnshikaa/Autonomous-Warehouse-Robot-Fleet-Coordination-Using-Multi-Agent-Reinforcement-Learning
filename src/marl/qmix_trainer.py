import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from python_infra.checkpoint_manager import CheckpointManager
from python_infra.config import ExperimentConfig, TrainingMetrics
from python_infra.experiment_tracker import ExperimentTracker
from src.marl.gymnasium_env import WarehouseGymEnv
from src.marl.qmix_learner import QMIXLearner
from src.marl.replay_buffer import ReplayBuffer


class QMIXTrainer:
    """
    QMIX Training Manager for Autonomous Warehouse MARL.

    Integrates:
    - Custom PyTorch QMIX learner and networks
    - Gymnasium multi-agent warehouse environment
    - Experience Replay Buffer for off-policy mini-batch training
    - MLflow experiment tracking
    - Algorithm-agnostic Checkpoint Management
    """

    def __init__(
        self,
        num_agents: int = 5,
        observation_dim: int = 50,
        action_dim: int = 4,
        state_dim: int = 250,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_interval: int = 10,
        max_steps: int = 20,
        episodes: int = 5,
        batch_size: int = 32,
        replay_buffer_capacity: int = 5000,
        replay_warmup_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        seed: int = 42,
        use_mlflow: bool = False,
        experiment_name: str = "warehouse_marl_coordination",
        run_name: Optional[str] = None,
        save_checkpoints: bool = False,
        checkpoint_dir: str = "artifacts/checkpoints",
        checkpoint_interval: int = 10,
    ):
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.max_steps = max_steps
        self.episodes = episodes
        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_warmup_size = replay_warmup_size

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.seed = seed

        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

        self._set_seed(seed)

        self.env = WarehouseGymEnv(
            num_agents=num_agents,
            max_steps=max_steps,
        )

        self.learner = QMIXLearner(
            num_agents=num_agents,
            observation_dim=observation_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_capacity,
            num_agents=num_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
            seed=seed,
        )

        self.tracker: Optional[ExperimentTracker] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.run_id: Optional[str] = None

    @staticmethod
    def _set_seed(seed: int):
        """Set deterministic seeds."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _epsilon_for_episode(self, episode: int) -> float:
        """Linearly decay epsilon across training episodes."""
        if self.episodes <= 1:
            return self.epsilon_end
        progress = episode / (self.episodes - 1)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _build_global_state(self, observation: np.ndarray) -> np.ndarray:
        """Concatenate agent observations into global state vector."""
        state = observation.reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"Expected global state dimension {self.state_dim}, received {state.shape[0]}"
            )
        return state.astype(np.float32)

    def _get_checkpoint_payload(self, episode: int, epsilon: float) -> Dict[str, Any]:
        """Construct checkpoint payload compatible with QMIXController and CheckpointManager."""
        return {
            "agent_network": self.learner.agent_network.state_dict(),
            "mixer": self.learner.mixer.state_dict(),
            "target_agent_network": self.learner.target_agent_network.state_dict(),
            "target_mixer": self.learner.target_mixer.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "update_count": self.learner.update_count,
            "episode": episode,
            "epsilon": epsilon,
            "seed": self.seed,
            "config": {
                "num_agents": self.num_agents,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
            },
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def train(self) -> List[Dict[str, Any]]:
        """
        Execute QMIX training loop.

        Returns:
            List of episode metrics dictionaries.
        """
        history = []

        # 1. Initialize MLflow tracker if requested
        if self.use_mlflow:
            self.tracker = ExperimentTracker(experiment_name=self.experiment_name)
            run_title = self.run_name or f"qmix_{self.num_agents}agents_{self.episodes}ep"
            self.tracker.start_run(
                run_name=run_title,
                tags={
                    "project": "Autonomous-Warehouse-MARL",
                    "algorithm": "QMIX",
                    "fleet_size": str(self.num_agents),
                    "environment_version": "1.0.0",
                    "experiment_type": "actual_training",
                },
            )
            self.run_id = self.tracker.run_id

            # Log Hyperparameters
            config = ExperimentConfig(
                project_name="Autonomous-Warehouse-MARL",
                experiment_name=self.experiment_name,
                algorithm="QMIX",
                number_of_agents=self.num_agents,
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                batch_size=self.batch_size,
                episode_count=self.episodes,
                max_episode_steps=self.max_steps,
                random_seed=self.seed,
                epsilon_start=self.epsilon_start,
                epsilon_finish=self.epsilon_end,
                experiment_type="actual_training",
                extra_params={
                    "replay_buffer_capacity": self.replay_buffer_capacity,
                    "replay_warmup_size": self.replay_warmup_size,
                    "target_update_interval": self.target_update_interval,
                    "state_dim": self.state_dim,
                },
            )
            self.tracker.log_config(config)

        # 2. Initialize CheckpointManager if requested
        if self.save_checkpoints:
            run_identifier = self.run_id or self.run_name or f"qmix_run_seed_{self.seed}"
            self.checkpoint_manager = CheckpointManager(
                base_dir=self.checkpoint_dir,
                algorithm="QMIX",
                run_id=run_identifier,
            )

        best_reward = float("-inf")

        # 3. Main Episode Training Loop
        for episode in range(self.episodes):
            observation, _ = self.env.reset()

            # Ensure deterministic task allocation for cooperative evaluation
            if not self.env.adapter.environment.tasks:
                for r_id in range(self.num_agents):
                    self.env.adapter.environment.add_task(
                        task_id=r_id + 1,
                        pickup_position=((r_id + 1) * 2, 1),
                        delivery_position=((r_id + 1) * 2, 15),
                        assigned_robot=r_id,
                    )

            epsilon = self._epsilon_for_episode(episode)
            episode_reward = 0.0
            losses = []
            step_count = 0

            for step in range(self.max_steps):
                step_count += 1
                obs_tensor = torch.tensor(observation, dtype=torch.float32)
                actions = self.learner.select_actions(obs_tensor, epsilon=epsilon)

                next_observation, reward, terminated, truncated, _ = self.env.step(actions.tolist())

                global_state = self._build_global_state(observation)
                next_global_state = self._build_global_state(next_observation)
                done = terminated or truncated

                # Push to Experience Replay Buffer
                self.replay_buffer.push(
                    observations=observation,
                    global_state=global_state,
                    actions=actions.cpu().numpy(),
                    reward=reward,
                    next_observations=next_observation,
                    next_global_state=next_global_state,
                    terminated=done,
                )

                # Train on sampled mini-batch if replay buffer has sufficient data
                if (
                    self.replay_buffer.can_sample(self.batch_size)
                    and len(self.replay_buffer) >= self.replay_warmup_size
                ):
                    batch = self.replay_buffer.sample(self.batch_size)
                    update_res = self.learner.train_batch(
                        observations=batch["observations"],
                        actions=batch["actions"],
                        rewards=batch["rewards"],
                        next_observations=batch["next_observations"],
                        global_states=batch["global_states"],
                        next_global_states=batch["next_global_states"],
                        terminated=batch["terminated"],
                    )
                    losses.append(update_res["loss"])

                episode_reward += float(reward)
                observation = next_observation

                if done:
                    break

            # Metrics Computation
            env_inst = self.env.adapter.environment
            tasks = env_inst.tasks
            completed_tasks = sum(1 for t in tasks if t.completed)
            total_tasks = len(tasks)
            completion_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
            conflicts = env_inst.safety_overrides
            avg_loss = float(np.mean(losses)) if losses else 0.0
            throughput = (completed_tasks / step_count * 100.0) if step_count > 0 else 0.0

            ep_metrics = {
                "episode": episode + 1,
                "steps": step_count,
                "epsilon": float(epsilon),
                "reward": episode_reward,
                "completed_tasks": completed_tasks,
                "task_completion_rate": completion_rate,
                "collision_or_conflict_count": conflicts,
                "throughput": throughput,
                "loss": avg_loss,
            }
            history.append(ep_metrics)

            # Log to MLflow
            if self.tracker:
                metrics_obj = TrainingMetrics(
                    episode_reward=episode_reward,
                    task_completion_rate=completion_rate,
                    completed_tasks=completed_tasks,
                    collision_or_conflict_count=conflicts,
                    average_delivery_time=0.0,
                    throughput=throughput,
                    idle_time=0.0,
                    episode_length=step_count,
                    extra_metrics={"training_loss": avg_loss, "epsilon": float(epsilon)},
                )
                self.tracker.log_metrics(metrics_obj, step=episode + 1)

            # Save Checkpoint
            if self.save_checkpoints and self.checkpoint_manager:
                payload = self._get_checkpoint_payload(episode + 1, epsilon)
                is_best = episode_reward > best_reward
                if is_best:
                    best_reward = episode_reward

                # Periodic or final step save
                if (episode + 1) % self.checkpoint_interval == 0 or (episode + 1) == self.episodes:
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_data=payload,
                        step=episode + 1,
                        is_best=is_best,
                        best_metric_val=episode_reward,
                        tracker=self.tracker,
                    )
                elif is_best:
                    self.checkpoint_manager.save_best_checkpoint(
                        checkpoint_data=payload,
                        step=episode + 1,
                        current_metric_value=episode_reward,
                        mode="max",
                        tracker=self.tracker,
                    )

        # Complete MLflow run cleanly
        if self.tracker:
            self.tracker.end_run(status="FINISHED")

        return history

    def load_checkpoint(self, checkpoint_target: str | int) -> Dict[str, Any]:
        """
        Load training checkpoint state and restore QMIX learner components.
        """
        if not self.checkpoint_manager:
            self.checkpoint_manager = CheckpointManager(
                base_dir=self.checkpoint_dir,
                algorithm="QMIX",
                run_id=self.run_id or self.run_name or f"qmix_run_seed_{self.seed}",
            )

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_target)
        payload = checkpoint["checkpoint_data"]

        agent_state = payload.get("agent_network") or payload.get("model_state_dict")
        if agent_state:
            self.learner.agent_network.load_state_dict(agent_state)
        mixer_state = payload.get("mixer")
        if mixer_state:
            self.learner.mixer.load_state_dict(mixer_state)
        target_agent_state = payload.get("target_agent_network")
        if target_agent_state:
            self.learner.target_agent_network.load_state_dict(target_agent_state)
        target_mixer_state = payload.get("target_mixer")
        if target_mixer_state:
            self.learner.target_mixer.load_state_dict(target_mixer_state)
        optimizer_state = payload.get("optimizer")
        if optimizer_state:
            self.learner.optimizer.load_state_dict(optimizer_state)
        replay_state = payload.get("replay_buffer")
        if replay_state:
            self.replay_buffer.load_state_dict(replay_state)

        if "update_count" in payload:
            self.learner.update_count = payload["update_count"]

        return checkpoint