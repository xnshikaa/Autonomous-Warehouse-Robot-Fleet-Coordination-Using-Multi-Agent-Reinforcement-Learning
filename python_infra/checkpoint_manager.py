"""
Algorithm-Agnostic Model & Training Checkpoint Manager for Autonomous Warehouse MARL.
"""

import os
import json
import shutil
import time
from typing import Dict, Any, Optional, Union, List
from python_infra.config import ExperimentConfig

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import pickle


class CheckpointManager:
    """
    Algorithm-agnostic checkpoint management system.
    Supports QMIX, IQL, baseline heuristic models, and future EPyMARL architectures.
    
    Directory layout:
        artifacts/
          checkpoints/
            {algorithm}/
              {run_id}/
                checkpoint_{step}/
                  state.pt (or state.pkl)
                  metadata.json
                latest/
                best/
    """
    def __init__(
        self,
        base_dir: str = "artifacts/checkpoints",
        algorithm: str = "QMIX",
        run_id: str = "default_run"
    ):
        self.base_dir = base_dir
        self.algorithm = algorithm
        self.run_id = run_id
        
        self.run_dir = os.path.join(self.base_dir, self.algorithm.lower(), self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.best_metric_value: Optional[float] = None

    def _save_payload(self, file_path: str, data: Dict[str, Any]) -> None:
        """Saves arbitrary Python payload (PyTorch state dicts, dicts, numpy arrays) algorithm-agnostically."""
        if HAS_TORCH:
            torch.save(data, file_path)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

    def _load_payload(self, file_path: str) -> Dict[str, Any]:
        """Loads arbitrary Python payload algorithm-agnostically."""
        if HAS_TORCH:
            # Load with weights_only=False or map_location if needed
            try:
                return torch.load(file_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(file_path, map_location="cpu")
        else:
            with open(file_path, "rb") as f:
                return pickle.load(f)

    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        step: int,
        is_best: bool = False,
        best_metric_val: Optional[float] = None,
        tracker: Optional[Any] = None
    ) -> str:
        """
        Saves a training checkpoint at a given step.
        
        Args:
            checkpoint_data: Dictionary containing model state_dicts, optimizer state, step, config, etc.
            step: Current training step or episode count.
            is_best: Whether this checkpoint is currently the best performing model.
            best_metric_val: Value of the tracked metric if is_best is True.
            tracker: Optional ExperimentTracker instance for MLflow artifact association.

        Returns:
            Path to the saved checkpoint directory.
        """
        step_dir = os.path.join(self.run_dir, f"checkpoint_{step}")
        os.makedirs(step_dir, exist_ok=True)

        state_file = os.path.join(step_dir, "checkpoint_state.pt" if HAS_TORCH else "checkpoint_state.pkl")
        self._save_payload(state_file, checkpoint_data)

        # Build clean metadata
        metadata = {
            "step": step,
            "algorithm": self.algorithm,
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "observation_vector": "50-D",
            "action_space": "Discrete(4)",
            "is_best": is_best,
            "best_metric_val": best_metric_val,
            "keys": list(checkpoint_data.keys())
        }

        meta_file = os.path.join(step_dir, "metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update latest directory
        latest_dir = os.path.join(self.run_dir, "latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(step_dir, latest_dir)

        # Update best directory if specified
        if is_best:
            best_dir = os.path.join(self.run_dir, "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(step_dir, best_dir)

        # Associate checkpoint with MLflow run if tracker is provided
        if tracker and hasattr(tracker, "log_artifact"):
            try:
                tracker.log_artifact(step_dir, artifact_path=f"checkpoints/checkpoint_{step}")
                if is_best:
                    tracker.log_artifact(step_dir, artifact_path="checkpoints/best")
            except Exception as e:
                print(f"[CheckpointManager] Note: Could not log artifact to MLflow: {e}")

        return step_dir

    def save_best_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        step: int,
        current_metric_value: float,
        mode: str = "max",
        tracker: Optional[Any] = None
    ) -> bool:
        """
        Saves model as best checkpoint if current metric improves upon best_metric_value.
        
        Args:
            mode: "max" (higher metric is better) or "min" (lower metric is better)
        """
        is_improved = False
        if self.best_metric_value is None:
            is_improved = True
        elif mode == "max" and current_metric_value > self.best_metric_value:
            is_improved = True
        elif mode == "min" and current_metric_value < self.best_metric_value:
            is_improved = True

        if is_improved:
            self.best_metric_value = current_metric_value
            self.save_checkpoint(
                checkpoint_data=checkpoint_data,
                step=step,
                is_best=True,
                best_metric_val=current_metric_value,
                tracker=tracker
            )

        return is_improved

    def load_checkpoint(self, checkpoint_target: Union[str, int]) -> Dict[str, Any]:
        """
        Loads checkpoint state and metadata.
        
        Args:
            checkpoint_target: Step number (e.g. 5000), "latest", "best", or absolute/relative directory path.
            
        Returns:
            Dictionary containing 'state' payload and 'metadata'.
        """
        if isinstance(checkpoint_target, int):
            target_dir = os.path.join(self.run_dir, f"checkpoint_{checkpoint_target}")
        elif checkpoint_target in ("latest", "best"):
            target_dir = os.path.join(self.run_dir, checkpoint_target)
        else:
            target_dir = checkpoint_target

        if not os.path.exists(target_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found: '{target_dir}'. "
                f"Ensure a checkpoint has been saved before loading."
            )

        state_file_pt = os.path.join(target_dir, "checkpoint_state.pt")
        state_file_pkl = os.path.join(target_dir, "checkpoint_state.pkl")
        
        if os.path.exists(state_file_pt):
            state = self._load_payload(state_file_pt)
        elif os.path.exists(state_file_pkl):
            state = self._load_payload(state_file_pkl)
        else:
            raise FileNotFoundError(f"No checkpoint state file (checkpoint_state.pt/pkl) found in '{target_dir}'.")

        metadata = {}
        meta_file = os.path.join(target_dir, "metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                metadata = json.load(f)

        return {
            "checkpoint_data": state,
            "metadata": metadata,
            "checkpoint_dir": target_dir
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lists metadata of all saved checkpoints for this run."""
        checkpoints = []
        if not os.path.exists(self.run_dir):
            return checkpoints

        for entry in sorted(os.listdir(self.run_dir)):
            full_path = os.path.join(self.run_dir, entry)
            if os.path.isdir(full_path):
                meta_path = os.path.join(full_path, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        checkpoints.append(json.load(f))
        return checkpoints

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Returns directory path of latest checkpoint, if it exists."""
        latest_dir = os.path.join(self.run_dir, "latest")
        return latest_dir if os.path.exists(latest_dir) else None
