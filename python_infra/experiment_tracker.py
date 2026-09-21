import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
from typing import Dict, Any, Optional, Union
from python_infra.config import ExperimentConfig, TrainingMetrics



class ExperimentTracker:
    """
    MLflow Experiment Tracking Wrapper.
    Provides clean, algorithm-agnostic experiment logging for QMIX, IQL, and baseline algorithms.
    """
    def __init__(self, experiment_name: str = "warehouse_marl_coordination", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name

        # Tracking URI resolution order:
        #   1. Explicit argument
        #   2. MLFLOW_TRACKING_URI env var
        #   3. Local SQLite database (student-project default)
        self.tracking_uri = (
            tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
            or f"sqlite:///{os.path.abspath('mlflow.db')}"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self.active_run: Optional[mlflow.ActiveRun] = None


    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> mlflow.ActiveRun:
        """Starts a new MLflow run and applies mandatory project tags."""
        self.active_run = mlflow.start_run(run_name=run_name)
        
        default_tags = {
            "project": "Autonomous-Warehouse-MARL",
            "observation_vector": "50-D",
            "action_space": "Discrete(4)",
            "environment_name": "Warehouse-20x20-v1",
            "environment_version": "1.0.0"
        }
        
        if tags:
            default_tags.update({k: str(v) for k, v in tags.items()})
            
        mlflow.set_tags(default_tags)
        return self.active_run

    def end_run(self, status: str = "FINISHED") -> None:
        """Ends current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run(status=status)
        self.active_run = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)

    def log_config(self, config: Union[ExperimentConfig, Dict[str, Any]]) -> None:
        """Logs experiment hyperparameters and applies specific metadata tags to MLflow."""
        if isinstance(config, ExperimentConfig):
            params_dict = config.to_dict()
            # Set metadata tags from config
            tags = {
                "algorithm": config.algorithm,
                "fleet_size": str(config.number_of_agents),
                "environment_version": config.environment_version,
                "experiment_type": config.experiment_type,
                "project": config.project_name
            }
            mlflow.set_tags(tags)
        else:
            params_dict = config

        # Log parameters to MLflow
        self.log_parameters(params_dict)

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to current MLflow run."""
        if not mlflow.active_run():
            raise RuntimeError("Cannot log parameters without an active MLflow run. Call start_run() first.")
        
        # Convert non-scalar parameter values to strings for MLflow compatibility
        formatted_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                formatted_params[k] = v
            else:
                formatted_params[k] = str(v)
                
        mlflow.log_params(formatted_params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Logs a single metric scalar value at a given step."""
        if not mlflow.active_run():
            raise RuntimeError("Cannot log metric without an active MLflow run. Call start_run() first.")
        mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: Union[TrainingMetrics, Dict[str, float]], step: Optional[int] = None) -> None:
        """Logs a dictionary or dataclass of metrics at a given step."""
        if not mlflow.active_run():
            raise RuntimeError("Cannot log metrics without an active MLflow run. Call start_run() first.")
        
        metrics_dict = metrics.to_dict() if isinstance(metrics, TrainingMetrics) else metrics
        formatted_metrics = {k: float(v) for k, v in metrics_dict.items()}
        mlflow.log_metrics(formatted_metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs a file or directory as an MLflow artifact."""
        if not mlflow.active_run():
            raise RuntimeError("Cannot log artifact without an active MLflow run. Call start_run() first.")
        
        if os.path.isdir(local_path):
            mlflow.log_artifacts(local_path, artifact_path=artifact_path)
        elif os.path.isfile(local_path):
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        else:
            raise FileNotFoundError(f"Artifact path not found: {local_path}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Logs a dictionary directly as a JSON artifact."""
        if not mlflow.active_run():
            raise RuntimeError("Cannot log dict artifact without an active MLflow run. Call start_run() first.")
        mlflow.log_dict(dictionary, artifact_file)

    @property
    def run_id(self) -> Optional[str]:
        """Returns active MLflow run ID."""
        run = mlflow.active_run()
        return run.info.run_id if run else None
