"""
Unit tests for MLflow ExperimentTracker.
"""

import os
import pytest
import mlflow

from python_infra.config import ExperimentConfig, TrainingMetrics
from python_infra.experiment_tracker import ExperimentTracker


@pytest.fixture
def tmp_tracker(tmp_path):
    """Fixture providing an ExperimentTracker instance writing to a temp tracking URI."""
    tracking_uri = str(tmp_path / "test_mlruns")
    tracker = ExperimentTracker(experiment_name="test_experiment", tracking_uri=tracking_uri)
    yield tracker
    if mlflow.active_run():
        mlflow.end_run()


def test_tracker_initialization(tmp_tracker):
    assert tmp_tracker.experiment_name == "test_experiment"
    assert os.path.exists(tmp_tracker.tracking_uri)


def test_start_and_end_run(tmp_tracker):
    run = tmp_tracker.start_run(run_name="test_run", tags={"environment_version": "1.0.0"})
    assert run is not None
    assert tmp_tracker.run_id is not None
    assert mlflow.active_run() is not None

    tmp_tracker.end_run()
    assert mlflow.active_run() is None


def test_context_manager(tmp_tracker):
    with tmp_tracker as tracker:
        assert tracker.run_id is not None
        assert mlflow.active_run() is not None
    assert mlflow.active_run() is None


def test_parameter_logging(tmp_tracker):
    config = ExperimentConfig(
        algorithm="QMIX",
        number_of_agents=10,
        observation_dim=50,
        action_space="Discrete(4)",
        learning_rate=0.0005
    )
    with tmp_tracker as tracker:
        tracker.log_config(config)
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(tracker.run_id).data
        assert run_data.params["algorithm"] == "QMIX"
        assert run_data.params["number_of_agents"] == "10"
        assert run_data.params["observation_dim"] == "50"
        assert run_data.params["action_space"] == "Discrete(4)"
        assert run_data.tags["algorithm"] == "QMIX"
        assert run_data.tags["fleet_size"] == "10"


def test_metric_logging(tmp_tracker):
    metrics = TrainingMetrics(
        episode_reward=100.5,
        task_completion_rate=0.85,
        completed_tasks=42,
        collision_or_conflict_count=0,
        average_delivery_time=12.4,
        throughput=150.0,
        idle_time=3.2,
        episode_length=80
    )
    with tmp_tracker as tracker:
        tracker.log_metrics(metrics, step=1)
        client = mlflow.tracking.MlflowClient()
        run_metrics = client.get_run(tracker.run_id).data.metrics
        assert run_metrics["episode_reward"] == pytest.approx(100.5)
        assert run_metrics["task_completion_rate"] == pytest.approx(0.85)
        assert run_metrics["completed_tasks"] == pytest.approx(42.0)
        assert run_metrics["collision_or_conflict_count"] == pytest.approx(0.0)


def test_artifact_logging(tmp_tracker, tmp_path):
    test_file = tmp_path / "sample_artifact.txt"
    test_file.write_text("Infrastructure test content")

    with tmp_tracker as tracker:
        tracker.log_artifact(str(test_file), artifact_path="test_artifacts")
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(tracker.run_id, path="test_artifacts")
        assert len(artifacts) > 0
        assert artifacts[0].path == "test_artifacts/sample_artifact.txt"
