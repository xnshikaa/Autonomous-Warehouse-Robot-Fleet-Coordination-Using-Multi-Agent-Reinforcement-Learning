"""Unit tests for Evidently drift monitoring and CodeCarbon sustainability tracker."""

from pathlib import Path
import pytest
from src.monitoring.drift_monitor import generate_task_queue_dataset, run_drift_analysis
from src.monitoring.carbon_tracker import SustainabilityTracker


def test_drift_dataset_generation():
    df = generate_task_queue_dataset(num_robots=5, episodes=2, demand_multiplier=1.0)
    assert len(df) == 60  # 2 episodes * 30 steps
    assert "queue_length" in df.columns
    assert "avg_delivery_distance" in df.columns
    assert "robot_density" in df.columns


def test_drift_analysis_execution(tmp_path):
    summary = run_drift_analysis(
        num_robots=5,
        reference_episodes=2,
        current_episodes=2,
        high_demand_multiplier=3.0,
        output_dir=tmp_path / "monitoring",
    )

    assert "dataset_drift_detected" in summary
    assert "drifted_columns" in summary
    assert (tmp_path / "monitoring" / "drift_report.html").is_file()
    assert (tmp_path / "monitoring" / "drift_report.json").is_file()


def test_sustainability_tracker_calculations(tmp_path):
    tracker = SustainabilityTracker(algorithm="QMIX", fleet_size=5, output_dir=tmp_path / "carbon")
    tracker.start_tracking()
    report = tracker.stop_tracking(total_steps=100, completed_deliveries=5, idle_steps=10)

    assert report["algorithm"] == "QMIX"
    assert report["fleet_size"] == 5
    assert report["simulated_robot_physical_energy"]["total_travel_distance_meters"] == 100.0
    assert report["simulated_robot_physical_energy"]["energy_per_delivery_joules"] > 0
    assert (tmp_path / "carbon" / "sustainability_qmix_n5.json").is_file()
