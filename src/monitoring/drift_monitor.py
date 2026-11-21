"""Evidently Task-Queue Data Drift Monitoring for Warehouse MARL Fleet."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.ai.environment import WarehouseEnvironment

logger = logging.getLogger(__name__)


def generate_task_queue_dataset(
    num_robots: int = 5,
    episodes: int = 10,
    demand_multiplier: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate task-queue feature distributions under normal or high-demand conditions."""
    np.random.seed(seed)
    records: List[Dict[str, float]] = []

    env = WarehouseEnvironment(num_robots=num_robots, load_layout_tasks=True)
    layout = env.get_active_layout()

    for ep in range(episodes):
        env.reset()
        num_tasks = int(round(len(env.tasks) * demand_multiplier))
        for step in range(30):
            # Sample step-level queue features
            active_tasks = [t for t in env.tasks if not t.completed]
            q_len = float(len(active_tasks) * demand_multiplier)

            distances = []
            for t in active_tasks:
                dx = abs(t.delivery_position[0] - t.pickup_position[0])
                dy = abs(t.delivery_position[1] - t.pickup_position[1])
                distances.append(float(dx + dy))

            avg_dist = float(np.mean(distances)) if distances else 5.0
            max_dist = float(np.max(distances)) if distances else 5.0
            assigned_ratio = (
                float(sum(t.assigned_robot is not None for t in active_tasks) / max(1, len(active_tasks)))
            )

            records.append({
                "queue_length": q_len + np.random.normal(0, 0.2),
                "avg_delivery_distance": avg_dist + np.random.normal(0, 0.1),
                "max_delivery_distance": max_dist + np.random.normal(0, 0.1),
                "assigned_ratio": max(0.0, min(1.0, assigned_ratio + np.random.normal(0, 0.05))),
                "robot_density": float(num_robots / (layout.width * layout.height)),
            })

    return pd.DataFrame(records)


def run_drift_analysis(
    num_robots: int = 5,
    reference_episodes: int = 10,
    current_episodes: int = 10,
    high_demand_multiplier: float = 3.0,
    output_dir: str | Path = "results/monitoring",
    psi_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Run Evidently data drift report comparing reference vs 3x high-demand current queue data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_df = generate_task_queue_dataset(
        num_robots=num_robots,
        episodes=reference_episodes,
        demand_multiplier=1.0,
        seed=100,
    )
    curr_df = generate_task_queue_dataset(
        num_robots=num_robots,
        episodes=current_episodes,
        demand_multiplier=high_demand_multiplier,
        seed=200,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    html_file = out / "drift_report.html"
    json_file = out / "drift_report.json"
    report.save_html(str(html_file))

    raw_json = json.loads(report.json())

    # Extract drifted features and trigger evaluation
    drift_metrics = raw_json.get("metrics", [{}])[0].get("result", {})
    drift_by_columns = drift_metrics.get("drift_by_columns", {})
    drifted_cols = [
        col for col, data in drift_by_columns.items()
        if data.get("drift_detected", False)
    ]

    retrain_recommended = len(drifted_cols) >= 2

    summary = {
        "dataset_drift_detected": drift_metrics.get("dataset_drift", False),
        "number_of_drifted_columns": len(drifted_cols),
        "drifted_columns": drifted_cols,
        "share_of_drifted_columns": drift_metrics.get("share_of_drifted_columns", 0.0),
        "retraining_trigger_threshold": {
            "psi_threshold": psi_threshold,
            "min_drifted_features": 2,
        },
        "retraining_recommended": retrain_recommended,
        "high_demand_multiplier": high_demand_multiplier,
        "reference_samples": len(ref_df),
        "current_samples": len(curr_df),
        "html_report": str(html_file),
    }

    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evidently drift analysis complete. Retraining recommended: {retrain_recommended}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently Task-Queue Drift Monitoring")
    parser.add_argument("--fleet-size", type=int, default=5)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--output", type=str, default="results/monitoring")
    args = parser.parse_args()

    run_drift_analysis(
        num_robots=args.fleet_size,
        high_demand_multiplier=args.multiplier,
        output_dir=args.output,
    )
