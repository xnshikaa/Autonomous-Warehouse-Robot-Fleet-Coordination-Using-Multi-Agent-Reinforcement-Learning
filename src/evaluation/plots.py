"""Graphs for Week 5 evaluation result rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_results_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write structured metric rows without inventing missing measurements."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm", "fleet_size", "episodes", "tasks", "completed_tasks",
        "completion_rate", "average_completion_time_steps", "unfinished_tasks",
        "collision_events", "collision_rate_per_episode",
        "robot_robot_collision_events", "safety_overrides",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def plot_completion_time(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Plot available completion-time rows against fleet size."""

    import matplotlib.pyplot as plt

    available = [
        row for row in rows
        if row.get("average_completion_time_steps") is not None
    ]
    if not available:
        raise ValueError("No completion-time data is available for plotting.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithms = sorted({str(row["algorithm"]) for row in available})
    for algorithm in algorithms:
        points = sorted(
            (int(row["fleet_size"]), float(row["average_completion_time_steps"]))
            for row in available if str(row["algorithm"]) == algorithm
        )
        plt.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=algorithm,
        )
    plt.xlabel("Fleet size (robots)")
    plt.ylabel("Average completion time (environment steps)")
    plt.title("Average Completion Time vs Fleet Size")
    plt.xticks([5, 10, 20])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()
    return destination
