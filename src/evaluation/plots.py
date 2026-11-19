"""Graphs for Week 5 evaluation result rows."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
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
        "safety_conflict_events", "safety_conflict_rate_per_episode",
        "collision_intent_groups", "collision_intent_groups_per_episode",
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


def plot_scalability_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Plot the specification's three fleet metrics and QMIX-IQL completion gap."""
    mpl_config = Path(tempfile.gettempdir()) / "warehouse-matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["summary"]
    intervals = {
        (item["algorithm"], item["fleet_size"], item["metric"]): item
        for item in report["confidence_intervals"]
    }
    sizes = sorted({int(row["fleet_size"]) for row in rows})
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = (
        (axes[0, 0], "completion_rate", "Task completion rate", "Completion rate"),
        (axes[0, 1], "average_completion_time_steps", "Average delivery time", "Environment steps"),
        (axes[1, 0], "collision_rate_per_episode", "Robot-robot collisions", "Events per episode"),
    )
    for axis, metric, title, ylabel in panels:
        for algorithm in algorithms:
            points = [
                (row, intervals.get((algorithm, int(row["fleet_size"]), metric)))
                for row in rows
                if row["algorithm"] == algorithm and row.get(metric) is not None
            ]
            points.sort(key=lambda pair: int(pair[0]["fleet_size"]))
            if not points:
                continue
            x = [int(row["fleet_size"]) for row, _ in points]
            y = [float(row[metric]) for row, _ in points]
            lower = [float(ci["lower"]) if ci else value for value, (_, ci) in zip(y, points)]
            upper = [float(ci["upper"]) if ci else value for value, (_, ci) in zip(y, points)]
            axis.errorbar(
                x, y, yerr=[[max(0.0, v - lo) for v, lo in zip(y, lower)],
                            [max(0.0, hi - v) for v, hi in zip(y, upper)]],
                marker="o", capsize=3, label=algorithm,
            )
        axis.set(title=title, xlabel="Fleet size (robots)", ylabel=ylabel, xticks=sizes)
        axis.grid(True, alpha=0.3)

    gap_axis = axes[1, 1]
    gaps = [
        item for item in report["paired_comparisons"]
        if item.get("algorithm_a") == "QMIX"
        and item.get("algorithm_b") == "IQL"
        and item.get("metric") == "completion_rate"
        and "paired_bootstrap_ci" in item
    ]
    gaps.sort(key=lambda item: int(item["fleet_size"]))
    if gaps:
        x = [int(item["fleet_size"]) for item in gaps]
        y = [float(item["difference_a_minus_b"]) for item in gaps]
        lower = [float(item["paired_bootstrap_ci"]["lower"]) for item in gaps]
        upper = [float(item["paired_bootstrap_ci"]["upper"]) for item in gaps]
        gap_axis.errorbar(
            x, y,
            yerr=[[max(0.0, v - lo) for v, lo in zip(y, lower)],
                  [max(0.0, hi - v) for v, hi in zip(y, upper)]],
            marker="o", capsize=3, color="black",
        )
    gap_axis.axhline(0.0, color="gray", linewidth=1)
    gap_axis.set(
        title="Cooperative benefit: QMIX − IQL completion",
        xlabel="Fleet size (robots)", ylabel="Completion-rate difference", xticks=sizes,
    )
    gap_axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle("Warehouse fleet scalability (95% episode-bootstrap CIs)")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
