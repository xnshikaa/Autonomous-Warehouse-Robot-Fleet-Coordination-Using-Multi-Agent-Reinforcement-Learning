"""Graphs for Week 6 evaluation result rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "algorithm",
    "fleet_size",
    "episodes",
    "tasks",
    "completed_tasks",
    "completion_rate",
    "average_completion_time_steps",
    "unfinished_tasks",
    "collision_events",
    "collision_rate_per_episode",
    "robot_robot_collision_events",
    "safety_overrides",
]


def write_results_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write structured metric rows without inventing missing measurements."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return destination


def _plot_metric(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    metric: str,
    ylabel: str,
    title: str,
    *,
    percentage: bool = False,
) -> Path:
    """Plot one metric against fleet size for every available algorithm."""

    import matplotlib.pyplot as plt

    available = [
        row for row in rows
        if row.get(metric) is not None
    ]

    if not available:
        raise ValueError(f"No {metric} data is available for plotting.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    algorithms = sorted(
        {str(row["algorithm"]) for row in available}
    )

    for algorithm in algorithms:
        points = sorted(
            (
                int(row["fleet_size"]),
                float(row[metric]),
            )
            for row in available
            if str(row["algorithm"]) == algorithm
        )

        values = [point[1] for point in points]

        if percentage:
            values = [value * 100 for value in values]

        plt.plot(
            [point[0] for point in points],
            values,
            marker="o",
            label=algorithm,
        )

    plt.xlabel("Fleet size (robots)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted({int(row["fleet_size"]) for row in available}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()

    return destination


def plot_completion_time(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot average completion time against fleet size."""

    return _plot_metric(
        rows,
        output_path,
        "average_completion_time_steps",
        "Average completion time (environment steps)",
        "Average Completion Time vs Fleet Size",
    )


def plot_completion_rate(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot task completion rate against fleet size."""

    return _plot_metric(
        rows,
        output_path,
        "completion_rate",
        "Completion rate (%)",
        "Task Completion Rate vs Fleet Size",
        percentage=True,
    )


def plot_collision_rate(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot collision events per episode against fleet size."""

    return _plot_metric(
        rows,
        output_path,
        "collision_rate_per_episode",
        "Collision events per episode",
        "Collision Rate vs Fleet Size",
    )


def plot_robot_robot_collisions(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot robot-robot collision events against fleet size."""

    return _plot_metric(
        rows,
        output_path,
        "robot_robot_collision_events",
        "Robot-robot collision events",
        "Robot-Robot Collisions vs Fleet Size",
    )


def plot_safety_overrides(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot safety overrides against fleet size."""

    return _plot_metric(
        rows,
        output_path,
        "safety_overrides",
        "Safety overrides",
        "Safety Overrides vs Fleet Size",
    )


def plot_unfinished_tasks(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot unfinished tasks against fleet size."""

    return _plot_metric(
        rows,
        output_path,
        "unfinished_tasks",
        "Unfinished tasks",
        "Unfinished Tasks vs Fleet Size",
    )


def plot_all_metrics(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Generate all available Week 6 performance graphs."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            plot_completion_time,
            "completion_time_vs_fleet_size.png",
        ),
        (
            plot_completion_rate,
            "completion_rate_vs_fleet_size.png",
        ),
        (
            plot_collision_rate,
            "collision_rate_vs_fleet_size.png",
        ),
        (
            plot_robot_robot_collisions,
            "robot_robot_collisions_vs_fleet_size.png",
        ),
        (
            plot_safety_overrides,
            "safety_overrides_vs_fleet_size.png",
        ),
        (
            plot_unfinished_tasks,
            "unfinished_tasks_vs_fleet_size.png",
        ),
    ]

    generated: list[Path] = []

    for plot_function, filename in plots:
        try:
            generated.append(
                plot_function(rows, destination / filename)
            )
        except ValueError:
            # A metric may legitimately be unavailable in a trace.
            continue

    return generated