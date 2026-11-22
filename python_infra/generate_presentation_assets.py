"""Generate genuine presentation figures and master CSV from frozen holdout benchmark data."""
import json
import csv
from pathlib import Path
import os

# Set writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent)

import matplotlib.pyplot as plt
import numpy as np

def generate_assets():
    project_root = Path(__file__).resolve().parents[1]
    holdout_summary_file = project_root / "results" / "project25_holdout" / "summary.json"
    output_dir = project_root / "presentation_materials"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(holdout_summary_file, "r") as f:
        data = json.load(f)

    # 1. Master CSV Table
    csv_file = output_dir / "EXPERIMENT_RESULTS_MASTER_TABLE.csv"
    summary_rows = data["summary"]
    
    headers = [
        "manifest_id", "algorithm", "fleet_size", "episodes", "tasks", "completed_tasks",
        "completion_rate", "average_completion_time_steps", "throughput_tasks_per_100_steps",
        "physical_collision_events", "safety_conflict_events", "safety_conflict_rate_per_episode",
        "collision_intent_groups_per_episode", "mean_episode_reward"
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in summary_rows:
            writer.writerow([
                data.get("manifest_id", "e47ae5704f026c96"),
                row["algorithm"],
                row["fleet_size"],
                row["episodes"],
                row["tasks"],
                row["completed_tasks"],
                f"{row['completion_rate'] * 100:.2f}%",
                f"{row.get('average_completion_time_steps', 0):.2f}",
                f"{row.get('throughput_tasks_per_100_steps', 0):.3f}",
                row["collision_events"],
                row["safety_conflict_events"],
                f"{row['safety_conflict_rate_per_episode']:.2f}",
                f"{row.get('collision_intent_groups_per_episode', 0):.2f}",
                f"{row.get('mean_episode_reward', 0):.2f}"
            ])
    print(f"Generated {csv_file}")

    # Set overall chart styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    # 2. Figure 1: Task Completion Rate by Fleet Size
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fleets = [5, 10, 20]
    algorithms = ["QMIX", "IQL", "Rule-Based"]
    colors = {"QMIX": "#2563eb", "IQL": "#dc2626", "Rule-Based": "#475569"}

    # Exact 95% bootstrap intervals from report.md
    completion_rates = {
        "QMIX": [78.0, 37.0, 0.75],
        "IQL": [78.0, 23.0, 0.25],
        "Rule-Based": [46.0, 39.5, 13.75]
    }
    ci_lower = {
        "QMIX": [70.0, 28.0, 0.0],
        "IQL": [70.0, 17.5, 0.0],
        "Rule-Based": [34.0, 30.5, 10.25]
    }
    ci_upper = {
        "QMIX": [86.0, 46.0, 1.5],
        "IQL": [86.0, 28.5, 0.75],
        "Rule-Based": [59.0, 49.5, 17.25]
    }

    x = np.arange(len(fleets))
    width = 0.26

    for i, algo in enumerate(algorithms):
        means = completion_rates[algo]
        yerr = [
            np.array(means) - np.array(ci_lower[algo]),
            np.array(ci_upper[algo]) - np.array(means)
        ]
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=yerr, capsize=4,
               label=algo, color=colors[algo], alpha=0.88, edgecolor="black", linewidth=0.8)
        # Add value labels
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Add NFR-1 line
    ax.axhline(80, color="#16a34a", linestyle="--", linewidth=1.5, label="NFR-1 Target (80%)")
    ax.set_xlabel("Fleet Size (Number of Robots)", fontweight="bold")
    ax.set_ylabel("Task Completion Rate (%)", fontweight="bold")
    ax.set_title("Project 25: Task Completion Rate vs Fleet Size (Frozen 20-Ep Holdout)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["N = 5", "N = 10", "N = 20"])
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_path = figures_dir / "task_completion_comparison.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Generated {fig1_path}")

    # 3. Figure 2: Safety Conflicts (Interventions) Comparison (Log Scale)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    conflicts = {
        "QMIX": [1.10, 2.15, 7.75],
        "IQL": [34.65, 88.85, 249.90],
        "Rule-Based": [274.70, 641.10, 1726.10]
    }
    for i, algo in enumerate(algorithms):
        means = conflicts[algo]
        bars = ax.bar(x + (i - 1) * width, means, width,
               label=algo, color=colors[algo], alpha=0.88, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{val:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("Fleet Size (Number of Robots)", fontweight="bold")
    ax.set_ylabel("Safety Conflict Events / Episode (Log Scale)", fontweight="bold")
    ax.set_title("Safety Conflict Events per Episode (Collision Interventions)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["N = 5", "N = 10", "N = 20"])
    ax.set_ylim(0.5, 3000)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig2_path = figures_dir / "safety_conflicts_comparison.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Generated {fig2_path}")

    # 4. Figure 3: Average Delivery Time
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    deliv_times = {
        "QMIX": [34.78, 34.40, 42.33],
        "IQL": [37.10, 36.31, 45.00],
        "Rule-Based": [27.38, 29.71, 25.08]
    }
    for i, algo in enumerate(algorithms):
        means = deliv_times[algo]
        bars = ax.bar(x + (i - 1) * width, means, width,
               label=algo, color=colors[algo], alpha=0.88, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{val:.1f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Fleet Size (Number of Robots)", fontweight="bold")
    ax.set_ylabel("Average Completed Delivery Steps", fontweight="bold")
    ax.set_title("Average Delivery Time for Completed Tasks", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["N = 5", "N = 10", "N = 20"])
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig3_path = figures_dir / "delivery_time_comparison.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"Generated {fig3_path}")

    # 5. Figure 4: Throughput per 100 Steps
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    throughput = {
        "QMIX": [5.76, 3.70, 0.15],
        "IQL": [5.14, 2.30, 0.05],
        "Rule-Based": [3.73, 3.95, 2.75]
    }
    for i, algo in enumerate(algorithms):
        means = throughput[algo]
        bars = ax.bar(x + (i - 1) * width, means, width,
               label=algo, color=colors[algo], alpha=0.88, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Fleet Size (Number of Robots)", fontweight="bold")
    ax.set_ylabel("Throughput (Deliveries / 100 Simulation Steps)", fontweight="bold")
    ax.set_title("System Delivery Throughput Across Fleet Sizes", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["N = 5", "N = 10", "N = 20"])
    ax.set_ylim(0, 8)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig4_path = figures_dir / "throughput_comparison.png"
    plt.savefig(fig4_path)
    plt.close()
    print(f"Generated {fig4_path}")

if __name__ == "__main__":
    generate_assets()
