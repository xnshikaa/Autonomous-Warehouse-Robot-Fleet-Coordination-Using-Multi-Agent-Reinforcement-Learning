"""Evaluate JSON episode traces and create performance graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import EpisodeRecord, evaluate_episodes
from src.evaluation.plots import (
    plot_all_metrics,
    write_results_csv,
)


def load_episode_records(paths: list[Path]) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []

    for path in paths:
        document = json.loads(
            path.read_text(encoding="utf-8")
        )

        for episode in document.get("episodes", []):
            records.append(
            EpisodeRecord(
                algorithm=episode["algorithm"],
                fleet_size=episode["fleet_size"],
                episode=episode["episode"],
                task_snapshots=episode.get("task_snapshots", []),
                collision_events=episode.get("collision_events", []),
                safety_overrides=episode.get("safety_overrides", 0),
            )
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--plot",
        type=Path,
        help="Generate only the completion-time graph.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Generate all available Week 6 performance graphs.",
    )

    args = parser.parse_args()

    rows = evaluate_episodes(
        load_episode_records(args.traces)
    )

    write_results_csv(rows, args.csv)

    if args.plot:
        from src.evaluation.plots import plot_completion_time

        plot_completion_time(rows, args.plot)

    if args.output_dir:
        generated = plot_all_metrics(
            rows,
            args.output_dir,
        )

        for path in generated:
            print(f"Generated graph: {path}")

    print(
        f"Wrote {len(rows)} aggregated result rows "
        f"to {args.csv}"
    )


if __name__ == "__main__":
    main()