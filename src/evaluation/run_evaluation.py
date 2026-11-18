"""Evaluate JSON episode traces and optionally create the completion graph."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import fields
from pathlib import Path

from src.evaluation.metrics import EpisodeRecord, evaluate_episodes
from src.evaluation.plots import plot_completion_time, write_results_csv


def load_episode_records(paths: list[Path]) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    record_fields = {field.name for field in fields(EpisodeRecord)}
    for path in paths:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                document = json.load(stream)
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
        for episode in document.get("episodes", []):
            records.append(
                EpisodeRecord(**{
                    key: value for key, value in episode.items()
                    if key in record_fields
                })
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rows = evaluate_episodes(load_episode_records(args.traces))
    write_results_csv(rows, args.csv)
    if args.plot:
        plot_completion_time(rows, args.plot)
    print(f"Wrote {len(rows)} aggregated result rows to {args.csv}")


if __name__ == "__main__":
    main()
