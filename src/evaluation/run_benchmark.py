"""Run reproducible QMIX/IQL/rule-based fleet-size comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.metrics import (
    EpisodeRecord,
    bootstrap_confidence_interval,
    evaluate_episodes,
    holm_adjust,
    paired_algorithm_comparison,
)
from src.evaluation.plots import plot_scalability_report
from src.evaluation.run_evaluation import load_episode_records
from src.evaluation.safety_gate import run_safety_gate
from src.evaluation.scenario_manifest import load_evaluation_manifest
from src.evaluation.trace_generator import generate_traces, write_traces


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(algorithm: str, checkpoint: Path) -> list[Path]:
    if algorithm == "IQL":
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"IQL checkpoint must be a directory: {checkpoint}")
        files = sorted(checkpoint.glob("iql_robot_*.pt"))
        if not files:
            raise FileNotFoundError(f"No IQL robot checkpoints found in {checkpoint}")
        return files
    if checkpoint.is_dir():
        files = [checkpoint / name for name in ("checkpoint_state.pt", "checkpoint_state.pkl")]
        files = [path for path in files if path.is_file()]
        if not files:
            raise FileNotFoundError(f"No QMIX state file found in {checkpoint}")
        return files
    if not checkpoint.is_file():
        raise FileNotFoundError(f"QMIX checkpoint file not found: {checkpoint}")
    return [checkpoint]


def _checkpoint_manifest(
    algorithm: str, checkpoint: Path | None, repository_root: Path
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    files = _checkpoint_files(algorithm, checkpoint)
    metadata = None
    if algorithm == "QMIX":
        state = next((path for path in files if path.suffix == ".pt"), files[0])
        payload = torch.load(state, map_location="cpu", weights_only=False)
        metadata = {
            "training_episode": payload.get("episode"),
            "training_seed": payload.get("seed"),
            "training_config": payload.get("config"),
        }
    def portable_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    return {
        "path": portable_path(checkpoint),
        "sha256": {portable_path(path): _sha256(path) for path in files},
        "metadata": metadata,
    }


def _manifest_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _episode_values(record: EpisodeRecord) -> dict[str, float | None]:
    row = evaluate_episodes([record])[0]
    return {
        "completion_rate": row["completion_rate"],
        "average_completion_time_steps": row["average_completion_time_steps"],
        "mean_completed_task_makespan_steps": row["mean_completed_task_makespan_steps"],
        "throughput_tasks_per_100_steps": (
            100.0 * row["completed_tasks"] / record.steps if record.steps else 0.0
        ),
        "collision_rate_per_episode": row["collision_rate_per_episode"],
        "safety_conflict_events": float(row["safety_conflict_events"]),
        "safety_overrides": float(row["safety_overrides"]),
        "team_reward": row["mean_episode_reward"],
    }


def _write_markdown_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write the concise scalability analysis required with benchmark outputs."""
    ci_map = {
        (row["algorithm"], row["fleet_size"], row["metric"]): row
        for row in report["confidence_intervals"]
    }
    lines = [
        "# Warehouse Fleet Scalability Evaluation",
        "",
        f"Manifest: `{report['manifest_id']}`",
        "",
        f"{report['episodes_per_run']} matched episodes were evaluated per algorithm and fleet size. "
        "Intervals are 95% episode-bootstrap confidence intervals; paired tests "
        "use task-seed-matched episodes and Holm correction across valid tests.",
        "",
        "## Metrics",
        "",
        "| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | "
        "Collisions/episode (95% CI) | Safety conflicts/episode | Throughput/100 steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["summary"], key=lambda item: (item["fleet_size"], item["algorithm"])):
        algorithm, size = row["algorithm"], row["fleet_size"]

        def estimate(metric: str, value: float | None, digits: int = 3) -> str:
            if value is None:
                return "n/a"
            ci = ci_map.get((algorithm, size, metric))
            if ci is None:
                return f"{value:.{digits}f} (CI n/a)"
            return f"{value:.{digits}f} ({ci['lower']:.{digits}f}-{ci['upper']:.{digits}f})"

        lines.append(
            f"| {algorithm} | {size} | {row['completed_tasks']}/{row['tasks']} | "
            f"{estimate('completion_rate', row['completion_rate'])} | "
            f"{estimate('average_completion_time_steps', row['average_completion_time_steps'], 2)} | "
            f"{estimate('collision_rate_per_episode', row['collision_rate_per_episode'])} | "
            f"{row['safety_conflict_rate_per_episode']:.2f} | "
            f"{row['throughput_tasks_per_100_steps']:.3f} |"
        )

    lines.extend([
        "",
        "## QMIX Cooperative Completion Gap",
        "",
        "| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |",
        "|---:|---:|---:|---:|",
    ])
    gaps = [
        item for item in report["paired_comparisons"]
        if item.get("algorithm_a") == "QMIX"
        and item.get("algorithm_b") == "IQL"
        and item.get("metric") == "completion_rate"
        and "paired_bootstrap_ci" in item
    ]
    for item in sorted(gaps, key=lambda row: row["fleet_size"]):
        ci = item["paired_bootstrap_ci"]
        lines.append(
            f"| {item['fleet_size']} | {item['difference_a_minus_b']:.3f} "
            f"({ci['lower']:.3f}-{ci['upper']:.3f}) | "
            f"{item['paired_sign_flip_p_value']:.5g} | "
            f"{item['holm_adjusted_p_value']:.5g} |"
        )

    lines.extend([
        "",
        "## Findings and Limits",
        "",
        "- The numeric N=5 QMIX point estimate meets the 80% completion target; "
        "its 95% interval extends below 80%.",
        "- The evaluation manifest was created after these checkpoints were trained. "
        "It records the tested scenarios and hashes but does not prove a pre-training sealed holdout.",
        "- The QMIX-IQL completion gap is not maintained as fleet size grows; "
        "the paired estimates are shown above.",
        f"- Dedicated corridor gate: {report['safety_gate']['attempted_corridor_entries']} "
        f"aggressive entries blocked, {report['safety_gate']['intrusions']} intrusions "
        f"in {report['safety_gate']['episodes']} episodes "
        f"({'PASS' if report['safety_gate']['passed'] else 'FAIL'}).",
        "- No robot-robot contact events were observed. Safety-prevented conflicts "
        "are measured separately and must not be interpreted as physical collisions.",
        "- Metrics requiring completed deliveries are unavailable for IQL at N=20 "
        "because it completed no tasks.",
        "- See `scalability.png` for all fleet-size curves. These results describe "
        "the supplied checkpoints and do not establish convergence.",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_benchmark(
    *,
    fleet_sizes: list[int],
    algorithms: list[str],
    episodes: int,
    max_steps: int,
    seed: int,
    qmix_checkpoints: dict[int, Path],
    iql_checkpoints: dict[int, Path],
    output_dir: Path,
    scenario_manifest_path: Path | None = None,
    bootstrap_resamples: int = 10_000,
    permutation_samples: int = 20_000,
) -> dict[str, Any]:
    """Execute matched scenarios, traces, aggregate metrics, and paired statistics."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if not fleet_sizes or any(size not in (5, 10, 20) for size in fleet_sizes):
        raise ValueError("fleet_sizes must be a non-empty selection of 5, 10, and 20")
    canonical = ["Rule-Based" if name.lower().replace("_", "-") in {"rule", "rule-based"}
                 else name.upper() for name in algorithms]
    if len(set(canonical)) != len(canonical) or any(name not in {"QMIX", "IQL", "Rule-Based"} for name in canonical):
        raise ValueError("algorithms must be unique QMIX, IQL, or Rule-Based values")

    checkpoint_map: dict[tuple[str, int], Path | None] = {}
    for size in fleet_sizes:
        for algorithm in canonical:
            checkpoint = (
                qmix_checkpoints.get(size) if algorithm == "QMIX"
                else iql_checkpoints.get(size) if algorithm == "IQL"
                else None
            )
            if algorithm != "Rule-Based" and checkpoint is None:
                raise ValueError(f"Missing {algorithm} checkpoint for fleet size {size}")
            checkpoint_map[(algorithm, size)] = checkpoint

    root = Path(__file__).resolve().parents[2]
    scenario_document = None
    scenario_sets: dict[int, list[dict[str, Any]]] = {}
    scenario_file_hash = None
    if scenario_manifest_path is not None:
        scenario_document = load_evaluation_manifest(scenario_manifest_path)
        if set(scenario_document["fleet_sizes"]) != set(fleet_sizes):
            raise ValueError("Fleet sizes must match the supplied scenario manifest")
        if scenario_document["episodes_per_fleet"] != episodes:
            raise ValueError("Episode count must match the supplied scenario manifest")
        if scenario_document["max_steps"] != max_steps:
            raise ValueError("max_steps must match the supplied scenario manifest")
        scenario_sets = {
            int(size): values for size, values in scenario_document["scenarios"].items()
        }
        if any(len(scenario_sets.get(size, [])) != episodes for size in fleet_sizes):
            raise ValueError("Scenario manifest is missing a fleet's complete episode set")
        seed = int(scenario_document["evaluation_seed"])
        scenario_file_hash = _sha256(scenario_manifest_path)
    configs = {
        name: _sha256(root / "config" / name)
        for name in ("environment.json", "reward.json")
        if (root / "config" / name).is_file()
    }
    sources = (
        "src/evaluation/metrics.py",
        "src/evaluation/trace_generator.py",
        "src/evaluation/run_evaluation.py",
        "src/evaluation/run_benchmark.py",
        "src/evaluation/plots.py",
        "src/evaluation/safety_gate.py",
        "src/evaluation/scenario_manifest.py",
        "src/ai/environment.py",
        "src/marl/environment_adapter.py",
        "src/marl/observation_encoder.py",
        "src/marl/iql_learner.py",
        "src/marl/qmix_controller.py",
        "src/marl/qmix_learner.py",
    )
    run_specs = [
        {
            "algorithm": algorithm,
            "fleet_size": size,
            "checkpoint": _checkpoint_manifest(
                algorithm, checkpoint_map[(algorithm, size)], root
            ),
        }
        for size in fleet_sizes for algorithm in canonical
    ]
    base_manifest = {
        "schema_version": 1,
        "episodes": episodes,
        "max_steps": max_steps,
        "evaluation_seed": seed,
        "task_seed_rule": (
            "explicit task queue from pretraining scenario manifest"
            if scenario_document else "evaluation_seed + episode_index - 1"
        ),
        "scenario_manifest_id": scenario_document["manifest_id"] if scenario_document else None,
        "fleet_sizes": fleet_sizes,
        "algorithms": canonical,
        "environment_config_sha256": configs,
        "source_sha256": {
            name: _sha256(root / name) for name in sources if (root / name).is_file()
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "runs": run_specs,
    }
    manifest_id = _manifest_id(base_manifest)
    manifest = {"manifest_id": manifest_id, **base_manifest}
    output_dir.mkdir(parents=True, exist_ok=True)
    if scenario_document is not None:
        scenario_copy = output_dir / "scenario_manifest.json"
        if scenario_manifest_path.resolve() != scenario_copy.resolve():
            shutil.copy2(scenario_manifest_path, scenario_copy)
        manifest["scenario_manifest"] = {
            "file": scenario_copy.relative_to(output_dir).as_posix(),
            "sha256": scenario_file_hash,
        }
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    trace_paths: dict[tuple[str, int], Path] = {}
    baseline_tasks: dict[int, list[dict[str, Any]]] = {}
    for size in fleet_sizes:
        for algorithm in canonical:
            checkpoint = checkpoint_map[(algorithm, size)]
            traces = generate_traces(
                algorithm, size, episodes, max_steps, seed, checkpoint,
                scenarios=scenario_sets.get(size),
            )
            for trace in traces:
                trace["manifest_id"] = manifest_id
                trace["environment_config_hash"] = configs.get("environment.json")
            task_sets = [trace["initial_tasks"] for trace in traces]
            if algorithm == canonical[0]:
                baseline_tasks[size] = task_sets
            elif task_sets != baseline_tasks[size]:
                raise RuntimeError(f"Task scenarios differ across algorithms for N={size}")
            run_manifest = next(
                run for run in run_specs
                if run["algorithm"] == algorithm and run["fleet_size"] == size
            )
            path = traces_dir / f"{algorithm.lower().replace('-', '_')}_n{size}.json.gz"
            write_traces(traces, path, {"manifest_id": manifest_id, **run_manifest})
            trace_paths[(algorithm, size)] = path

    records: list[EpisodeRecord] = []
    for path in trace_paths.values():
        records.extend(load_episode_records([path]))
    summaries = evaluate_episodes(records)

    confidence_intervals = []
    for algorithm in canonical:
        for size in fleet_sizes:
            group = [r for r in records if r.algorithm == algorithm and r.fleet_size == size]
            samples = [_episode_values(record) for record in group]
            metric_names = list(samples[0]) if samples else []
            for metric in metric_names:
                values = [row[metric] for row in samples if row[metric] is not None]
                if values:
                    confidence_intervals.append({
                        "algorithm": algorithm,
                        "fleet_size": size,
                        "metric": metric,
                        **bootstrap_confidence_interval(
                            values, resamples=bootstrap_resamples,
                            seed=seed + size + sum(map(ord, algorithm + metric)),
                        ),
                        "episode_samples_excluded": len(samples) - len(values),
                    })

    comparisons = []
    pair_metrics = (
        "completion_rate", "average_completion_time_steps",
        "completed_task_makespan_steps", "throughput_tasks_per_100_steps",
        "collision_rate_per_episode", "safety_conflict_events",
        "safety_overrides", "team_reward",
    )
    if "Rule-Based" in canonical:
        for size in fleet_sizes:
            for index, algorithm_a in enumerate(canonical):
                for algorithm_b in canonical[index + 1:]:
                    for metric in pair_metrics:
                        try:
                            comparisons.append(paired_algorithm_comparison(
                                records, algorithm_a, algorithm_b, size,
                                metric=metric, resamples=bootstrap_resamples,
                                permutation_samples=permutation_samples,
                                seed=seed + size + sum(map(ord, algorithm_a + algorithm_b + metric)),
                            ))
                        except ValueError as error:
                            comparisons.append({
                                "algorithm_a": algorithm_a,
                                "algorithm_b": algorithm_b,
                                "fleet_size": size,
                                "metric": metric,
                                "status": "unavailable",
                                "reason": str(error),
                            })

    test_indices = [i for i, row in enumerate(comparisons) if "paired_sign_flip_p_value" in row]
    adjusted = holm_adjust([comparisons[i]["paired_sign_flip_p_value"] for i in test_indices]) if test_indices else []
    for index, value in zip(test_indices, adjusted):
        comparisons[index]["holm_adjusted_p_value"] = value

    for run in run_specs:
        run["trace_file"] = str(trace_paths[(run["algorithm"], run["fleet_size"])].relative_to(output_dir))
        run["trace_sha256"] = _sha256(trace_paths[(run["algorithm"], run["fleet_size"])])
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    safety_gate = run_safety_gate(episodes=20, fleet_size=5, max_steps=200)
    safety_path = output_dir / "safety_gate.json"
    safety_path.write_text(json.dumps(safety_gate, indent=2, allow_nan=False), encoding="utf-8")
    report = {
        "manifest_id": manifest_id,
        "episodes_per_run": episodes,
        "fleet_sizes": fleet_sizes,
        "algorithms": canonical,
        "summary": summaries,
        "confidence_intervals": confidence_intervals,
        "paired_comparisons": comparisons,
        "safety_gate": safety_gate,
    }
    plot_path = output_dir / "scalability.png"
    plot_scalability_report(report, plot_path)
    report_path = _write_markdown_report(report, output_dir / "report.md")
    manifest["scalability_plot"] = {
        "file": plot_path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(plot_path),
    }
    manifest["analysis_report"] = {
        "file": report_path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(report_path),
    }
    manifest["safety_gate"] = {
        "file": safety_path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(safety_path),
    }
    report["scalability_plot"] = manifest["scalability_plot"]
    report["analysis_report"] = manifest["analysis_report"]
    report["safety_gate_artifact"] = manifest["safety_gate"]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {"manifest": manifest, **report}


def _fleet_checkpoints(values: list[str], label: str) -> dict[int, Path]:
    result = {}
    for value in values:
        try:
            size, path = value.split("=", 1)
            result[int(size)] = Path(path)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{label} entries must use N=PATH") from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-sizes", nargs="+", type=int, choices=(5, 10, 20), default=[5, 10, 20])
    parser.add_argument("--algorithms", nargs="+", choices=("QMIX", "IQL", "Rule-Based"), default=["QMIX", "IQL", "Rule-Based"])
    parser.add_argument("--qmix-checkpoint", action="append", default=[], metavar="N=PATH")
    parser.add_argument("--iql-checkpoint", action="append", default=[], metavar="N=DIR")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-manifest", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=20_000)
    args = parser.parse_args()
    report = run_benchmark(
        fleet_sizes=args.fleet_sizes,
        algorithms=args.algorithms,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        qmix_checkpoints=_fleet_checkpoints(args.qmix_checkpoint, "--qmix-checkpoint"),
        iql_checkpoints=_fleet_checkpoints(args.iql_checkpoint, "--iql-checkpoint"),
        output_dir=args.output,
        scenario_manifest_path=args.scenario_manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_samples=args.permutation_samples,
    )
    print(f"Wrote benchmark {report['manifest_id']} to {args.output}")


if __name__ == "__main__":
    main()
