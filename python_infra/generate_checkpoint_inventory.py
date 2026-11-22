"""Generate exhaustive CHECKPOINT_INVENTORY.md auditing all repository checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.marl.checkpoint_validator import extract_or_infer_metadata
from src.evaluation.trace_generator import generate_traces


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_completed_run(run: Path, root: Path) -> dict[str, Any]:
    """Verify the final policy and training history against the saved run."""
    manifest = json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))
    algorithm, fleet = manifest["algorithm"], manifest["fleet_size"]
    history = json.loads((run / manifest["history"]).read_text(encoding="utf-8"))
    completed = manifest["episodes_completed_total"]
    expected_episodes = list(range(completed - manifest["episodes"] + 1, completed + 1))
    if [row["episode"] for row in history] != expected_episodes:
        raise ValueError(f"Incomplete episode history in {run}")
    if not all(math.isfinite(row[key]) for row in history for key in ("loss", "reward")):
        raise ValueError(f"Non-finite training metrics in {run}")

    checkpoint = run / manifest["checkpoint"]
    files = [checkpoint] if algorithm == "QMIX" else sorted(checkpoint.iterdir())
    hashes = {}
    for name, expected in manifest["sha256"].items():
        artifact = run / name.replace("\\", "/")
        if sha256_file(artifact) != expected:
            raise ValueError(f"Checksum mismatch for {artifact}")
        if artifact in files:
            hashes[artifact.relative_to(root).as_posix()] = expected
    if len(hashes) != len(files):
        raise ValueError(f"Missing checksums for final policy in {run}")

    steps = sum(row["steps"] for row in history)
    final_snapshot = run / "snapshots" / f"episode_{completed:05d}"
    if algorithm == "QMIX":
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = extract_or_infer_metadata(model, str(checkpoint))
        if (metadata.episode, metadata.num_agents) != (completed, fleet):
            raise ValueError(f"Final checkpoint episode or fleet mismatch in {run}")
        snapshot = torch.load(
            final_snapshot / "checkpoint_state.pt", map_location="cpu", weights_only=False
        )
        for network in ("agent_network", "mixer"):
            if model[network].keys() != snapshot[network].keys() or any(
                not torch.equal(tensor, snapshot[network][name])
                for name, tensor in model[network].items()
            ):
                raise ValueError(f"Final {network} differs from its last snapshot in {run}")
        optimizers = [model["optimizer"]]
        state_dim = metadata.state_dim
        schema = metadata.schema_version
        observation_dim = metadata.observation_dim
        action_dim = metadata.action_dim
    else:
        metadata = json.loads((checkpoint / "iql_manifest.json").read_text(encoding="utf-8"))
        if (metadata["episodes_completed"], metadata["num_robots"]) != (completed, fleet):
            raise ValueError(f"Final checkpoint episode or fleet mismatch in {run}")
        if any(sha256_file(final_snapshot / artifact.name) != hashes[artifact.relative_to(root).as_posix()]
               for artifact in files):
            raise ValueError(f"Final IQL checkpoint differs from its last snapshot in {run}")
        optimizers = [
            torch.load(checkpoint / f"iql_robot_{robot}.pt", map_location="cpu", weights_only=True)["optimizer"]
            for robot in range(fleet)
        ]
        state_dim = None
        schema = metadata["schema_version"]
        observation_dim = metadata["observation_dim"]
        action_dim = metadata["action_dim"]
    updates = []
    for optimizer in optimizers:
        counters = {int(state["step"]) for state in optimizer["state"].values()}
        if len(counters) != 1 or min(counters) <= 0:
            raise ValueError(f"Missing or inconsistent optimizer updates in {run}")
        updates.append(counters.pop())
    if algorithm == "QMIX" and updates != [model["update_count"]]:
        raise ValueError(f"QMIX optimizer and checkpoint counters disagree in {run}")

    generate_traces(algorithm, fleet, 1, 1, 20261120, checkpoint)
    return {
        "algorithm": algorithm,
        "fleet_size": fleet,
        "checkpoint": checkpoint.relative_to(root).as_posix(),
        "observation_dim": observation_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "checkpoint_schema": schema,
        "training_seed": manifest["seed"],
        "episodes": completed,
        "environment_steps": steps,
        "optimizer_updates": updates,
        "inference_verified": True,
        "training_duration_seconds": manifest.get("training_duration_seconds"),
        "mlflow_run_id": manifest.get("mlflow_run_id"),
        "history": (run / manifest["history"]).relative_to(root).as_posix(),
        "history_sha256": sha256_file(run / manifest["history"]),
        "sha256": hashes,
    }


def find_evaluated_checkpoints(root: Path) -> set[tuple[str, frozenset[str]]]:
    """Find exact checkpoint file sets referenced by benchmark manifests."""
    evaluated = set()
    for manifest_path in root.glob("results/**/manifest.json"):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            runs = data.get("runs", [])
            for run in runs:
                ckpt = run.get("checkpoint") or {}
                hashes = frozenset((ckpt.get("sha256") or {}).values())
                path = ckpt.get("path")
                if path and hashes:
                    evaluated.add((path.replace("\\", "/"), hashes))
        except (OSError, ValueError):
            pass
    return evaluated


def audit_qmix_checkpoint(path: Path, evaluated_set: set[tuple[str, frozenset[str]]], root: Path) -> dict[str, Any]:
    file_sha = sha256_file(path)
    rel_path = str(path.relative_to(root)).replace("\\", "/")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        metadata = extract_or_infer_metadata(data, str(path))
        episodes = data.get("episode", metadata.episode)
        updates = data.get("update_count") or 0
        run_id = metadata.run_id or (metadata.config or {}).get("run_id") or "untracked"

        # Check if evaluated
        is_eval = (rel_path, frozenset([file_sha])) in evaluated_set
        status = "evaluated" if is_eval else "trained but unevaluated"
        if updates == 0:
            status = "no optimizer updates verified"
        if metadata.observation_dim != 456 or metadata.state_dim != 11 * metadata.num_agents:
            status = "unsupported by current inference (legacy dimensions)"

        return {
            "algorithm": "QMIX",
            "fleet_size": metadata.num_agents,
            "obs_dim": metadata.observation_dim,
            "state_dim": metadata.state_dim,
            "episodes": episodes if episodes is not None else "unknown",
            "updates": updates,
            "run_id": run_id,
            "path": rel_path,
            "sha256": file_sha,
            "status": status,
        }
    except Exception as exc:
        return {
            "algorithm": "QMIX (corrupt/unsupported)",
            "fleet_size": "unknown",
            "obs_dim": "unknown",
            "state_dim": "unknown",
            "episodes": "unknown",
            "updates": 0,
            "run_id": "none",
            "path": rel_path,
            "sha256": file_sha,
            "status": f"unsupported ({exc})",
        }


def audit_iql_dir(agent_dir: Path, evaluated_set: set[tuple[str, frozenset[str]]], root: Path) -> dict[str, Any]:
    rel_path = str(agent_dir.relative_to(root)).replace("\\", "/")
    manifest_file = agent_dir / "iql_manifest.json"
    robot_files = sorted(agent_dir.glob("iql_robot_*.pt"))

    hashes = [sha256_file(path) for path in robot_files]
    dir_sha = hashlib.sha256("".join(hashes).encode()).hexdigest()

    if manifest_file.is_file():
        try:
            m = json.loads(manifest_file.read_text(encoding="utf-8"))
            num_robots = m.get("num_robots", len(robot_files))
            obs_dim = m.get("observation_dim", 456)
            episodes = m.get("episodes_completed", "unknown")
            run_id = m.get("run_id", "untracked")
        except Exception:
            num_robots = len(robot_files)
            obs_dim = 456
            episodes = "unknown"
            run_id = "unknown"
    else:
        num_robots = len(robot_files)
        obs_dim = 456
        episodes = "unknown"
        run_id = "unknown"

    if manifest_file.is_file():
        hashes.append(sha256_file(manifest_file))
    is_eval = (rel_path, frozenset(hashes)) in evaluated_set
    status = "evaluated" if is_eval else "trained but unevaluated"
    updates = []
    for path in robot_files:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        counters = {int(state["step"]) for state in weights["optimizer"]["state"].values()}
        if len(counters) == 1:
            updates.append(counters.pop())
    if len(updates) != len(robot_files) or min(updates, default=0) == 0:
        status = "no optimizer updates verified"

    return {
        "algorithm": "IQL",
        "fleet_size": num_robots,
        "obs_dim": obs_dim,
        "state_dim": "n/a (decentralized)",
        "episodes": episodes,
        "updates": f"{sum(updates):,} total",
        "run_id": run_id,
        "path": rel_path,
        "sha256": dir_sha,
        "status": status,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    evaluated_set = find_evaluated_checkpoints(root)
    completed_runs = [
        verify_completed_run(root / "artifacts" / "training" / f"{algorithm}_n{fleet}_v1", root)
        for fleet in (5, 10, 20) for algorithm in ("qmix", "iql")
    ]
    evidence = root / "artifacts" / "checkpoint_verification.json"
    evidence.write_text(json.dumps(completed_runs, indent=2) + "\n", encoding="utf-8")
    experiments = [
        verify_completed_run(path.parent, root)
        for path in sorted((root / "artifacts/training").glob("*_remediation_*/training_manifest.json"))
    ]
    if experiments:
        experiment_evidence = root / "artifacts/remediation_checkpoint_verification.json"
        experiment_evidence.write_text(json.dumps(experiments, indent=2) + "\n", encoding="utf-8")

    records = []

    # Audit QMIX .pt files
    for pt in root.glob("artifacts/**/*.pt"):
        # Skip individual IQL robot weights (they are audited at directory level)
        if "iql_robot_" in pt.name:
            continue
        records.append(audit_qmix_checkpoint(pt, evaluated_set, root))

    # Audit IQL directories
    seen_iql_dirs = set()
    for rf in root.glob("artifacts/**/iql_robot_*.pt"):
        d = rf.parent
        if d not in seen_iql_dirs:
            seen_iql_dirs.add(d)
            records.append(audit_iql_dir(d, evaluated_set, root))

    records.sort(key=lambda r: (str(r["algorithm"]), str(r["fleet_size"]), str(r["path"])))
    final_paths = {
        (path.parent / json.loads(path.read_text(encoding="utf-8"))["checkpoint"]).relative_to(root).as_posix()
        for path in (root / "artifacts/training").glob("*/training_manifest.json")
    }
    validation_candidates = set()
    validation_winners = set()
    for directory in ("project25_validation", "project25_remediation"):
        for path in (root / "results" / directory).glob("*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            if "selected_checkpoint" in report:
                validation_winners.add(report["selected_checkpoint"].replace("\\", "/"))
                validation_candidates.update(row["candidate"].replace("\\", "/") for row in report["results"])
    for record in records:
        path = record["path"]
        if path in final_paths:
            record["status"] += ", final weights"
        if path in validation_winners:
            record["status"] += ", validation winner"
        elif path in validation_candidates:
            record["status"] += ", validation candidate"

    # Generate Markdown
    lines = [
        "# Complete Checkpoint & Model Inventory",
        "",
        "Audited catalog of all reinforcement learning checkpoints across the project repository, "
        "tracking algorithm architecture, fleet dimensions, training provenance, SHA-256 integrity, "
        "and evaluation lifecycle status.",
        "",
        "## Verified Final Training Runs",
        "",
        "All six final policies passed saved checksum verification, episode-history checks, "
        "optimizer counter checks and action selection through the evaluation inference path.",
        "Rule-Based uses no learned checkpoint.",
        "Final weights match each run's last snapshot. Full hashes, history paths, schema versions "
        "and optimizer counters are in "
        "[checkpoint_verification.json](artifacts/checkpoint_verification.json).",
        "",
        "| Algorithm | N | Observation | Global state | Episodes | Environment steps | Optimizer updates (total) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {r['algorithm']} | {r['fleet_size']} | {r['observation_dim']} | "
            f"{r['state_dim'] if r['state_dim'] is not None else 'n/a'} | "
            f"{r['episodes']:,} | {r['environment_steps']:,} | {sum(r['optimizer_updates']):,} |"
            for r in completed_runs
        ],
        "",
        "The historical manifests do not record wall-clock duration or an MLflow run ID. "
        "IQL update totals sum independent optimizers across the fleet.",
        "",
        "The trained live demo uses the three QMIX snapshots recorded in "
        "[the validation selections](results/project25_validation/).",
        "Remediation training artifacts are listed separately by run path. Their full "
        "checksums and optimizer counters are in "
        "[remediation checkpoint verification](artifacts/remediation_checkpoint_verification.json).",
        "",
        "## Inventory Table",
        "",
        "| Algorithm | N | Obs Dim | State Dim | Episodes | Updates | Run ID | File / Directory Path | SHA-256 (Prefix) | Evaluation Status |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]

    for r in records:
        lines.append(
            f"| {r['algorithm']} | {r['fleet_size']} | {r['obs_dim']} | {r['state_dim']} | "
            f"{r['episodes']} | {r['updates']} | {r['run_id']} | `{r['path']}` | "
            f"`{r['sha256'][:16]}...` | **{r['status']}** |"
        )

    lines.extend([
        "",
        "## Status Taxonomy",
        "- **`evaluated`**: The exact path and file digests match a benchmark manifest. "
        "Identical bytes at another path retain their own lifecycle status.",
        "- **`trained but unevaluated`**: Positive optimizer counters with no matching benchmark reference.",
        "- **`no optimizer updates verified`**: No positive training counter was recovered.",
        "",
        "## Reproducibility and Validation Rules",
        "1. Checkpoints with incompatible fleet sizes (N=5 vs N=10 vs N=20) or observation dimensions are strictly rejected by `checkpoint_validator.py`.",
        "2. No checkpoint is silently reinitialized or partially loaded (`strict=True` is enforced across all loaders).",
        "3. Models evaluated on held-out test scenarios are strictly selected using the separate validation set (`select_checkpoint.py`).",
    ])

    out_file = root / "CHECKPOINT_INVENTORY.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {out_file} with {len(records)} checkpoints audited.")


if __name__ == "__main__":
    main()
