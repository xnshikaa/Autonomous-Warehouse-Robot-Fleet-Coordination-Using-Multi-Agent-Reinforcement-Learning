"""
MLflow Tracing entry point for Autonomous Warehouse MARL training loop.

Sets tracking URI and experiment, then exposes @mlflow.trace-decorated
wrappers around the key warehouse training operations so every run produces
a trace that is searchable in the MLflow UI.

Usage (smoke-test):
    python -m python_infra.tracing_demo

Usage (future EPyMARL integration):
    from python_infra.tracing import (
        trace_episode,
        trace_checkpoint_save,
        trace_metric_collection,
        configure_tracing,
    )
"""

import os
import mlflow
from mlflow.entities import SpanType


def configure_tracing(experiment_name: str = "my-experiment") -> str:
    """
    Configure MLflow tracking URI and experiment.

    Resolution order for tracking URI:
      1. MLFLOW_TRACKING_URI environment variable (if set by the user)
      2. Local SQLite database ./mlflow.db (student-project default, full feature support)

    Returns the active tracking URI.
    """
    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{os.path.abspath('mlflow.db')}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


# ──────────────────────────────────────────────────────────────────────────────
# Traced operations — use @mlflow.trace decorator (auto-captures I/O)
# ──────────────────────────────────────────────────────────────────────────────

@mlflow.trace(name="run_episode", span_type=SpanType.CHAIN)
def trace_episode(
    episode: int,
    num_agents: int,
    max_steps: int,
    algorithm: str,
    *,
    _step_fn=None,   # callable(step) -> dict  (metrics for that step)
):
    """
    Trace one full training episode.

    Records: episode number, fleet size, algorithm, per-step metrics.
    Returns a dict of episode-level aggregated metrics.
    """
    rewards = []
    tasks_completed = []
    collisions = []

    for step in range(max_steps):
        if _step_fn is not None:
            step_result = trace_env_step(episode=episode, step=step, fn=_step_fn)
        else:
            # Smoke-test fallback: synthetic values
            step_result = {
                "reward": -1.0 + step * 0.02,
                "tasks_completed": step % 3,
                "collisions": max(0, 2 - step // 5),
            }
        rewards.append(step_result.get("reward", 0.0))
        tasks_completed.append(step_result.get("tasks_completed", 0))
        collisions.append(step_result.get("collisions", 0))

    episode_metrics = {
        "episode": episode,
        "algorithm": algorithm,
        "num_agents": num_agents,
        "episode_reward": sum(rewards),
        "completed_tasks": sum(tasks_completed),
        "collision_or_conflict_count": sum(collisions),
        "episode_length": max_steps,
    }
    return episode_metrics


@mlflow.trace(name="env_step", span_type=SpanType.TOOL)
def trace_env_step(*, episode: int, step: int, fn=None):
    """
    Trace a single environment step (action → obs → reward).
    """
    if fn is not None:
        return fn(step)
    return {"reward": -1.0 + step * 0.02, "tasks_completed": step % 3, "collisions": max(0, 2 - step // 5)}


@mlflow.trace(name="collect_metrics", span_type=SpanType.TOOL)
def trace_metric_collection(episode_metrics: dict, episode: int):
    """
    Trace the metric-collection step: receives raw episode metrics,
    returns a cleaned dict ready for MLflow logging.
    """
    return {
        "episode_reward": float(episode_metrics.get("episode_reward", 0.0)),
        "task_completion_rate": min(1.0, episode_metrics.get("completed_tasks", 0) / 10.0),
        "completed_tasks": int(episode_metrics.get("completed_tasks", 0)),
        "collision_or_conflict_count": int(episode_metrics.get("collision_or_conflict_count", 0)),
        "episode_length": int(episode_metrics.get("episode_length", 0)),
    }


@mlflow.trace(name="save_checkpoint", span_type=SpanType.TOOL)
def trace_checkpoint_save(
    checkpoint_mgr,
    checkpoint_data: dict,
    step: int,
    tracker=None,
):
    """
    Trace a checkpoint-save operation so the trace records exactly which
    step was persisted and which run owns the file.
    """
    path = checkpoint_mgr.save_checkpoint(
        checkpoint_data=checkpoint_data,
        step=step,
        tracker=tracker,
    )
    return {"saved_to": path, "step": step}


# ──────────────────────────────────────────────────────────────────────────────
# Demo: run a traced smoke-test loop and verify the trace was recorded
# ──────────────────────────────────────────────────────────────────────────────

def run_tracing_demo():
    """
    Runs a short traced loop against the 'my-experiment' experiment and verifies
    at least one trace is recorded.

    ALL metrics are SYNTHETIC INFRASTRUCTURE TEST DATA.
    This does not represent real QMIX or IQL training results.
    """
    print("=" * 72)
    print("  AUTONOMOUS WAREHOUSE MARL — MLflow TRACING DEMO")
    print("  [INFRASTRUCTURE TEST — all metrics are synthetic]")
    print("=" * 72)

    uri = configure_tracing(experiment_name="my-experiment")
    print(f"[Tracing] Tracking URI : {uri}")
    print(f"[Tracing] Experiment   : my-experiment")

    with mlflow.start_run(run_name="tracing_demo_run") as run:
        mlflow.set_tags({
            "project": "Autonomous-Warehouse-MARL",
            "observation_vector": "50-D",
            "action_space": "Discrete(4)",
            "experiment_type": "tracing_smoke_test",
            "is_synthetic": "true",
        })
        mlflow.log_params({
            "algorithm": "QMIX",
            "num_agents": 10,
            "observation_dim": 50,
            "action_space": "Discrete(4)",
        })

        for ep in range(1, 4):   # 3 traced episodes
            episode_metrics = trace_episode(
                episode=ep,
                num_agents=10,
                max_steps=5,
                algorithm="QMIX",
            )
            metrics = trace_metric_collection(episode_metrics, episode=ep)
            mlflow.log_metrics(metrics, step=ep)
            print(f"  Episode {ep}/3 | reward={metrics['episode_reward']:.2f} | "
                  f"tasks={metrics['completed_tasks']} | "
                  f"collisions={metrics['collision_or_conflict_count']}")

        run_id = run.info.run_id

    # Flush async trace logging before searching
    mlflow.flush_trace_async_logging()

    # Verify traces were recorded
    exp = mlflow.get_experiment_by_name("my-experiment")
    traces = mlflow.search_traces(locations=[exp.experiment_id])
    print(f"\n[Verification] Found {len(traces)} trace(s) in 'my-experiment'")
    assert len(traces) > 0, "No traces found — check tracking URI and experiment settings"

    # Show spans for the first trace
    trace_id = traces.iloc[0]["trace_id"]
    spans = mlflow.get_trace(trace_id).data.spans
    print(f"[Verification] Trace '{trace_id[:16]}...' has {len(spans)} span(s):")
    for span in spans:
        print(f"    - {span.name}  ({span.span_type})")

    print("\n" + "=" * 72)
    print("  TRACING DEMO COMPLETE — trace visible in MLflow UI")
    print(f"  http://127.0.0.1:5000  ->  Experiments -> my-experiment")
    print("=" * 72)
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tracing_demo() else 1)
