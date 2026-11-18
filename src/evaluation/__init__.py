"""Algorithm-agnostic evaluation metrics and plotting utilities."""

from .metrics import (
    CollisionMetrics,
    CompletionMetrics,
    EpisodeRecord,
    TaskTiming,
    bootstrap_confidence_interval,
    evaluate_episodes,
    holm_adjust,
    paired_algorithm_comparison,
)
from .trace_generator import generate_traces, write_traces


def run_benchmark(*args, **kwargs):
    """Lazily load the runner without interfering with ``python -m``."""
    from .run_benchmark import run_benchmark as _run_benchmark

    return _run_benchmark(*args, **kwargs)

__all__ = [
    "CollisionMetrics",
    "CompletionMetrics",
    "EpisodeRecord",
    "TaskTiming",
    "bootstrap_confidence_interval",
    "evaluate_episodes",
    "holm_adjust",
    "paired_algorithm_comparison",
    "run_benchmark",
    "generate_traces",
    "write_traces",
]
