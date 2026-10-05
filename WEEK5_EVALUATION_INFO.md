# Week 5 Evaluation Information

This document summarizes the Week 5 Member 3 evaluation work prepared in the
isolated `week5-work` folder. It is intended as a handoff document for the
next team member and for integration into the overall project.

## Scope

The Week 5 evaluation implementation covers:

- Completion-time metrics.
- Robot-robot collision statistics.
- Algorithm-agnostic aggregation for QMIX, IQL, Rule-Based, and other
  algorithm names.
- Fleet-size comparison for N=5, N=10, and N=20.
- Structured CSV output.
- Completion-time graph generation.

The original repository and its existing branches were not modified.

## Evaluation files

### `src/evaluation/metrics.py`

This module:

- Calculates task completion time in environment steps.
- Handles completed tasks and unfinished tasks.
- Aggregates metrics across episodes.
- Aggregates results by algorithm and fleet size.
- Counts collision events.
- Separates robot-robot collisions from other collision event types.
- Keeps safety overrides separate from collisions.
- Includes collision inference matching the existing environment rules:
  - Multiple robots targeting the same cell count as one conflict event.
  - Two robots swapping positions count as one conflict event.

The evaluator consumes episode traces and does not depend on a specific
controller. This allows the same metric definitions to be applied to QMIX,
IQL, and Rule-Based evaluation runs.

### `src/evaluation/plots.py`

This module:

- Writes structured metric rows to CSV.
- Generates an average completion-time versus fleet-size graph.
- Distinguishes available algorithm result series in the graph.
- Supports QMIX, IQL, Rule-Based, and other algorithm names without separate
  plotting implementations.
- Does not fabricate missing result data. It raises an error when no
  completion-time measurements are available for plotting.

### `tests/test_evaluation_metrics.py`

The Week 5 tests cover:

- Known task start and completion timing.
- Multiple episode aggregation.
- Collision aggregation.
- Empty episodes and missing metric data.
- Same-target robot collision detection.
- Separation of safety overrides from collisions.

## Definitions based on the existing project

The existing environment defines task completion when an assigned robot
reaches the task's `delivery_position`.

Episode time is represented by the environment's discrete step counter.
Therefore, completion time is reported as environment steps rather than wall
clock seconds.

The existing collision implementation identifies robot-robot conflicts when:

1. Two or more robots propose the same target cell in one step; or
2. Two robots directly swap positions in one step.

Safety-layer overrides are tracked separately by the environment and are not
counted as collision events.

The existing environment does not persist per-task timestamps or collision
event objects. The Week 5 evaluator therefore accepts structured episode
traces and provides an adapter/helper for deriving collision events from
recorded positions.

## Test output

### Focused Week 5 tests

Command:

```powershell
python -m pytest tests\test_evaluation_metrics.py -q
```

Output:

```text
.....                                                                    [100%]
5 passed in 0.08s
```

All five new Week 5 evaluation tests passed.

### Full existing test suite

Command:

```powershell
python -m pytest -q
```

The full suite could not complete in the system Python environment because
dependencies were missing during test collection:

```text
ModuleNotFoundError: No module named 'mlflow'
ModuleNotFoundError: No module named 'gymnasium'
ModuleNotFoundError: No module named 'torch'
```

Summary:

```text
8 errors during collection
```

These errors came from existing infrastructure, Gymnasium, and QMIX tests.
They were dependency/setup errors, not failures in the Week 5 evaluation
tests.

## Recommended virtual-environment test commands

From the `week5-work` directory on Windows PowerShell:

```powershell
py -3.13 -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install gymnasium

python -m pytest -q
```

The project requirements already include MLflow, PyTorch, NumPy, pandas,
pytest, and PyYAML. `gymnasium` is not currently listed in
`requirements.txt`, so it is installed separately by the command above.

To run only the new evaluation tests after activating the environment:

```powershell
python -m pytest tests\test_evaluation_metrics.py -q
```

## Data status and limitations

The repository audit found existing Rule-Based result data, but it did not
provide complete real evaluation data for every combination of:

- QMIX, IQL, and Rule-Based;
- N=5, N=10, and N=20;
- completion-time traces;
- collision-event traces.

The test data in `test_evaluation_metrics.py` is synthetic and is used only
to verify metric behavior. It must not be presented as project performance
results.

Before producing final comparison graphs, the team should connect the
evaluator to real evaluation traces and confirm that each row preserves the
algorithm, fleet size, episode, task timing, and collision information.

## Handoff notes

- The evaluator is contained in `src/evaluation/`.
- The copied environment and previous-week files remain unchanged.
- The source audit clone is separate from this Week 5 working copy.
- The original repository audit clone had a clean Git status after the work.
- Generated graphs should only be produced after real completion-time data is
  available.
