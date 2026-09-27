# Week 4 (Member 1)- Environment

This worktree adds a reusable layout library and a stable Python environment
for the warehouse simulation. The existing `src/ai`, `src/safety`, and
`src/marl` interfaces remain the integration boundary for QMIX/EPyMARL.

## Layout Library

`src/warehouse/layouts.py` defines one schema for every layout:

- `layout_id`, `name`, `dimensions`, and inclusive coordinate `boundaries`
- `walkable_cells` and `obstacles`
- `robot_starts`
- static pickup/drop-off `tasks`
- required `human_corridors.cells`

The built-in layouts are `standard` and `cross_dock`. They are both 20 x 20,
support 20 starts, and contain explicit corridor annotations. New layouts are
added as data through `WarehouseLayout.from_mapping()` and registered in a
`LayoutLibrary`; environment movement code is not duplicated.

## Environment APIs

`WarehouseEnvironment` now exposes:

- `get_active_layout()`, `get_layout_ids()`, and `set_layout(layout_id)`
- `get_dimensions()`, `is_within_bounds(cell)`, and `is_walkable(cell)`
- `get_robot_state(robot_id)`
- `get_target_cell(robot_or_id, action)` / `calculate_target_cell(...)`
- `get_safety_events()` and `get_safety_event_history()`

The target-cell API uses the repository's existing action semantics: `0=UP`,
`1=DOWN`, `2=LEFT`, `3=RIGHT`. Out-of-bounds targets return `None`; obstacle
checks are applied by the same movement implementation before state changes.

## Safety step hook and telemetry

`step()` performs the complete safety pass before applying any robot position.
It calls the existing Week 3 `SafetyLayer`, then blocks out-of-bounds and
obstacle targets, resolves simultaneous robot conflicts, and only then updates
robot state. `SafetyOverrideEvent` records robot ID, current cell, proposed
action, target cell, final action, override flag, and reason (`human_corridor`,
`out_of_bounds`, `obstacle`, or `robot_collision`).

Robot state retains the latest requested/executed action and block reason.
Reset clears positions, tasks, episode counters, safety counters, and event
telemetry. Layout switching is followed by `reset()`.

## JavaScript/Python connection

`src/api/warehouse_backend.py` serializes the environment into the existing
React `WarehouseState` shape and provides an optional FastAPI WebSocket at
`ws://127.0.0.1:8000/ws`, which matches `src/api/PythonWebSocketProvider.ts`.
Run it after installing dependencies with:

```text
python -m src.api.warehouse_backend
```

The bridge also supports the provider's `reset`, `pause`, `resume`, and
`set_speed` messages. It is telemetry/control glue only; QMIX action selection
and training remain in the existing MARL layer.

## Tests

Run the relevant suite with:

```text
pytest -q
```

`tests/test_week4_environment.py` covers schema validation, multiple layouts,
corridors, dimensions/bounds, active-layout and robot-state access, target
cells, blocked movement, pre-movement safety, telemetry, layout switching,
reset isolation, and JSON frontend payloads.

The WebSocket bridge requires the optional `fastapi` and `uvicorn` dependencies
listed in `requirements.txt`. EPyMARL itself is not vendored in this repository;
the existing QMIX-ready adapter remains the supported training integration.
