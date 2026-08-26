# Week 3 Member 4: Safety Module

This branch contains the Week 3 Member 4 safety deliverables for the
Autonomous Warehouse Robot Fleet Coordination project:

- `src/safety/corridor.py`: a configuration-driven Corridor Module.
- `src/safety/safety_layer.py`: a deterministic hard safety gate.
- `tests/test_corridor.py`: Corridor Module unit tests.
- `tests/test_safety_layer.py`: Safety Layer unit tests with a clearly marked
  test-only movement adapter.
- `config/environment.json`: the smallest schema extension needed for future corridor annotations (`human_corridors.cells`, currently empty).

## Corridor flow

```text
warehouse configuration / warehouse layout
                    |
            human corridor cells
                    |
             CorridorModule
                    |
              is_corridor(cell)
```

`CorridorModule.from_config` reads the existing `human_corridors.enabled`
setting and optional `cells` list. Coordinates are validated as integer,
non-negative, two-dimensional cells and, when warehouse dimensions are
available, must be within those bounds. Duplicate annotations are stored only
once. The production configuration intentionally contains no invented
corridor coordinates.

## Hard safety flow

```text
current robot state + proposed action
                    |
       injected target-cell provider
                    |
              corridor check
                    |
        safe action + override metadata
```

`SafetyLayer.check_action` returns a `SafetyDecision` containing at least
`safe_action` and `was_overridden`, plus robot ID, current cell, proposed
action, target cell, final action, and override reason. It can also be
unpacked as `(safe_action, was_overridden)`.

The layer does not own robot movement. The target-cell provider is injected so
Member 1's eventual movement engine can supply the real orientation,
coordinate transform, and boundary behavior. The test adapter uses the fixed
project action IDs `0=forward`, `1=backward`, `2=left`, `3=right` and a simple
axis-aligned transform only for unit testing; it is not a claim about the final
warehouse movement semantics.

The current action mapping has no verified no-op action. Therefore a corridor
intrusion returns `safe_action=None` by default, meaning the movement layer
must not execute the proposed action. A project-defined blocked/hold action
can be injected later with `blocked_action` once Member 1's movement contract
defines one.

## QMIX independence

The gate accepts a raw proposed action and never imports or consults QMIX,
Q-values, rewards, confidence, or training state. The intended integration is:

```text
QMIX policy -> proposed action -> SafetyLayer -> movement API
```

An override can be passed to the existing project logger through the optional
`logger` argument; the core logic remains independently testable.

## Current dependencies and limitations

The repository still lacks the completed warehouse/grid layout,
production robot movement API, and final coordinate/orientation semantics.
It also has no final corridor cell annotations. Supplying those external
inputs should replace the configuration data and test adapter, not require a
rewrite of the Corridor Module or Safety Layer.

This simulation safety gate is not ISO certification and does not replace
hardware emergency stops, sensors, or other required personnel-protection
systems in a physical deployment.
