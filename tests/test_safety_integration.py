from src.ai.environment import WarehouseEnvironment
from src.warehouse.layouts import get_default_layout_library


def _find_corridor_entry(layout):
    """Find a valid approach using only the active layout data."""

    for target in sorted(layout.human_corridors):
        for action, (dx, dy) in WarehouseEnvironment.ACTION_DELTAS.items():
            current = (target[0] - dx, target[1] - dy)
            if (
                layout.is_within_bounds(current)
                and layout.is_walkable(current)
                and current not in layout.human_corridors
            ):
                return current, action, target

    raise AssertionError(f"No valid corridor approach found for {layout.layout_id}.")


class RecordingLogger:
    """Small test logger that records the existing logger call."""

    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


def test_live_layout_corridor_is_blocked_before_robot_movement():
    for layout in get_default_layout_library().all():
        environment = WarehouseEnvironment(num_robots=1, layout_id=layout.layout_id)
        environment.reset()
        current, action, target = _find_corridor_entry(layout)
        environment.get_robot_state(0).position = current

        environment.step({0: action})

        event = environment.get_safety_events()[0]
        assert event.current_cell == current
        assert event.target_cell == target
        assert event.final_action is None
        assert event.overridden is True
        assert event.reason == "human_corridor"
        assert environment.get_robot_state(0).position == current


def test_live_safety_override_is_logged_with_runtime_metadata():
    logger = RecordingLogger()
    layout = get_default_layout_library().get("standard")
    environment = WarehouseEnvironment(
        num_robots=1,
        layout_id=layout.layout_id,
        logger=logger,
    )
    environment.reset()
    current, action, target = _find_corridor_entry(layout)
    environment.get_robot_state(0).position = current

    environment.step({0: action})

    event = environment.get_safety_events()[0]
    assert len(logger.messages) == 1
    message = logger.messages[0]
    assert "Safety override" in message
    assert f"robot_id={event.robot_id}" in message
    assert f"reason={event.reason}" in message
    assert f"target_cell={event.target_cell}" in message
    assert f"episode={event.episode}" in message
    assert f"timestep={event.timestep}" in message


def test_layout_switch_rebuilds_the_runtime_corridor_source():
    library = get_default_layout_library()
    environment = WarehouseEnvironment(num_robots=1, layout_id="standard")
    environment.reset()
    standard_cells = environment.get_active_layout().human_corridors

    environment.set_layout("cross_dock")
    environment.reset()

    assert environment.get_active_layout().layout_id == "cross_dock"
    assert environment.get_active_layout().human_corridors != standard_cells
    assert environment.corridor.get_corridor_cells() == library.get("cross_dock").human_corridors
