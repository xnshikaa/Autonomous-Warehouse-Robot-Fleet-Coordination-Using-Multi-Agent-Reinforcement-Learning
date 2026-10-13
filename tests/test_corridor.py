import pytest

from src.safety.corridor import CorridorModule


def test_corridor_cell_is_detected():
    # Test fixture only: these are not project warehouse coordinates.
    corridor = CorridorModule(cells=[(5, 5)], warehouse_size=(20, 20))

    assert corridor.is_corridor((5, 5)) is True


def test_non_corridor_cell_is_not_detected():
    # Test fixture only.
    corridor = CorridorModule(cells=[(5, 5)], warehouse_size=(20, 20))

    assert corridor.is_corridor((5, 6)) is False


def test_multiple_corridor_cells_are_detected():
    # Test fixture only.
    corridor = CorridorModule(
        cells=[(5, 5), (5, 6), (5, 7)],
        warehouse_size=(20, 20),
    )

    assert all(corridor.is_corridor(cell) for cell in [(5, 5), (5, 6), (5, 7)])


def test_duplicate_annotations_are_deduplicated():
    # Test fixture only.
    corridor = CorridorModule(
        cells=[(5, 5), (5, 5), [5, 5]],
        warehouse_size=(20, 20),
    )

    assert corridor.get_corridor_cells() == frozenset({(5, 5)})


@pytest.mark.parametrize(
    "cells",
    [
        [(-1, 0)],
        [(20, 0)],
        [(0, 20)],
        [(1,)],
        [(1, "bad")],
    ],
)
def test_invalid_or_out_of_bounds_definitions_fail_fast(cells):
    with pytest.raises(ValueError):
        CorridorModule(cells=cells, warehouse_size=(20, 20))


def test_disabled_corridors_never_match():
    # Test fixture only; cells remain valid but the feature is disabled.
    corridor = CorridorModule(
        cells=[(5, 5)],
        enabled=False,
        warehouse_size=(20, 20),
    )

    assert corridor.is_corridor((5, 5)) is False


def test_full_environment_config_uses_enabled_flag_and_warehouse_bounds():
    config = {
        "warehouse": {"width": 20, "height": 20},
        "human_corridors": {"enabled": True, "cells": [[5, 5]]},
    }

    corridor = CorridorModule.from_config(config)

    assert corridor.is_corridor((5, 5)) is True
