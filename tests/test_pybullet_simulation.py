"""Unit tests for the PyBullet warehouse simulation integration layer."""

import pytest
from src.simulation.pybullet_warehouse import PyBulletWarehouseSimulation, AMRPhysicsState


def test_pybullet_warehouse_initialization():
    sim = PyBulletWarehouseSimulation(
        grid_width=20,
        grid_height=20,
        cell_size=1.0,
        obstacles=[(1, 2), (1, 3)],
        human_corridors=[(10, y) for y in range(20)],
    )

    assert sim.grid_width == 20
    assert sim.grid_height == 20
    assert (1, 2) in sim.obstacles
    assert sim.check_corridor_intrusion((10, 5)) is True
    assert sim.check_corridor_intrusion((5, 5)) is False
    sim.close()


def test_coordinate_conversion():
    sim = PyBulletWarehouseSimulation(grid_width=20, grid_height=20, cell_size=1.0)
    world_pos = sim._grid_to_world(0, 0, z=0.2)
    grid_pos = sim._world_to_grid(world_pos[0], world_pos[1])

    assert grid_pos == (0, 0)

    world_mid = sim._grid_to_world(10, 10, z=0.2)
    grid_mid = sim._world_to_grid(world_mid[0], world_mid[1])
    assert grid_mid == (10, 10)
    sim.close()


def test_robot_registration_and_state_sync():
    sim = PyBulletWarehouseSimulation(grid_width=20, grid_height=20)
    state = sim.register_robot(0, (2, 3))
    assert isinstance(state, AMRPhysicsState)
    assert state.robot_id == 0
    assert state.grid_position == (2, 3)

    mock_robots = [
        {"robot_id": 0, "position": (3, 3), "hasCargo": True},
        {"robot_id": 1, "position": (4, 4), "hasCargo": False},
    ]

    synced = sim.sync_from_grid_state(mock_robots)
    assert len(synced) == 2
    assert synced[0].grid_position == (3, 3)
    assert synced[0].carrying is True
    assert synced[1].grid_position == (4, 4)
    assert synced[1].carrying is False
    assert sim.step_counter == 1
    sim.close()
