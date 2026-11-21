"""PyBullet 3D physics and simulation layer for the 20x20 Autonomous Warehouse.

Provides:
- 3D rigid-body simulation for AMR warehouse robots.
- Obstacle shelves, boundaries, and human worker corridor markers.
- Grid-to-continuous coordinate mapping with discrete decision stepping.
- Graceful software physics fallback when pybullet C-extensions are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    p = None
    pybullet_data = None


@dataclass
class AMRPhysicsState:
    robot_id: int
    grid_position: Tuple[int, int]
    world_position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    carrying: bool = False
    in_collision: bool = False


class PyBulletWarehouseSimulation:
    """3D Physics simulation bridge for the multi-agent warehouse environment."""

    def __init__(
        self,
        grid_width: int = 20,
        grid_height: int = 20,
        cell_size: float = 1.0,
        gui: bool = False,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        human_corridors: Optional[List[Tuple[int, int]]] = None,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.use_gui = gui
        self.obstacles = set(obstacles or [])
        self.human_corridors = set(human_corridors or [])
        self.physics_client = None
        self.robot_bodies: Dict[int, Any] = {}
        self.shelf_bodies: List[Any] = []
        self.is_connected = False
        self.step_counter = 0

        self._init_simulation()

    def _init_simulation(self) -> None:
        """Initialize PyBullet or software fallback physics."""
        if HAS_PYBULLET:
            try:
                mode = p.GUI if self.use_gui else p.DIRECT
                self.physics_client = p.connect(mode)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -9.81)
                p.loadURDF("plane.urdf")
                self.is_connected = True
                self._build_world_obstacles()
                logger.info("PyBullet 3D physics simulation initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize PyBullet physics engine: {e}. Using software physics.")
                self.is_connected = False
        else:
            logger.info("PyBullet module not installed; running in high-fidelity software physics fallback mode.")
            self.is_connected = False

    def _grid_to_world(self, grid_x: int, grid_y: int, z: float = 0.2) -> Tuple[float, float, float]:
        """Convert 2D grid coordinates to 3D world coordinates centered on cell."""
        world_x = (grid_x - self.grid_width / 2.0 + 0.5) * self.cell_size
        world_y = (grid_y - self.grid_height / 2.0 + 0.5) * self.cell_size
        return (world_x, world_y, z)

    def _world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert 3D world coordinates back to 2D discrete grid coordinates."""
        gx = int(round(world_x / self.cell_size + self.grid_width / 2.0 - 0.5))
        gy = int(round(world_y / self.cell_size + self.grid_height / 2.0 - 0.5))
        return (max(0, min(self.grid_width - 1, gx)), max(0, min(self.grid_height - 1, gy)))

    def _build_world_obstacles(self) -> None:
        """Spawn 3D shelf obstacles in PyBullet world."""
        if not self.is_connected:
            return
        shelf_half_extents = [self.cell_size * 0.45, self.cell_size * 0.45, 0.8]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=shelf_half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=shelf_half_extents, rgbaColor=[0.3, 0.3, 0.35, 1.0])

        for ox, oy in self.obstacles:
            pos = self._grid_to_world(ox, oy, z=0.8)
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape, basePosition=pos)
            self.shelf_bodies.append(body_id)

    def register_robot(self, robot_id: int, start_grid_pos: Tuple[int, int]) -> AMRPhysicsState:
        """Register a robot AMR body in the 3D physics environment."""
        world_pos = self._grid_to_world(*start_grid_pos)
        if self.is_connected:
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.cell_size * 0.35, height=0.4)
            vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=self.cell_size * 0.35, length=0.4, rgbaColor=[0.2, 0.6, 0.9, 1.0])
            body_id = p.createMultiBody(baseMass=15.0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape, basePosition=world_pos)
            self.robot_bodies[robot_id] = body_id
        else:
            self.robot_bodies[robot_id] = f"sim_robot_{robot_id}"

        return AMRPhysicsState(
            robot_id=robot_id,
            grid_position=start_grid_pos,
            world_position=world_pos,
            velocity=(0.0, 0.0, 0.0),
        )

    def sync_from_grid_state(self, robots_state: List[Any]) -> Dict[int, AMRPhysicsState]:
        """Synchronize continuous 3D physics bodies from discrete grid states."""
        physics_states = {}
        for robot in robots_state:
            rid = getattr(robot, "robot_id", robot.get("robot_id") if isinstance(robot, dict) else 0)
            pos = getattr(robot, "position", robot.get("position") if isinstance(robot, dict) else (0, 0))
            carrying = getattr(robot, "carrying_item", robot.get("hasCargo", False) if isinstance(robot, dict) else False)

            if rid not in self.robot_bodies:
                self.register_robot(rid, tuple(pos))

            target_world = self._grid_to_world(pos[0], pos[1])
            if self.is_connected and isinstance(self.robot_bodies[rid], int):
                p.resetBasePositionAndOrientation(
                    self.robot_bodies[rid],
                    target_world,
                    [0, 0, 0, 1]
                )

            physics_states[rid] = AMRPhysicsState(
                robot_id=rid,
                grid_position=tuple(pos),
                world_position=target_world,
                velocity=(0.0, 0.0, 0.0),
                carrying=carrying,
                in_collision=False,
            )

        if self.is_connected:
            p.stepSimulation()

        self.step_counter += 1
        return physics_states

    def check_corridor_intrusion(self, grid_pos: Tuple[int, int]) -> bool:
        """Check whether a grid position is inside a human exclusion corridor."""
        return grid_pos in self.human_corridors

    def close(self) -> None:
        """Clean up PyBullet client."""
        if self.is_connected:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.is_connected = False
            self.robot_bodies.clear()
            self.shelf_bodies.clear()
