"""Data-driven warehouse layout library for the Week 4 environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

Cell = tuple[int, int]


class LayoutValidationError(ValueError):
    """Raised when a warehouse layout cannot be used safely."""


def _cell(value: Sequence[int], field: str) -> Cell:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LayoutValidationError(f"{field} must be a two-item coordinate.")
    if len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise LayoutValidationError(f"{field} must contain two integer coordinates.")
    return int(value[0]), int(value[1])


def _cells(values, field: str) -> frozenset[Cell]:
    try:
        return frozenset(_cell(value, f"{field}[{index}]") for index, value in enumerate(values or ()))
    except TypeError as error:
        raise LayoutValidationError(f"{field} must be a collection of coordinates.") from error


@dataclass(frozen=True)
class LayoutTask:
    task_id: int
    pickup_position: Cell
    delivery_position: Cell
    assigned_robot: int | None = None


@dataclass(frozen=True)
class WarehouseLayout:
    layout_id: str
    name: str
    width: int
    height: int
    obstacles: frozenset[Cell]
    robot_starts: tuple[Cell, ...]
    task_locations: tuple[LayoutTask, ...]
    human_corridors: frozenset[Cell]
    walkable_cells: frozenset[Cell] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise LayoutValidationError("layout_id must be a non-empty string.")
        if not isinstance(self.name, str) or not self.name:
            raise LayoutValidationError("name must be a non-empty string.")
        if (isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0
                or isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0):
            raise LayoutValidationError("width and height must be positive integers.")
        obstacles = frozenset(_cell(c, "obstacles") for c in self.obstacles)
        corridors = frozenset(_cell(c, "human_corridors") for c in self.human_corridors)
        starts = tuple(_cell(c, "robot_starts") for c in self.robot_starts)
        if len(starts) != len(set(starts)):
            raise LayoutValidationError("robot_starts must contain unique cells.")
        walkable = (frozenset((x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in obstacles)
                    if self.walkable_cells is None else frozenset(_cell(c, "walkable_cells") for c in self.walkable_cells))
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "human_corridors", corridors)
        object.__setattr__(self, "robot_starts", starts)
        object.__setattr__(self, "walkable_cells", walkable)
        self.validate()

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def boundaries(self) -> dict[str, int]:
        return {"min_x": 0, "max_x": self.width - 1, "min_y": 0, "max_y": self.height - 1}

    def is_within_bounds(self, cell: Sequence[int]) -> bool:
        x, y = _cell(cell, "cell")
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, cell: Sequence[int]) -> bool:
        return _cell(cell, "cell") in self.walkable_cells

    def validate(self, *, required_robots: int | None = None) -> None:
        for field, values in {"obstacles": self.obstacles, "walkable_cells": self.walkable_cells,
                              "robot_starts": self.robot_starts, "human_corridors": self.human_corridors}.items():
            for cell in values:
                if not self.is_within_bounds(cell):
                    raise LayoutValidationError(f"{field} contains {cell}, outside {self.dimensions}.")
        if self.obstacles & self.human_corridors:
            raise LayoutValidationError("obstacles and human_corridors cannot overlap.")
        if self.obstacles & set(self.walkable_cells):
            raise LayoutValidationError("obstacles must not be included in walkable_cells.")
        for start in self.robot_starts:
            if not self.is_walkable(start) or start in self.human_corridors:
                raise LayoutValidationError(f"robot start {start} is not a valid walkable start.")
        for task in self.task_locations:
            if not isinstance(task, LayoutTask):
                raise LayoutValidationError("task_locations must contain LayoutTask values.")
            for label, cell in (("pickup", task.pickup_position), ("delivery", task.delivery_position)):
                if not self.is_within_bounds(cell) or not self.is_walkable(cell) or cell in self.human_corridors:
                    raise LayoutValidationError(f"task {task.task_id} {label} {cell} is invalid.")
        if required_robots is not None and required_robots > len(self.robot_starts):
            raise LayoutValidationError(f"layout provides {len(self.robot_starts)} starts but {required_robots} robots were requested.")

    def to_dict(self) -> dict[str, Any]:
        return {"layout_id": self.layout_id, "name": self.name,
                "dimensions": {"width": self.width, "height": self.height},
                "boundaries": self.boundaries,
                "walkable_cells": [list(c) for c in sorted(self.walkable_cells)],
                "obstacles": [list(c) for c in sorted(self.obstacles)],
                "robot_starts": [list(c) for c in self.robot_starts],
                "tasks": [{"task_id": t.task_id, "pickup_position": list(t.pickup_position),
                            "delivery_position": list(t.delivery_position), "assigned_robot": t.assigned_robot}
                           for t in self.task_locations],
                "human_corridors": {"cells": [list(c) for c in sorted(self.human_corridors)]}}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WarehouseLayout":
        if not isinstance(data, Mapping) or not isinstance(data.get("dimensions"), Mapping):
            raise LayoutValidationError("layout data and dimensions must be mappings.")
        tasks = tuple(LayoutTask(int(item["task_id"]), _cell(item["pickup_position"], "task.pickup_position"),
                                 _cell(item["delivery_position"], "task.delivery_position"), item.get("assigned_robot"))
                      for item in data.get("tasks", data.get("task_locations", ())))
        corridor_data = data.get("human_corridors", {})
        corridor_cells = corridor_data.get("cells", ()) if isinstance(corridor_data, Mapping) else corridor_data
        dimensions = data["dimensions"]
        return cls(str(data["layout_id"]), str(data.get("name", data["layout_id"])), int(dimensions["width"]),
                   int(dimensions["height"]), _cells(data.get("obstacles", ()), "obstacles"),
                   tuple(_cell(c, "robot_starts") for c in data.get("robot_starts", ())), tasks,
                   _cells(corridor_cells, "human_corridors"),
                   _cells(data["walkable_cells"], "walkable_cells") if "walkable_cells" in data else None)


class LayoutLibrary:
    def __init__(self, layouts: Sequence[WarehouseLayout] = ()) -> None:
        self._layouts: dict[str, WarehouseLayout] = {}
        for layout in layouts:
            self.register(layout)

    def register(self, layout: WarehouseLayout) -> None:
        if not isinstance(layout, WarehouseLayout):
            raise TypeError("layout must be a WarehouseLayout.")
        if layout.layout_id in self._layouts:
            raise LayoutValidationError(f"layout_id {layout.layout_id!r} is already registered.")
        layout.validate()
        self._layouts[layout.layout_id] = layout

    def get(self, layout_id: str) -> WarehouseLayout:
        if layout_id not in self._layouts:
            raise KeyError(f"Unknown layout {layout_id!r}; available layouts: {', '.join(self.ids())}")
        return self._layouts[layout_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._layouts))

    def all(self) -> tuple[WarehouseLayout, ...]:
        return tuple(self._layouts[i] for i in self.ids())


def get_default_layout_library() -> LayoutLibrary:
    standard = WarehouseLayout("standard", "Standard Aisle Warehouse", 20, 20,
        frozenset((x, y) for x in (1, 2, 5, 6, 11, 12, 16, 17) for y in range(2, 8)),
        tuple([(0, y) for y in range(8)] + [(4, y) for y in range(8)] + [(9, y) for y in range(4)]),
        (LayoutTask(1, (3, 2), (0, 19)), LayoutTask(2, (7, 3), (4, 19)), LayoutTask(3, (13, 4), (9, 19)), LayoutTask(4, (18, 5), (14, 19))),
        frozenset((x, 9) for x in range(20)))
    cross_dock = WarehouseLayout("cross_dock", "Cross-Dock Warehouse", 20, 20,
        frozenset((x, y) for x in (2, 3, 6, 7, 14, 15, 17, 18) for y in range(3, 9)),
        tuple([(0, y) for y in range(8)] + [(4, y) for y in range(8)] + [(8, y) for y in range(4)]),
        (LayoutTask(1, (1, 3), (0, 19)), LayoutTask(2, (5, 4), (4, 19)), LayoutTask(3, (9, 5), (8, 19)), LayoutTask(4, (13, 6), (12, 19))),
        frozenset((10, y) for y in range(20)))
    return LayoutLibrary((standard, cross_dock))
