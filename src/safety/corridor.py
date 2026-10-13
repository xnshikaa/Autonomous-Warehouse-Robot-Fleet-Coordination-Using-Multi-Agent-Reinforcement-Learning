"""Human-worker corridor annotations and membership checks.

The module owns no warehouse layout.  Corridor cells are supplied by the
environment configuration or by an equivalent injected source.  This keeps
the safety rule reusable when Member 1's warehouse implementation is added.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


Cell = tuple[int, int]
WarehouseSize = tuple[int, int]


def _normalise_cell(cell: Sequence[int], *, field_name: str = "cell") -> Cell:
    """Validate and normalise a coordinate without applying bounds."""

    if isinstance(cell, (str, bytes)) or not isinstance(cell, Sequence):
        raise ValueError(f"{field_name} must be a two-item coordinate.")

    if len(cell) != 2:
        raise ValueError(f"{field_name} must contain exactly two values.")

    if any(isinstance(value, bool) or not isinstance(value, int) for value in cell):
        raise ValueError(f"{field_name} coordinates must be integers.")

    return int(cell[0]), int(cell[1])


def _validate_warehouse_size(warehouse_size: Sequence[int] | None) -> WarehouseSize | None:
    if warehouse_size is None:
        return None

    width, height = _normalise_cell(warehouse_size, field_name="warehouse_size")
    if width <= 0 or height <= 0:
        raise ValueError("warehouse_size values must be positive.")
    return width, height


class CorridorModule:
    """Represent configured human-worker corridor cells.

    A configured cell is validated once and stored in a set for efficient
    membership checks.  Invalid annotations fail fast because silently
    dropping a forbidden-zone definition could make the safety boundary
    incomplete.  ``enabled=False`` makes every membership check return false,
    while still validating the supplied configuration.
    """

    def __init__(
        self,
        cells: Iterable[Sequence[int]] = (),
        *,
        enabled: bool = True,
        warehouse_size: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")

        self.enabled = enabled
        self.warehouse_size = _validate_warehouse_size(warehouse_size)

        if cells is None:
            raise ValueError("cells must be an iterable of coordinates.")

        validated_cells: set[Cell] = set()
        for index, cell in enumerate(cells):
            normalised = _normalise_cell(cell, field_name=f"cells[{index}]")
            if normalised[0] < 0 or normalised[1] < 0:
                raise ValueError(f"cells[{index}] must not contain negative coordinates.")

            if self.warehouse_size is not None:
                width, height = self.warehouse_size
                if normalised[0] >= width or normalised[1] >= height:
                    raise ValueError(
                        f"cells[{index}]={normalised} is outside the warehouse bounds "
                        f"{self.warehouse_size}."
                    )

            validated_cells.add(normalised)

        self._cells = frozenset(validated_cells)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CorridorModule":
        """Build a module from the project environment configuration.

        The method accepts either the full ``environment.json`` mapping or
        the nested ``human_corridors`` mapping.  The warehouse dimensions are
        used when the full configuration is supplied.
        """

        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping.")

        corridor_config = config.get("human_corridors", config)
        if not isinstance(corridor_config, Mapping):
            raise ValueError("human_corridors configuration must be a mapping.")

        warehouse_size = None
        warehouse_config = config.get("warehouse")
        if warehouse_config is not None:
            if not isinstance(warehouse_config, Mapping):
                raise ValueError("warehouse configuration must be a mapping.")
            try:
                warehouse_size = (
                    warehouse_config["width"],
                    warehouse_config["height"],
                )
            except KeyError as error:
                raise ValueError("warehouse must define width and height.") from error

        return cls(
            cells=corridor_config.get("cells", ()),
            enabled=corridor_config.get("enabled", True),
            warehouse_size=warehouse_size,
        )

    def is_corridor(self, cell: Sequence[int]) -> bool:
        """Return whether ``cell`` is an enabled forbidden corridor cell."""

        normalised = _normalise_cell(cell)
        return self.enabled and normalised in self._cells

    def get_corridor_cells(self) -> frozenset[Cell]:
        """Return an immutable snapshot of the configured corridor cells."""

        return self._cells
