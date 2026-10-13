"""Deterministic pre-action safety gate for robot movement actions.

The layer is deliberately independent of QMIX, Q-values, rewards, and
training state.  A caller supplies the current robot state, policy action,
corridor module, and a target-cell provider owned by the movement system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.safety.corridor import Cell, CorridorModule


class TargetCellProvider(Protocol):
    """Interface implemented by the warehouse movement implementation."""

    def get_target_cell(self, robot_state: Any, action: int) -> Cell | None:
        """Return the action target, or ``None`` when movement is unavailable."""


@dataclass(frozen=True)
class SafetyDecision:
    """Result of one independent safety check.

    ``safe_action=None`` means the proposed movement is blocked.  The current
    four-action project mapping has no verified no-op action, so the movement
    implementation must treat ``None`` as "do not move" unless a
    project-specific ``blocked_action`` is injected into ``SafetyLayer``.
    """

    safe_action: int | None
    was_overridden: bool
    robot_id: Any
    current_cell: Cell
    proposed_action: int
    target_cell: Cell | None
    final_action: int | None
    override_reason: str | None = None

    def __iter__(self):
        """Allow the compact ``safe_action, overridden = decision`` form."""

        yield self.safe_action
        yield self.was_overridden


@dataclass(frozen=True)
class SafetyOverrideEvent:
    """Serializable telemetry for one pre-movement safety decision."""

    robot_id: Any
    current_cell: Cell
    proposed_action: int
    target_cell: Cell | None
    final_action: int | None
    overridden: bool
    reason: str | None = None
    episode: int | None = None
    timestep: int | None = None

    @classmethod
    def from_decision(
        cls,
        decision: SafetyDecision,
        *,
        episode: int | None = None,
        timestep: int | None = None,
    ) -> "SafetyOverrideEvent":
        return cls(decision.robot_id, decision.current_cell, decision.proposed_action,
                   decision.target_cell, decision.final_action, decision.was_overridden,
                   decision.override_reason, episode, timestep)

    def to_dict(self) -> dict[str, Any]:
        return {"robot_id": self.robot_id, "current_cell": list(self.current_cell),
                "proposed_action": self.proposed_action,
                "target_cell": list(self.target_cell) if self.target_cell is not None else None,
                "final_action": self.final_action, "overridden": self.overridden,
                "reason": self.reason, "episode": self.episode,
                "timestep": self.timestep}


def _normalise_current_cell(position: Sequence[int]) -> Cell:
    if isinstance(position, (str, bytes)) or not isinstance(position, Sequence):
        raise ValueError("robot_state.position must be a two-item coordinate.")
    if len(position) != 2:
        raise ValueError("robot_state.position must contain exactly two values.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in position):
        raise ValueError("robot_state.position coordinates must be integers.")
    return int(position[0]), int(position[1])


class SafetyLayer:
    """Block actions whose calculated target is an enabled corridor cell.

    ``target_cell_provider`` is intentionally injected.  This class does not
    assume orientation, action vectors, boundary behavior, or a robot engine;
    those semantics belong to Member 1's eventual movement implementation.
    """

    def __init__(
        self,
        corridor: CorridorModule,
        target_cell_provider: TargetCellProvider
        | Callable[[Any, int], Cell | None],
        *,
        blocked_action: int | None = None,
        logger: Any | None = None,
    ) -> None:
        if not isinstance(corridor, CorridorModule):
            raise TypeError("corridor must be a CorridorModule.")
        if not callable(target_cell_provider) and not hasattr(
            target_cell_provider, "get_target_cell"
        ):
            raise TypeError(
                "target_cell_provider must be callable or implement get_target_cell."
            )

        self.corridor = corridor
        self.target_cell_provider = target_cell_provider
        self.blocked_action = blocked_action
        self.logger = logger

    def _calculate_target_cell(self, robot_state: Any, action: int) -> Cell | None:
        provider = self.target_cell_provider
        if hasattr(provider, "get_target_cell"):
            target_cell = provider.get_target_cell(robot_state, action)
        else:
            target_cell = provider(robot_state, action)

        if target_cell is None:
            return None

        if (
            isinstance(target_cell, (str, bytes))
            or not isinstance(target_cell, Sequence)
            or len(target_cell) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in target_cell
            )
        ):
            raise ValueError("target_cell_provider must return a two-item integer coordinate or None.")

        return int(target_cell[0]), int(target_cell[1])

    def check_action(self, robot_state: Any, proposed_action: int) -> SafetyDecision:
        """Check one policy action and return the safe action plus diagnostics."""

        if not isinstance(proposed_action, int) or isinstance(proposed_action, bool):
            raise ValueError("proposed_action must be an integer action ID.")

        if not hasattr(robot_state, "position"):
            raise ValueError("robot_state must expose a position attribute.")

        current_cell = _normalise_current_cell(robot_state.position)
        target_cell = self._calculate_target_cell(robot_state, proposed_action)
        robot_id = getattr(robot_state, "robot_id", None)

        if target_cell is not None and self.corridor.is_corridor(target_cell):
            reason = "human_corridor"
            decision = SafetyDecision(
                safe_action=self.blocked_action,
                was_overridden=True,
                robot_id=robot_id,
                current_cell=current_cell,
                proposed_action=proposed_action,
                target_cell=target_cell,
                final_action=self.blocked_action,
                override_reason=reason,
            )
            if self.logger is not None:
                self.logger.warning(
                    "Safety override | robot_id=%s | reason=%s | "
                    "current_cell=%s | target_cell=%s | proposed_action=%s",
                    robot_id,
                    reason,
                    current_cell,
                    target_cell,
                    proposed_action,
                )
            return decision

        return SafetyDecision(
            safe_action=proposed_action,
            was_overridden=False,
            robot_id=robot_id,
            current_cell=current_cell,
            proposed_action=proposed_action,
            target_cell=target_cell,
            final_action=proposed_action,
        )
