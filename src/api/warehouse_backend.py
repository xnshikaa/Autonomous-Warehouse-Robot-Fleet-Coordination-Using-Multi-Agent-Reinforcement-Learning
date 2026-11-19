"""JSON telemetry and optional FastAPI WebSocket bridge for the visualizer."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from typing import Any

from src.ai.environment import WarehouseEnvironment
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.qmix_controller import QMIXController

try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover - optional backend dependency
    WebSocket = Any

    class WebSocketDisconnect(Exception):
        pass


def _next_policy_actions(environment: WarehouseEnvironment) -> dict[int, int]:
    """Choose deterministic demo actions for the optional visual fallback.

    The default live bridge uses ``QMIXController`` instead. Movement itself
    remains exclusively in ``WarehouseEnvironment.step`` in both modes.
    """
    actions: dict[int, int] = {}
    static_blocked = set(environment.get_active_layout().obstacles) | set(environment.get_active_layout().human_corridors)
    reserved: set[tuple[int, int]] = set()

    for robot in environment.robots:
        occupied = {other.position for other in environment.robots if other.robot_id != robot.robot_id}
        path = _planned_path(environment, robot, static_blocked | occupied | reserved)
        if len(path) < 2:
            actions[robot.robot_id] = 0
            continue
        dx = path[1][0] - robot.position[0]
        dy = path[1][1] - robot.position[1]
        action = next((action for action, delta in environment.ACTION_DELTAS.items() if delta == (dx, dy)), 0)
        actions[robot.robot_id] = action
        target = environment.get_target_cell(robot, action)
        if target is not None and target not in static_blocked:
            reserved.add(target)
    return actions


def _robot_goal(environment: WarehouseEnvironment, robot) -> tuple[int, int]:
    task = environment._get_robot_task(robot.robot_id)
    if task is not None:
        return task.delivery_position if robot.position == task.pickup_position else task.pickup_position
    waypoints = tuple((x, min(environment.warehouse_height - 1, 8))
                      for x in (0, 4, 9, 14, 19) if x < environment.warehouse_width)
    return waypoints[(robot.robot_id + environment.episode_manager.current_step // 30) % len(waypoints)]


def _planned_path(environment: WarehouseEnvironment, robot, extra_blocked=()) -> list[tuple[int, int]]:
    """Return a visual route using the same action deltas as the environment."""
    goal = _robot_goal(environment, robot)
    blocked = set(environment.get_active_layout().obstacles) | set(environment.get_active_layout().human_corridors)
    blocked.update(extra_blocked)
    blocked.discard(robot.position)
    queue = deque([robot.position])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {robot.position: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in environment.ACTION_DELTAS.values():
            candidate = current[0] + dx, current[1] + dy
            if (candidate not in parent and environment.is_within_bounds(candidate)
                    and candidate not in blocked):
                parent[candidate] = current
                queue.append(candidate)
    if goal not in parent:
        return [robot.position]
    path = [goal]
    while path[-1] != robot.position:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def build_state_payload(
    environment: WarehouseEnvironment,
    policy_telemetry: dict[str, object] | None = None,
    update_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    """Build a JSON snapshot matching the existing React state contract."""
    policy = policy_telemetry or {
        "source": "QMIX",
        "checkpointLoaded": False,
        "trained": False,
        "checkpointPath": None,
    }
    policy = {**policy, "trainingProgress": policy.get("trainingProgress")}
    is_qmix = policy.get("source") == "QMIX"
    layout = environment.get_active_layout()
    tasks = [{
        "id": f"T{task.task_id:02d}", "pickupPos": list(task.pickup_position),
        "deliveryPos": list(task.delivery_position),
        "assignedRobotId": f"R{task.assigned_robot + 1:02d}" if task.assigned_robot is not None else None,
        "status": (
            "COMPLETED" if task.completed
            else "IN_PROGRESS" if task.assigned_robot is not None
            and environment.robots[task.assigned_robot].carrying_item
            else "ASSIGNED" if task.assigned_robot is not None
            else "UNASSIGNED"
        ),
    } for task in environment.tasks]
    robots = []
    for robot in environment.robots:
        task = environment._get_robot_task(robot.robot_id)
        path = _planned_path(environment, robot)
        target = _robot_goal(environment, robot)
        robots.append({
            "id": f"R{robot.robot_id + 1:02d}", "position": list(robot.position),
            "targetPosition": list(target), "rotation": None,
            "status": "WARNING" if robot.action_overridden else ("MOVING" if robot.last_executed_action is not None else "IDLE"),
            "taskId": f"T{task.task_id:02d}" if task is not None else None,
            "hasCargo": robot.carrying_item, "battery": None, "path": [list(cell) for cell in path],
            "pathSource": "BFS_ESTIMATE",
            "safetyAlert": robot.action_overridden,
            "lastRequestedAction": robot.last_requested_action,
            "lastExecutedAction": "NO_OP" if robot.action_overridden else robot.last_executed_action,
            "actionOverridden": robot.action_overridden,
        })
    corridor = sorted(layout.human_corridors)
    safety_zones = []
    if corridor:
        xs, ys = [c[0] for c in corridor], [c[1] for c in corridor]
        safety_zones.append({"id": "HUMAN_CORRIDOR", "name": "Human-worker corridor",
                             "bounds": {"minX": min(xs), "maxX": max(xs), "minY": min(ys), "maxY": max(ys)},
                             "activeWarning": bool(environment.get_safety_events())})
    events = [{
        "robotId": f"R{event.robot_id + 1:02d}",
        "currentCell": list(event.current_cell),
        "proposedAction": event.proposed_action,
        "targetCell": None if event.target_cell is None else list(event.target_cell),
        "finalAction": "NO_OP" if event.final_action is None else event.final_action,
        "overridden": event.overridden,
        "reason": event.reason,
        "episode": event.episode,
        "timestep": event.timestep,
    } for event in environment.get_safety_events()]
    step = environment.episode_manager.current_step
    return {
        "type": "state_update", "timestep": step, "episode": environment.episode,
        "algorithm": "QMIX" if is_qmix else "DEMO",
        "controllerType": "RL_BACKEND" if is_qmix else "DEMO_CONTROLLER",
        "policy": policy,
        "config": {"gridWidth": layout.width, "gridHeight": layout.height, "cellSize": 1,
                   "layoutId": layout.layout_id,
                   "updateIntervalSeconds": max(0.01, float(update_interval_seconds))},
        "layout": layout.to_dict(), "robots": robots,
        "shelves": [{"id": f"S{i:02d}", "gridPos": list(c), "levels": 1, "aisleId": "LAYOUT"}
                    for i, c in enumerate(sorted(layout.obstacles), 1)],
        "tasks": tasks, "safetyZones": safety_zones, "recentCollisions": [], "safetyEvents": events,
        "metrics": {"activeRobots": len(robots), "totalRobots": len(robots),
                     "activeTasks": sum(not t.completed for t in environment.tasks),
                     "completedTasks": sum(t.completed for t in environment.tasks),
                     "throughputPerHour": None, "averageDeliveryTimeSec": None,
                     "collisionAttempts": None, "conflictingProposals": None,
                     "safetyOverrides": environment.safety_overrides, "physicalCollisions": 0,
                     "totalCollisions": 0, "collisionCount": 0,
                     "blockedConflictGroups": environment.blocked_conflict_groups,
                     "blockedActionCount": environment.safety_overrides,
                     "overrideCount": environment.safety_overrides,
                     "movementSteps": environment.movement_steps,
                     "idleRobotSteps": environment.idle_robot_steps,
                     "timestep": step, "episode": environment.episode,
                     "reward": (sum(components["total"] for components in environment.last_reward_components.values())
                                if environment.last_reward_components else None),
                     "fps": None},
        "safetyOverrideActive": any(event["overridden"] for event in events), "isDemoMode": not is_qmix,
        "isConnectedToBackend": True,
    }


def create_app(
    environment: WarehouseEnvironment | None = None,
    interval_seconds: float = 0.1,
    policy_source: str = "qmix",
    checkpoint_path: str | None = None,
):
    """Create the FastAPI app used by the visualizer.

    ``policy_source="qmix"`` is the default and routes live actions through
    ``MARLEnvironmentAdapter`` and ``QMIXController``. Use
    ``policy_source="demo"`` only for a deterministic visual fallback.
    """
    try:
        from fastapi import FastAPI
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("FastAPI is required for the WebSocket bridge; install requirements.txt.") from error
    policy_source = policy_source.lower().strip()
    if policy_source not in {"qmix", "demo"}:
        raise ValueError("policy_source must be 'qmix' or 'demo'")

    env = environment or WarehouseEnvironment(load_layout_tasks=True)
    env.reset()
    adapter = MARLEnvironmentAdapter(environment=env)
    checkpoint_path = checkpoint_path or os.environ.get("QMIX_CHECKPOINT")
    qmix_controller = (
        QMIXController(
            adapter,
            checkpoint_path=checkpoint_path,
            epsilon=float(os.environ.get("QMIX_EPSILON", "0")),
        )
        if policy_source == "qmix"
        else None
    )
    app = FastAPI(title="Warehouse Environment Backend")
    simulation_paused = False
    simulation_delay = max(0.01, float(interval_seconds))
    simulation_driver: WebSocket | None = None

    def current_policy() -> dict[str, object]:
        if qmix_controller is not None:
            return qmix_controller.telemetry
        return {
            "source": "DEMO",
            "checkpointLoaded": False,
            "trained": False,
            "checkpointPath": None,
        }

    @app.get("/health")
    async def health():
        return {"ok": True, "layout": env.get_active_layout().layout_id}

    @app.get("/state")
    async def state():
        return build_state_payload(env, current_policy(), simulation_delay)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal simulation_paused, simulation_delay, simulation_driver
        await websocket.accept()
        try:
            while True:
                try:
                    command = json.loads(await asyncio.wait_for(
                        websocket.receive_text(), timeout=simulation_delay
                    ))
                    if command.get("type") == "reset":
                        env.reset()
                        simulation_paused = False
                    elif command.get("type") == "set_fleet_size":
                        env.set_num_robots(int(command["size"]))
                        if qmix_controller is not None:
                            qmix_controller.reset_for_fleet()
                    elif command.get("type") == "pause":
                        simulation_paused = True
                    elif command.get("type") == "resume":
                        simulation_paused = False
                    elif command.get("type") == "set_speed":
                        simulation_delay = max(
                            0.01,
                            float(interval_seconds) / max(0.1, float(command.get("speed", 1))),
                        )
                    await websocket.send_json(build_state_payload(env, current_policy(), simulation_delay))
                except asyncio.TimeoutError:
                    if simulation_driver is None:
                        simulation_driver = websocket
                    if not simulation_paused and simulation_driver is websocket:
                        actions = (
                            qmix_controller.select_actions()
                            if qmix_controller is not None
                            else _next_policy_actions(env)
                        )
                        _, _, terminated, _ = adapter.step(actions)
                        if terminated:
                            env.reset()
                    await websocket.send_json(build_state_payload(env, current_policy(), simulation_delay))
        except WebSocketDisconnect:
            pass
        finally:
            if simulation_driver is websocket:
                simulation_driver = None

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
