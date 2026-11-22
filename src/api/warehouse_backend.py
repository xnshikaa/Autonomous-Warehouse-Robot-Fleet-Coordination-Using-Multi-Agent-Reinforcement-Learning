"""JSON telemetry and optional FastAPI WebSocket bridge for the visualizer."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from random import Random
from typing import Any, Callable
from uuid import uuid4

from src.ai.environment import WarehouseEnvironment
from src.ai.live_navigation import WarehouseNavigator
from src.ai.single_robot_mission import MISSION_SEED, queue_mission
from src.evaluation.metrics import infer_robot_robot_collisions
from src.evaluation.trace_generator import _iql_provider
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.qmix_controller import QMIXController
from src.warehouse.task_generation import generate_task_assignments

try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover - optional backend dependency
    WebSocket = Any

    class WebSocketDisconnect(Exception):
        pass


logger = logging.getLogger(__name__)
ALGORITHMS = ("QMIX", "IQL", "Rule-Based")
FLEET_SIZES = (5, 10, 20)
REFERENCE_SEEDS = {"QMIX": 310011, "IQL": 310001, "Rule-Based": 310011}


class CommandValidationError(ValueError):
    """A client command was malformed or is not supported."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _parse_command(message: str) -> dict[str, Any]:
    """Parse and validate one command without allowing bad input to escape."""
    try:
        command = json.loads(message)
    except (json.JSONDecodeError, TypeError) as error:
        raise CommandValidationError("invalid_json", "Message must contain valid JSON.") from error
    if not isinstance(command, dict):
        raise CommandValidationError("invalid_command", "Command must be a JSON object.")

    command_type = command.get("type")
    if not isinstance(command_type, str) or not command_type:
        raise CommandValidationError("missing_type", "Command requires a non-empty string 'type'.")

    if command_type not in {"reset", "load_reference", "pause", "resume", "start", "set_speed", "set_fleet_size", "set_algorithm"}:
        raise CommandValidationError("unsupported_command", f"Unsupported command type: {command_type}.")

    if command_type == "set_speed":
        speed = command.get("speed")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise CommandValidationError("invalid_speed", "'speed' must be a finite positive number.")
        try:
            speed = float(speed)
        except (OverflowError, ValueError) as error:
            raise CommandValidationError("invalid_speed", "'speed' must be a finite positive number.") from error
        if not math.isfinite(speed) or speed <= 0:
            raise CommandValidationError("invalid_speed", "'speed' must be a finite positive number.")
        command["speed"] = speed

    if command_type == "set_fleet_size":
        size = command.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CommandValidationError("invalid_fleet_size", "'size' must be a positive integer.")
        if "checkpoint_path" in command:
            raise CommandValidationError("invalid_checkpoint_source", "Checkpoints are configured on the server.")

    if command_type == "set_algorithm" and command.get("algorithm") not in ALGORITHMS:
        raise CommandValidationError("invalid_algorithm", "Choose QMIX, IQL, or Rule-Based.")
    if "checkpoint_path" in command:
        raise CommandValidationError("invalid_checkpoint_source", "Checkpoints are configured on the server.")
    return command


def _process_command_message(
    message: str,
    environment: WarehouseEnvironment,
    qmix_controller: QMIXController | None,
    policy_getter: Callable[[], dict[str, object]],
    interval_seconds: float,
    simulation_delay: float,
    speed_multiplier: float,
    simulation_paused: bool,
    *,
    configure: Callable[[int, str], None] | None = None,
    load_reference: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], float, float, bool]:
    """Apply one client message and return a response plus updated loop state."""
    try:
        command = _parse_command(message)
    except CommandValidationError as error:
        return ({
            "type": "command_error",
            "error": {"code": error.code, "message": str(error)},
        }, simulation_delay, speed_multiplier, simulation_paused)

    try:
        command_type = command["type"]
        if command_type == "reset":
            environment.reset()
            simulation_paused = configure is not None
        elif command_type == "load_reference":
            if load_reference is None:
                raise ValueError("Reference missions are unavailable.")
            load_reference()
            simulation_paused = True
        elif command_type == "set_fleet_size":
            new_size = command["size"]
            if configure is not None:
                configure(new_size, "Rule-Based" if new_size == 1 else str(policy_getter()["source"]))
            else:
                environment.get_active_layout().validate(required_robots=new_size)
                if qmix_controller is not None:
                    qmix_controller.reconfigure_fleet(new_size, allow_untrained=False)
                environment.set_num_robots(new_size)
        elif command_type == "set_algorithm":
            if configure is None:
                raise ValueError("Algorithm switching is unavailable.")
            configure(environment.num_robots, command["algorithm"])
        elif command_type == "pause":
            simulation_paused = True
        elif command_type in {"start", "resume"}:
            simulation_paused = False
        elif command_type == "set_speed":
            speed_multiplier = float(command["speed"])
            simulation_delay = max(0.01, float(interval_seconds) / speed_multiplier)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return ({
            "type": "command_error",
            "error": {"code": "invalid_command_value", "message": str(error)},
        }, simulation_delay, speed_multiplier, simulation_paused)

    payload = build_state_payload(
        environment, policy_getter(), simulation_delay, speed_multiplier, simulation_paused
    )
    return payload, simulation_delay, speed_multiplier, simulation_paused


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
            # Other robots and reservations are transient. If they seal every
            # route, plan against fixed hazards and let WarehouseEnvironment's
            # simultaneous collision resolver accept or block the first step.
            # Action 0 is UP, not a no-op, so it is never a safe substitute for
            # an unavailable dynamic route.
            path = _planned_path(environment, robot, static_blocked)
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
        return environment._task_target(robot, task)
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
    speed_multiplier: float = 1.0,
    simulation_paused: bool = False,
    *,
    trajectories: dict[int, deque[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    """Build a JSON snapshot matching the existing React state contract."""
    policy = policy_telemetry or {
        "source": "QMIX",
        "checkpointLoaded": False,
        "trained": False,
        "checkpointPath": None,
    }
    policy = {**policy, "trainingProgress": policy.get("trainingProgress")}
    algorithm = policy["source"]
    learned = algorithm in {"QMIX", "IQL"}
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
        path = _planned_path(environment, robot) if trajectories is None else trajectories[robot.robot_id]
        target = (_robot_goal(environment, robot) if trajectories is None
                  else environment._task_target(robot, task) if task else None)
        status = "IDLE"
        if robot.action_overridden:
            status = "WARNING"
        elif robot.carrying_item:
            status = "PICKUP" if task and robot.position == task.pickup_position else "DELIVERING"
        elif robot.last_executed_action is not None:
            status = "MOVING"
        robots.append({
            "id": f"R{robot.robot_id + 1:02d}", "position": list(robot.position),
            "targetPosition": list(target) if target is not None else None, "rotation": None,
            "status": status,
            "taskId": f"T{task.task_id:02d}" if task is not None else None,
            "hasCargo": robot.carrying_item, "battery": None, "path": [list(cell) for cell in path],
            "pathSource": "BFS_ESTIMATE" if trajectories is None else "POLICY_TRAJECTORY",
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
        "algorithm": algorithm,
        "controllerType": "RL_BACKEND" if learned else "RULE_BASED",
        "policy": policy,
        "config": {"gridWidth": layout.width, "gridHeight": layout.height, "cellSize": 1,
                   "layoutId": layout.layout_id,
                   "updateIntervalSeconds": max(0.01, float(update_interval_seconds)),
                   "speedMultiplier": float(speed_multiplier)},
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
        "safetyOverrideActive": any(event["overridden"] for event in events), "isDemoMode": algorithm == "DEMO",
        "simulationPaused": bool(simulation_paused),
        "isConnectedToBackend": True,
    }


def create_app(
    environment: WarehouseEnvironment | None = None,
    interval_seconds: float = 0.1,
    policy_source: str = "auto",
    checkpoint_path: str | None = None,
):
    """Create the FastAPI app used by the visualizer.

    Auto mode starts with a configured QMIX policy or the rule-based controller.
    Explicit learned-policy mode requires a compatible checkpoint at startup.
    """
    from fastapi import FastAPI

    requested = {"auto": "auto", "qmix": "QMIX", "iql": "IQL",
                 "rule-based": "Rule-Based", "demo": "Rule-Based"}.get(policy_source.lower().strip())
    if requested is None:
        raise ValueError("Choose auto, qmix, iql, or rule-based.")
    env = environment or WarehouseEnvironment(conservative_motion=True)
    env.conservative_motion = True
    env.reset()
    adapter = MARLEnvironmentAdapter(environment=env)
    @asynccontextmanager
    async def lifespan(_app):
        clock = asyncio.create_task(run_clock())
        try:
            yield
        finally:
            clock.cancel()
            with suppress(asyncio.CancelledError):
                await clock

    app = FastAPI(title="Warehouse Environment Backend", lifespan=lifespan)
    stream_id = uuid4().hex
    initial_size = env.num_robots
    initial_qmix = checkpoint_path or os.environ.get("QMIX_CHECKPOINT")
    active_policy = {}
    action_provider = None
    episode = None
    travelled = {}
    delivery_steps = {}
    conflict_events = 0
    physical_contacts = 0
    recent_collisions = []
    finished = False
    navigator: WarehouseNavigator | None = None
    pickup_steps = {}
    successful_moves = {}
    mission_seed = MISSION_SEED - 1
    reference_seed: int | None = None
    reference_episode: int | None = None

    def configure(size: int, algorithm: str) -> None:
        nonlocal active_policy, action_provider, navigator
        if size not in (1, *FLEET_SIZES):
            raise ValueError("Choose a fleet size of 1, 5, 10, or 20.")
        if size == 1 and algorithm != "Rule-Based":
            raise ValueError("The single-robot mission uses the Rule-Based navigator.")
        env.get_active_layout().validate(required_robots=size)
        candidate_navigator = None
        if algorithm == "QMIX":
            path = (initial_qmix if size == initial_size else None) or os.environ.get(f"QMIX_CHECKPOINT_N{size}")
            if not path:
                raise ValueError(f"No QMIX checkpoint configured for {size} robots.")
            controller = QMIXController(adapter)
            controller.reconfigure_fleet(size, checkpoint_path=path, allow_untrained=False)
            candidate_provider = controller.select_actions
            telemetry = controller.telemetry
        elif algorithm == "IQL":
            path = os.environ.get(f"IQL_CHECKPOINT_N{size}")
            if not path:
                raise ValueError(f"No IQL checkpoint configured for {size} robots.")
            directory = Path(path)
            metadata = json.loads((directory / "iql_manifest.json").read_text(encoding="utf-8"))
            if (metadata.get("algorithm"), metadata.get("num_robots"), metadata.get("observation_dim"),
                    metadata.get("action_dim"), metadata.get("include_static_map"),
                    metadata.get("include_path_costs")) != ("IQL", size, 456, 4, True, True):
                raise ValueError(f"IQL checkpoint is incompatible with the {size}-robot warehouse.")
            try:
                iql = _iql_provider(directory, size)
            except Exception as error:
                raise ValueError(f"IQL checkpoint could not be loaded: {error}") from error
            candidate_provider = lambda: iql(env)
            telemetry = {"source": algorithm, "checkpointLoaded": True, "trained": True,
                         "checkpointPath": str(directory), "checkpointId": directory.name}
        else:
            candidate_navigator = WarehouseNavigator(env)
            candidate_provider = candidate_navigator.get_actions
            telemetry = {"source": algorithm, "checkpointLoaded": False, "trained": False,
                         "checkpointPath": None}
        if active_policy:
            env.set_num_robots(size)
        active_policy = {**telemetry, "activeFleetSize": size}
        action_provider = candidate_provider
        navigator = candidate_navigator
        app.state.policy_source = algorithm

    startup_error = None
    if requested == "auto":
        try:
            configure(initial_size, "QMIX")
        except (OSError, ValueError) as error:
            startup_error = str(error)
            configure(initial_size, "Rule-Based")
    else:
        configure(initial_size, requested)

    simulation_paused = True
    simulation_delay = max(0.01, float(interval_seconds))
    speed_multiplier = 1.0

    def current_policy() -> dict[str, object]:
        return active_policy

    def load_reference_mission() -> None:
        nonlocal reference_seed, reference_episode
        if env.num_robots != 5 or environment is not None or env.get_active_layout().layout_id != "standard":
            raise ValueError("Reference missions require the standard five-robot fleet.")
        env.reset()
        reference_seed = REFERENCE_SEEDS[str(active_policy["source"])]
        reference_episode = env.episode

    def refresh_episode() -> None:
        nonlocal episode, travelled, delivery_steps, conflict_events, physical_contacts, recent_collisions, finished
        nonlocal pickup_steps, successful_moves, mission_seed
        if episode != env.episode:
            episode = env.episode
            if env.num_robots == 1 and not env.tasks:
                mission_seed += 1
                queue_mission(env, mission_seed)
            elif environment is None:
                seed = reference_seed if reference_episode == env.episode else 300_000 + env.episode
                tasks = generate_task_assignments(env, Random(seed))
                for task in tasks:
                    env.add_task(task.task_id, task.pickup_position, task.delivery_position, task.assigned_robot)
            travelled = {robot.robot_id: deque([robot.position], maxlen=32) for robot in env.robots}
            delivery_steps = {}
            pickup_steps = {}
            successful_moves = {}
            if navigator is not None:
                navigator.routes.clear()
                navigator.waiting.clear()
            conflict_events = 0
            physical_contacts = 0
            recent_collisions = []
            finished = False

    def tick() -> None:
        nonlocal conflict_events, physical_contacts, recent_collisions, finished
        refresh_episode()
        previous = {robot.robot_id: robot.position for robot in env.robots}
        _, _, finished, _ = adapter.step(action_provider())
        positions = {robot.robot_id: robot.position for robot in env.robots}
        step = env.episode_manager.current_step
        for robot_id, position in positions.items():
            if position != previous[robot_id]:
                travelled[robot_id].append(position)
                successful_moves[robot_id] = {"from": previous[robot_id], "to": position, "step": step}
        for task in env.tasks:
            if task.assigned_robot is not None and (task.completed or env.robots[task.assigned_robot].carrying_item):
                pickup_steps.setdefault(task.task_id, step)
            if task.completed:
                delivery_steps.setdefault(task.task_id, step)
        conflict_events += sum(event.reason == "robot_collision" for event in env.get_safety_events())
        contacts = infer_robot_robot_collisions(previous, positions)
        physical_contacts += len(contacts)
        recent_collisions = [
            {"id": f"{episode}-{step}-{index}", "type": "ROBOT_ROBOT_CONFLICT",
             "robots": [f"R{robot + 1:02d}" for robot in contact["robots"]],
             "position": positions[contact["robots"][0]], "timestamp": step}
            for index, contact in enumerate(contacts)
        ]

    def snapshot() -> dict[str, Any]:
        refresh_episode()
        payload = build_state_payload(env, current_policy(), simulation_delay, speed_multiplier,
                                      simulation_paused, trajectories=travelled)
        payload["simulationFinished"] = finished
        payload["streamId"] = stream_id
        payload["startupWarning"] = startup_error
        payload["recentCollisions"] = recent_collisions
        if env.num_robots == 1:
            payload["missionSeed"] = mission_seed
        if reference_episode == env.episode:
            payload["referenceSeed"] = reference_seed
        for task, record in zip(env.tasks, payload["tasks"], strict=True):
            record["pickupTimestep"] = pickup_steps.get(task.task_id)
            record["completedTimestep"] = delivery_steps.get(task.task_id)
        for robot, record in zip(env.robots, payload["robots"], strict=True):
            record["lastSuccessfulMove"] = successful_moves.get(robot.robot_id)
            record["blockedReason"] = robot.blocked_reason
            history = list(travelled[robot.robot_id])[-4:]
            record["oscillating"] = len(history) == 4 and history[0] == history[2] and history[1] == history[3]
            if navigator is not None and env._get_robot_task(robot.robot_id) is not None:
                route = navigator.routes.get(robot.robot_id, [])
                record["plannedPath"] = route[route.index(robot.position):] if robot.position in route else []
                record["replanning"] = robot.robot_id in navigator.waiting
            if finished and env._get_robot_task(robot.robot_id) is None:
                record["status"] = "IDLE"
        metrics = payload["metrics"]
        step = env.episode_manager.current_step
        metrics.update({
            "completionRate": len(delivery_steps) / len(env.tasks) if env.tasks else None,
            "averageDeliveryTimeSteps": sum(delivery_steps.values()) / len(delivery_steps) if delivery_steps else None,
            "throughputPer100Steps": len(delivery_steps) * 100 / step if step else None,
            "safetyConflictEvents": conflict_events,
            "physicalCollisions": physical_contacts,
            "totalCollisions": physical_contacts,
            "collisionCount": physical_contacts,
            "idleRobots": sum(robot.last_executed_action is None or
                              (finished and env._get_robot_task(robot.robot_id) is None) for robot in env.robots),
            "blockedMoves": env.safety_overrides,
            "pickups": len(pickup_steps),
        })
        return payload

    async def run_clock() -> None:
        nonlocal simulation_paused
        while True:
            await asyncio.sleep(simulation_delay)
            if not simulation_paused:
                tick()
                if finished:
                    simulation_paused = True

    @app.get("/health")
    async def health():
        return {"ok": True, "layout": env.get_active_layout().layout_id}

    @app.get("/readiness")
    async def readiness():
        return {"ok": True, "policy": current_policy()}

    @app.get("/state")
    async def state():
        return snapshot()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal simulation_paused, simulation_delay, speed_multiplier
        await websocket.accept()
        try:
            while True:
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=simulation_delay
                    )
                    response, simulation_delay, speed_multiplier, simulation_paused = _process_command_message(
                        raw_message,
                        env,
                        None,
                        current_policy,
                        interval_seconds,
                        simulation_delay,
                        speed_multiplier,
                        simulation_paused,
                        configure=configure,
                        load_reference=load_reference_mission,
                    )
                    if response["type"] == "state_update":
                        refresh_episode()
                        if finished:
                            simulation_paused = True
                        response = snapshot()
                    await websocket.send_json(response)
                except asyncio.TimeoutError:
                    await websocket.send_json(snapshot())
        except WebSocketDisconnect:
            pass

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
