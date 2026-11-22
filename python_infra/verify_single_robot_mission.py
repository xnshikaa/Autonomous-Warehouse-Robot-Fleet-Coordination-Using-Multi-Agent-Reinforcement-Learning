"""Audit seeded single-robot missions under both navigation authorities."""
import argparse
import json
import logging
from pathlib import Path

from src.ai.environment import WarehouseEnvironment
from src.ai.live_navigation import WarehouseNavigator
from src.ai.single_robot_mission import MISSION_SEED, queue_mission
from src.evaluation.metrics import infer_robot_robot_collisions
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.qmix_controller import QMIXController


def trial(seed, mission, environment, actions):
    mission.reset()
    queue_mission(mission, seed)
    if environment is not mission:
        environment.reset()
        task = mission.tasks[0]
        environment.add_task(1, task.pickup_position, task.delivery_position, 0)
    robot = environment.robots[0]
    task = environment.tasks[0]
    positions = [robot.position]
    pickup = None
    contacts = obstacles = corridor = blocked = oscillations = 0
    occupied_corridors = environment.get_active_layout().human_corridors
    for step in range(1, environment.max_steps + 1):
        before = {item.robot_id: item.position for item in environment.robots}
        moves = actions()
        _, _, terminal, _ = environment.step(moves)
        after = {item.robot_id: item.position for item in environment.robots}
        contacts += len(infer_robot_robot_collisions(before, after))
        obstacles += sum(not environment.is_walkable(cell) for cell in after.values())
        corridor += sum(cell in occupied_corridors for cell in after.values())
        blocked += sum(event.overridden for event in environment.get_safety_events())
        positions.append(robot.position)
        if len(positions) >= 4:
            a, b, c, d = positions[-4:]
            oscillations += a == c and b == d and a != b
        if pickup is None and robot.carrying_item:
            assert robot.position == task.pickup_position
            pickup = step
        if terminal:
            break
    if task.completed:
        assert pickup is not None and robot.position == task.delivery_position and not robot.carrying_item
    return {
        "seed": seed, "pickup_step": pickup, "delivery_step": step if task.completed else None,
        "physical_collisions": contacts, "shelf_or_wall_contacts": obstacles,
        "corridor_intrusions": corridor, "blocked_moves": blocked,
        "two_cell_cycles": oscillations, "stuck": not task.completed,
        "idle": task.completed and environment._get_robot_task(0) is None and not robot.carrying_item,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    logger = logging.getLogger("mission_validation")
    logger.setLevel(logging.ERROR)
    mission = WarehouseEnvironment(num_robots=1, conservative_motion=True, logger=logger)
    incumbent = WarehouseEnvironment(num_robots=5, conservative_motion=True, logger=logger)
    checkpoint = Path("artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt")
    qmix = QMIXController(MARLEnvironmentAdapter(environment=incumbent), checkpoint_path=checkpoint)

    def qmix_with_parked_fleet():
        return {0: qmix.select_actions()[0], **dict.fromkeys(range(1, 5))}

    runs = {}
    for name, environment, actions in (
        ("QMIX N=5 with four robots held", incumbent, qmix_with_parked_fleet),
        ("Rule-Based reservation navigator", mission, WarehouseNavigator(mission).get_actions),
    ):
        rows = [trial(seed, mission, environment, actions) for seed in range(MISSION_SEED, MISSION_SEED + args.trials)]
        delivery_steps = [row["delivery_step"] for row in rows if row["delivery_step"] is not None]
        summary = {
            "pickups": sum(row["pickup_step"] is not None for row in rows),
            "deliveries": len(delivery_steps),
            "completion_rate": len(delivery_steps) / len(rows),
            "average_delivery_steps": sum(delivery_steps) / len(delivery_steps) if delivery_steps else None,
            **{key: sum(row[key] for row in rows) for key in (
                "physical_collisions", "shelf_or_wall_contacts", "corridor_intrusions",
                "blocked_moves", "two_cell_cycles", "stuck", "idle",
            )},
        }
        print(name, summary, flush=True)
        runs[name] = {"summary": summary, "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump({"layout": mission.get_active_layout().layout_id, "runs": runs}, file, indent=2)


if __name__ == "__main__":
    main()
