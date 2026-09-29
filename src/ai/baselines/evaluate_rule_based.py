"""
Rule-Based Baseline Evaluation

Week 4 - Member 3

Evaluates the deterministic rule-based baseline using
different fleet sizes and records initial performance results.
"""

import csv
import os

from src.ai.environment import WarehouseEnvironment
from src.ai.baselines.rule_based import RuleBasedController


def add_test_tasks(environment):
    """
    Add one deterministic task for each robot.

    All coordinates remain inside the 20 x 20 warehouse.
    """

    for robot in environment.robots:
        robot_id = robot.robot_id

        # Keep tasks within warehouse bounds.
        row = robot_id % environment.warehouse_height

        pickup_position = (5, row)
        delivery_position = (10, row)

        environment.add_task(
            task_id=robot_id,
            pickup_position=pickup_position,
            delivery_position=delivery_position,
            assigned_robot=robot_id,
        )


def evaluate_fleet(
    num_robots,
    max_steps=200,
):
    """
    Evaluate one fleet size using the rule-based controller.
    """

    environment = WarehouseEnvironment(
        num_robots=num_robots,
        max_steps=max_steps,
    )

    environment.reset()

    add_test_tasks(environment)

    controller = RuleBasedController(
        environment
    )

    total_reward = 0.0
    steps = 0
    terminated = False

    while not terminated:

        actions = controller.get_actions()

        (
            observations,
            rewards,
            terminated,
            global_state,
        ) = environment.step(actions)

        total_reward += sum(
            rewards.values()
        )

        steps += 1

        if steps >= max_steps:
            break

    completed_tasks = sum(
        1
        for task in environment.tasks
        if task.completed
    )

    total_tasks = len(
        environment.tasks
    )

    completion_rate = (
        completed_tasks / total_tasks * 100
        if total_tasks > 0
        else 0.0
    )

    average_reward = (
        total_reward / num_robots
        if num_robots > 0
        else 0.0
    )

    return {
        "robots": num_robots,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "completion_rate": round(
            completion_rate,
            2,
        ),
        "steps": steps,
        "total_reward": round(
            total_reward,
            2,
        ),
        "average_reward": round(
            average_reward,
            2,
        ),
        "safety_overrides":
            environment.safety_overrides,
    }


def save_results(results):
    """Save evaluation results to CSV."""

    os.makedirs(
        "results",
        exist_ok=True,
    )

    file_path = (
        "results/rule_based_results.csv"
    )

    fieldnames = [
        "robots",
        "completed_tasks",
        "total_tasks",
        "completion_rate",
        "steps",
        "total_reward",
        "average_reward",
        "safety_overrides",
    ]

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    return file_path


def main():

    fleet_sizes = [
        5,
        10,
        20,
    ]

    results = []

    print()
    print("=" * 86)
    print("RULE-BASED BASELINE - INITIAL RESULTS")
    print("=" * 86)

    for fleet_size in fleet_sizes:

        result = evaluate_fleet(
            num_robots=fleet_size,
        )

        results.append(result)

        print(
            f"Robots: {result['robots']:>2} | "
            f"Completed: "
            f"{result['completed_tasks']:>2}/"
            f"{result['total_tasks']:<2} | "
            f"Completion: "
            f"{result['completion_rate']:>6.2f}% | "
            f"Steps: {result['steps']:>3} | "
            f"Avg Reward: "
            f"{result['average_reward']:>8.2f} | "
            f"Safety Overrides: "
            f"{result['safety_overrides']}"
        )

    print("=" * 86)

    file_path = save_results(
        results
    )

    print()
    print(
        f"Results saved to: {file_path}"
    )


if __name__ == "__main__":
    main()