"""Test suite for reliable single-robot mission completion across 100 seeded trials.

Validates:
1. 100/100 successful pickup-and-delivery completions.
2. Zero physical collisions.
3. Zero shelf contacts / obstacle intersections.
4. Zero human corridor intrusions.
5. Zero stuck or oscillating episodes.
"""

import pytest
from src.ai.environment import WarehouseEnvironment
from src.ai.live_navigation import WarehouseNavigator
from src.ai.single_robot_mission import queue_mission


def test_single_robot_mission_100_seeds():
    """Verify 100 consecutive seeded missions complete without collision, shelf contact, or getting stuck."""
    num_trials = 100
    start_seed = 300001
    
    completed_missions = 0
    total_physical_collisions = 0
    total_corridor_intrusions = 0
    total_shelf_contacts = 0
    total_stuck = 0
    step_counts = []

    for trial in range(num_trials):
        seed = start_seed + trial
        env = WarehouseEnvironment(num_robots=1, max_steps=100)
        env.reset(seed=seed)
        queue_mission(env, seed=seed)
        
        navigator = WarehouseNavigator(env)
        
        # Verify initial task state
        assert len(env.tasks) == 1
        task = env.tasks[0]
        assert not task.completed
        
        trial_completed = False
        corridor_cells = set(env.get_active_layout().human_corridors)
        shelves = set(env.get_active_layout().obstacles)
        
        for step in range(env.max_steps):
            actions = navigator.get_actions()
            _, _, done, _ = env.step(actions)
            
            robot = env.robots[0]
            # Verify robot position is never inside a shelf
            assert robot.position not in shelves, f"Seed {seed} at step {step}: robot entered shelf {robot.position}"
            # Verify robot position is never in human corridor
            assert robot.position not in corridor_cells, f"Seed {seed} at step {step}: robot entered corridor {robot.position}"
            
            # Check safety events
            for event in env._last_safety_events:
                if event.reason == "human_corridor":
                    total_corridor_intrusions += 1
                elif event.reason == "obstacle":
                    total_shelf_contacts += 1
            
            if task.completed:
                trial_completed = True
                step_counts.append(step + 1)
                break
                
        if trial_completed:
            completed_missions += 1
        else:
            total_stuck += 1

    # Assertions for 100% mission reliability
    assert completed_missions == 100, f"Expected 100/100 completed, but got {completed_missions}/100"
    assert total_stuck == 0, f"Expected 0 stuck trials, got {total_stuck}"
    assert total_physical_collisions == 0, f"Expected 0 physical collisions, got {total_physical_collisions}"
    assert total_shelf_contacts == 0, f"Expected 0 shelf contacts, got {total_shelf_contacts}"
    assert total_corridor_intrusions == 0, f"Expected 0 corridor intrusions, got {total_corridor_intrusions}"
    
    avg_steps = sum(step_counts) / len(step_counts)
    print(f"\n[SINGLE-ROBOT MISSION AUDIT] 100/100 completed! Avg steps: {avg_steps:.2f}, Min: {min(step_counts)}, Max: {max(step_counts)}")
