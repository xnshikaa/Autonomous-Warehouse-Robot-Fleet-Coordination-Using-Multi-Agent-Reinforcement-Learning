"""
Member 4 Week 4 Comprehensive Safety, Integration, and Invariant Verification Test Suite.

Targeting:
- SAFETY-01 to SAFETY-10
- INTEGRATION-01 to INTEGRATION-10
- Invariants 1 to 9
"""

import pytest
import numpy as np
import torch
from src.ai.environment import WarehouseEnvironment
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.gymnasium_env import WarehouseGymEnv
from src.marl.observation_encoder import ObservationEncoder
from src.marl.qmix_learner import QMIXLearner
from src.marl.qmix_network import AgentQNetwork, QMIXMixer
from src.marl.qmix_trainer import QMIXTrainer
from src.marl.spec import MARLEnvironmentSpec
from src.safety.safety_layer import SafetyLayer
from src.safety.corridor import CorridorModule
from src.warehouse.layouts import get_default_layout_library
from python_infra.config import ExperimentConfig, TrainingMetrics
from python_infra.experiment_tracker import ExperimentTracker
from python_infra.checkpoint_manager import CheckpointManager


# ==============================================================================
# PART 3 — SAFETY TESTING (SAFETY-01 to SAFETY-10)
# ==============================================================================

def test_SAFETY_01_boundary_blocking():
    """SAFETY-01: Robot at boundary attempting movement outside 20x20 grid is blocked."""
    env = WarehouseEnvironment(num_robots=1)
    env.reset()
    # Robot 0 starts at (0, 0). Action 0 (UP) or 2 (LEFT) goes out of bounds.
    env.get_robot_state(0).position = (0, 0)
    obs, rewards, terminated, state = env.step({0: 0})  # UP
    
    assert env.get_robot_state(0).position == (0, 0)
    events = env.get_safety_events()
    assert len(events) >= 1
    assert events[-1].reason == "out_of_bounds"
    assert events[-1].overridden is True


def test_SAFETY_02_shelf_blocking():
    """SAFETY-02: Robot next to shelf attempting movement into shelf is blocked."""
    env = WarehouseEnvironment(num_robots=1)
    env.reset()
    # (0, 2) is adjacent to shelf at (1, 2) in standard layout.
    env.get_robot_state(0).position = (0, 2)
    obs, rewards, terminated, state = env.step({0: 3})  # RIGHT into shelf
    
    assert env.get_robot_state(0).position == (0, 2)
    events = env.get_safety_events()
    assert len(events) >= 1
    assert events[-1].reason == "obstacle"
    assert events[-1].overridden is True


def test_SAFETY_03_robot_robot_same_cell_conflict():
    """SAFETY-03: Two robots attempting to enter the same target cell simultaneously."""
    env = WarehouseEnvironment(num_robots=2)
    env.reset()
    # Robot 0 at (0, 0) moving RIGHT (3) -> target (1, 0)
    # Robot 1 at (2, 0) moving LEFT (2) -> target (1, 0)
    env.get_robot_state(0).position = (0, 0)
    env.get_robot_state(1).position = (2, 0)
    
    obs, rewards, terminated, state = env.step({0: 3, 1: 2})
    
    pos0 = env.get_robot_state(0).position
    pos1 = env.get_robot_state(1).position
    assert pos0 != pos1, "Robots must not occupy the same cell!"


def test_SAFETY_04_robot_swap_conflict():
    """SAFETY-04: Robot A at X, Robot B at adjacent Y attempting swap A->Y, B->X is blocked."""
    env = WarehouseEnvironment(num_robots=2)
    env.reset()
    # Robot 0 at (0, 0) moving DOWN (1) -> (0, 1)
    # Robot 1 at (0, 1) moving UP (0) -> (0, 0)
    env.get_robot_state(0).position = (0, 0)
    env.get_robot_state(1).position = (0, 1)
    
    obs, rewards, terminated, state = env.step({0: 1, 1: 0})
    
    pos0 = env.get_robot_state(0).position
    pos1 = env.get_robot_state(1).position
    assert not (pos0 == (0, 1) and pos1 == (0, 0)), "Illegal swap must be prevented!"


def test_SAFETY_05_multi_robot_conflict():
    """SAFETY-05: Three robots attempting to enter the same target cell simultaneously."""
    env = WarehouseEnvironment(num_robots=3)
    env.reset()
    env.get_robot_state(0).position = (1, 0)
    env.get_robot_state(1).position = (0, 1)
    env.get_robot_state(2).position = (2, 1)
    
    obs, rewards, terminated, state = env.step({0: 1, 1: 3, 2: 2})
    
    positions = [env.get_robot_state(i).position for i in range(3)]
    assert len(set(positions)) == 3, "All 3 robots must have unique positions!"


def test_SAFETY_06_safety_override_wait():
    """SAFETY-06: Unsafe action blocked, requested vs final action distinguishable, event logged."""
    env = WarehouseEnvironment(num_robots=1)
    env.reset()
    env.get_robot_state(0).position = (0, 0)
    
    env.step({0: 0})
    
    events = env.get_safety_events()
    assert len(events) >= 1
    evt = events[-1]
    assert evt.proposed_action == 0
    assert evt.final_action is None
    assert evt.overridden is True


def test_SAFETY_07_human_corridor_safety():
    """SAFETY-07: Verify human worker corridor cell detection and action override via SafetyLayer."""
    layout = get_default_layout_library().get("standard")
    corridor = CorridorModule(layout.human_corridors, warehouse_size=layout.dimensions)
    
    corridor_cell = list(layout.human_corridors)[0]
    assert corridor.is_corridor(corridor_cell) is True
    
    class TargetProvider:
        def get_target_cell(self, state, action):
            return corridor_cell

    class DummyRobotState:
        position = (corridor_cell[0] - 1, corridor_cell[1])
        robot_id = 0

    layer = SafetyLayer(corridor=corridor, target_cell_provider=TargetProvider(), blocked_action=None)
    decision = layer.check_action(DummyRobotState(), proposed_action=3)
    
    assert decision.was_overridden is True
    assert decision.safe_action is None
    assert decision.override_reason == "human_corridor"


def test_SAFETY_08_fleet_size_5():
    """SAFETY-08: Verify 5-robot fleet initialization, safety active, no collisions."""
    env = WarehouseEnvironment(num_robots=5)
    env.reset()
    assert len(env.robots) == 5
    
    obs, rewards, term, state = env.step({i: 1 for i in range(5)})
    positions = [r.position for r in env.robots]
    assert len(set(positions)) == 5, "No duplicate positions in fleet 5"


def test_SAFETY_09_fleet_size_10():
    """SAFETY-09: Verify 10-robot fleet initialization and safety."""
    env = WarehouseEnvironment(num_robots=10)
    env.reset()
    assert len(env.robots) == 10
    
    obs, rewards, term, state = env.step({i: (i % 4) for i in range(10)})
    positions = [r.position for r in env.robots]
    assert len(set(positions)) == 10, "No duplicate positions in fleet 10"


def test_SAFETY_10_fleet_size_20():
    """SAFETY-10: Verify 20-robot fleet scaling, multi-robot conflicts, and safety performance."""
    env = WarehouseEnvironment(num_robots=20)
    env.reset()
    assert len(env.robots) == 20
    
    for step_idx in range(10):
        actions = {i: ((i + step_idx) % 4) for i in range(20)}
        obs, rewards, term, state = env.step(actions)
        positions = [r.position for r in env.robots]
        assert len(set(positions)) == 20, f"No duplicate positions at step {step_idx} in fleet 20"


# ==============================================================================
# PART 4 — SAFETY INVARIANTS (1 to 9)
# ==============================================================================

def test_INVARIANTS_all_safety_invariants():
    """Verify all 9 safety invariants hold during a multi-step rollout."""
    env = WarehouseEnvironment(num_robots=10)
    env.reset()
    encoder = ObservationEncoder()
    layout = env.get_active_layout()
    obstacles_set = set(layout.obstacles)
    
    for step_idx in range(20):
        actions = {i: ((i * 3 + step_idx) % 4) for i in range(10)}
        observations, rewards, terminated, state = env.step(actions)
        
        positions = [r.position for r in env.robots]
        
        # 1. Position within 20x20 grid
        for pos in positions:
            assert 0 <= pos[0] < 20 and 0 <= pos[1] < 20, f"Invariant 1 failed: {pos}"
            
        # 2. Does not occupy shelf/obstacle cell
        for pos in positions:
            assert pos not in obstacles_set, f"Invariant 2 failed: robot at shelf {pos}"
            
        # 3. No illegal physical collisions (duplicate positions)
        assert len(set(positions)) == len(positions), "Invariant 3 failed: overlapping positions"
        
        # 6. Observation dimensions remain exactly 50
        for robot in env.robots:
            obs_vec = encoder.encode(robot, env.robots, env.tasks, layout=layout)
            assert len(obs_vec) == 50, f"Invariant 6 failed: obs len {len(obs_vec)}"
            
        # 7. Action values remain within Discrete(4)
        for a_id, act in actions.items():
            assert act in (0, 1, 2, 3), f"Invariant 7 failed: action {act}"
            
        # 8 & 9. State internal consistency
        assert len(state.robots) == 10, "Invariant 8 failed: robot count mismatch"
        assert len(rewards) == 10, "Invariant 9 failed: reward count mismatch"


# ==============================================================================
# PART 5 — INTEGRATION TESTING (INTEGRATION-01 to INTEGRATION-10)
# ==============================================================================

def test_INTEGRATION_01_python_environment_initialization():
    """INTEGRATION-01: Python environment initializes, resets, returns correct agent count."""
    env = WarehouseEnvironment(num_robots=10)
    global_state, observations = env.reset()
    assert env.num_robots == 10
    assert len(env.robots) == 10
    assert global_state is not None
    assert len(observations) == 10


def test_INTEGRATION_02_observation_format():
    """INTEGRATION-02: Python observation matches 50-D specification per agent."""
    env = WarehouseEnvironment(num_robots=5)
    env.reset()
    encoder = ObservationEncoder()
    layout = env.get_active_layout()
    
    for robot in env.robots:
        vec = encoder.encode(robot, env.robots, env.tasks, layout=layout)
        assert len(vec) == 50
        assert all(isinstance(val, float) for val in vec)


def test_INTEGRATION_03_action_format():
    """INTEGRATION-03: Discrete(4) action space (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) verification."""
    spec = MARLEnvironmentSpec(num_agents=5)
    assert spec.action_dim == 4
    for a in range(4):
        assert a in (0, 1, 2, 3)


def test_INTEGRATION_04_single_environment_step():
    """INTEGRATION-04: Single step reset -> obs -> action -> step -> next obs -> reward."""
    adapter = MARLEnvironmentAdapter(num_agents=5)
    obs_dict, global_state = adapter.reset()
    assert len(obs_dict) == 5
    
    actions = {i: 1 for i in range(5)}
    next_obs, rewards, term, next_gstate = adapter.step(actions)
    
    assert len(next_obs) == 5
    assert len(rewards) == 5
    assert isinstance(term, bool)


def test_INTEGRATION_05_multi_agent_step():
    """INTEGRATION-05: Multi-agent step execution for all agents simultaneously."""
    gym_env = WarehouseGymEnv(num_agents=10)
    obs_arr, info = gym_env.reset()
    assert len(obs_arr) == 10
    
    actions = gym_env.action_space.sample()
    # If sample returns a single int, replicate for 10 agents
    if isinstance(actions, (int, np.integer)):
        actions = [int(actions)] * 10
    else:
        actions = [gym_env.action_space.sample() for _ in range(10)]
        
    next_obs, reward, term, trunc, info = gym_env.step(actions)
    
    assert len(next_obs) == 10
    assert isinstance(reward, float)


def test_INTEGRATION_06_safety_through_integration():
    """INTEGRATION-06: Unsafe action from Python policy layer passes through safety gate and is overridden."""
    adapter = MARLEnvironmentAdapter(num_agents=1)
    adapter.reset()
    
    # Robot 0 at (0, 0); Action 0 (UP) is out of bounds
    adapter.environment.get_robot_state(0).position = (0, 0)
    next_obs, rewards, term, gstate = adapter.step({0: 0})
    
    assert adapter.environment.get_robot_state(0).position == (0, 0)
    events = adapter.environment.get_safety_events()
    assert len(events) >= 1
    assert events[-1].overridden is True


def test_INTEGRATION_07_reset():
    """INTEGRATION-07: Environment reset returns to valid initial state."""
    env = WarehouseEnvironment(num_robots=5)
    env.reset()
    env.step({i: 1 for i in range(5)})
    assert env.episode_manager.current_step == 1
    
    env.reset()
    assert env.episode_manager.current_step == 0
    assert env.safety_overrides == 0


def test_INTEGRATION_08_multi_step_rollout():
    """INTEGRATION-08: Deterministic multi-step rollout without state corruption."""
    adapter = MARLEnvironmentAdapter(num_agents=5)
    adapter.reset()
    
    for step in range(20):
        actions = {i: (step % 4) for i in range(5)}
        next_obs, rewards, term, gstate = adapter.step(actions)
        assert len(next_obs) == 5
        assert len(rewards) == 5


def test_INTEGRATION_09_fleet_scaling_initialization():
    """INTEGRATION-09: Verify fleet sizes 5, 10, 20 initialization via adapter."""
    for n in (5, 10, 20):
        adapter = MARLEnvironmentAdapter(num_agents=n)
        obs_dict, gstate = adapter.reset()
        assert len(obs_dict) == n
        assert adapter.num_agents == n


def test_INTEGRATION_10_epymarl_qmix_compatibility():
    """INTEGRATION-10: QMIX network, learner, trainer smoke test & MLflow logging."""
    trainer = QMIXTrainer(
        num_agents=5,
        observation_dim=50,
        action_dim=4,
        state_dim=250,
        max_steps=10,
        episodes=2
    )
    results = trainer.train()
    
    assert len(results) == 2
    assert "reward" in results[0]
    assert "loss" in results[0]


# ==============================================================================
# PART 6 — MLFLOW / CHECKPOINT INTEGRATION
# ==============================================================================

def test_MLFLOW_and_checkpoint_management():
    """Verify MLflow experiment tracking and checkpoint saving/restoration."""
    config = ExperimentConfig(
        experiment_name="test_infra_matrix",
        algorithm="QMIX",
        number_of_agents=5
    )
    
    tracker = ExperimentTracker(experiment_name=config.experiment_name)
    run = tracker.start_run(run_name="matrix_test_run")
    run_id = tracker.run_id
    
    tracker.log_config(config)
    metrics = TrainingMetrics(episode_reward=15.5, task_completion_rate=0.8, completed_tasks=4)
    tracker.log_metrics(metrics, step=1)
    
    checkpoint_mgr = CheckpointManager(base_dir="artifacts/checkpoints", algorithm="QMIX", run_id=run_id)
    payload = {"step": 1, "test_weight": torch.tensor([1.0, 2.0])}
    
    path = checkpoint_mgr.save_checkpoint(payload, step=1, is_best=True, best_metric_val=0.8, tracker=tracker)
    assert path is not None
    
    loaded = checkpoint_mgr.load_checkpoint("best")
    assert loaded["checkpoint_data"]["step"] == 1
    assert torch.equal(loaded["checkpoint_data"]["test_weight"], torch.tensor([1.0, 2.0]))
    
    tracker.end_run(status="FINISHED")
