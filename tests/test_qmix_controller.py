from src.ai.environment import WarehouseEnvironment
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.qmix_controller import QMIXController


def test_qmix_controller_selects_valid_actions_for_shared_environment():
    environment = WarehouseEnvironment(num_robots=2, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=environment)
    adapter.reset()
    controller = QMIXController(adapter)

    actions = controller.select_actions()

    assert set(actions) == {0, 1}
    assert all(action in range(4) for action in actions.values())
    assert controller.telemetry["source"] == "QMIX"
    assert controller.telemetry["checkpointLoaded"] is False


def test_qmix_controller_actions_use_the_environment_step_boundary():
    environment = WarehouseEnvironment(num_robots=1)
    adapter = MARLEnvironmentAdapter(environment=environment)
    adapter.reset()
    controller = QMIXController(adapter)

    actions = controller.select_actions()
    _, rewards, terminated, global_state = adapter.step(actions)

    assert len(rewards) == 1
    assert isinstance(terminated, bool)
    assert len(global_state.robots) == 1
