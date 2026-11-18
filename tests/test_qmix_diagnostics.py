from python_infra.checkpoint_manager import CheckpointManager
from src.evaluation.qmix_diagnostics import evaluate_checkpoint
from src.marl.qmix_trainer import QMIXTrainer


def test_qmix_diagnostics_records_real_policy_and_environment_state(tmp_path):
    trainer = QMIXTrainer(num_agents=5, episodes=1, max_steps=2, seed=42)
    manager = CheckpointManager(
        base_dir=str(tmp_path), algorithm="QMIX", run_id="diagnostic"
    )
    manager.save_checkpoint(
        trainer._get_checkpoint_payload(1, trainer.epsilon_end), step=1
    )

    results = evaluate_checkpoint(
        f"{manager.run_dir}/checkpoint_1", episodes=1, max_steps=2, seed=42
    )

    assert results["checkpoint_loaded"] is True
    assert results["total_tasks"] == 5
    episode = results["episode_results"][0]
    assert episode["steps"] == 2
    assert len(episode["trajectory"]) == 2
    step = episode["trajectory"][0]
    assert len(step["actions"]) == 5
    assert len(step["q_values"]) == 5
    assert len(step["robot_states"]) == 5
    assert "safety_events" in step
