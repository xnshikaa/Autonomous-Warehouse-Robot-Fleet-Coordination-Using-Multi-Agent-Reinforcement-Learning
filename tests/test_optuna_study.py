"""Unit tests for the Optuna hyperparameter optimization study."""

from pathlib import Path
import pytest
from src.marl.optuna_study import run_optuna_study


def test_optuna_study_execution_and_summary(tmp_path):
    summary = run_optuna_study(
        num_agents=5,
        n_trials=3,
        episodes=2,
        max_steps=10,
        seed=101,
        output_dir=tmp_path / "optuna",
    )

    assert summary["n_trials"] == 3
    assert "best_trial" in summary
    assert "best_params" in summary["best_trial"]
    assert "learning_rate" in summary["best_trial"]["best_params"]
    assert (tmp_path / "optuna" / "qmix_study.db").is_file()
    assert (tmp_path / "optuna" / "study_summary.json").is_file()
    assert len(summary["pareto_optimal_trials"]) >= 1
