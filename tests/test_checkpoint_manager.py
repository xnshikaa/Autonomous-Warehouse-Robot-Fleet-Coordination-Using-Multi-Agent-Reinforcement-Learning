"""
Unit tests for algorithm-agnostic CheckpointManager.
"""

import os
import pytest
from python_infra.checkpoint_manager import CheckpointManager


@pytest.fixture
def tmp_checkpoint_mgr(tmp_path):
    base_dir = str(tmp_path / "checkpoints")
    mgr = CheckpointManager(base_dir=base_dir, algorithm="QMIX", run_id="test_run_123")
    return mgr


def test_save_and_load_checkpoint(tmp_checkpoint_mgr):
    data = {
        "model_weights": [1.0, 2.0, 3.0],
        "optimizer_state": {"lr": 0.001},
        "step": 100
    }
    saved_dir = tmp_checkpoint_mgr.save_checkpoint(data, step=100)
    assert os.path.exists(saved_dir)
    assert os.path.exists(os.path.join(saved_dir, "metadata.json"))

    loaded = tmp_checkpoint_mgr.load_checkpoint(100)
    assert loaded["metadata"]["step"] == 100
    assert loaded["metadata"]["algorithm"] == "QMIX"
    assert loaded["metadata"]["observation_vector"] == "50-D"
    assert loaded["metadata"]["action_space"] == "Discrete(4)"
    assert loaded["checkpoint_data"]["model_weights"] == [1.0, 2.0, 3.0]


def test_latest_and_best_checkpoint(tmp_checkpoint_mgr):
    data1 = {"val": 10}
    tmp_checkpoint_mgr.save_best_checkpoint(data1, step=10, current_metric_value=0.5, mode="max")

    data2 = {"val": 20}
    tmp_checkpoint_mgr.save_best_checkpoint(data2, step=20, current_metric_value=0.9, mode="max")

    latest = tmp_checkpoint_mgr.load_checkpoint("latest")
    assert latest["metadata"]["step"] == 20
    assert latest["checkpoint_data"]["val"] == 20

    best = tmp_checkpoint_mgr.load_checkpoint("best")
    assert best["metadata"]["step"] == 20
    assert best["metadata"]["best_metric_val"] == pytest.approx(0.9)
    assert best["checkpoint_data"]["val"] == 20


def test_resume_functionality(tmp_checkpoint_mgr):
    data = {"epoch": 50, "loss": 0.05}
    tmp_checkpoint_mgr.save_checkpoint(data, step=5000)

    latest_path = tmp_checkpoint_mgr.get_latest_checkpoint_path()
    assert latest_path is not None
    assert "latest" in latest_path

    resumed = tmp_checkpoint_mgr.load_checkpoint("latest")
    assert resumed["checkpoint_data"]["epoch"] == 50


def test_missing_checkpoint_graceful_handling(tmp_checkpoint_mgr):
    with pytest.raises(FileNotFoundError) as exc_info:
        tmp_checkpoint_mgr.load_checkpoint("non_existent_step_9999")
    assert "Checkpoint directory not found" in str(exc_info.value)


def test_list_checkpoints(tmp_checkpoint_mgr):
    tmp_checkpoint_mgr.save_checkpoint({"a": 1}, step=10)
    tmp_checkpoint_mgr.save_checkpoint({"a": 2}, step=20)

    chkpts = tmp_checkpoint_mgr.list_checkpoints()
    assert len(chkpts) >= 2
    steps = [c["step"] for c in chkpts if "step" in c]
    assert 10 in steps
    assert 20 in steps
