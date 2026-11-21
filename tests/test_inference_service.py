"""Tests for the MARL inference service (FastAPI /decide endpoint)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _make_obs(fleet_size: int, obs_dim: int = 456) -> list[list[float]]:
    rng = np.random.default_rng(0)
    return rng.random((fleet_size, obs_dim)).tolist()


# ---------------------------------------------------------------------------
# Service with no model loaded
# ---------------------------------------------------------------------------

@pytest.fixture
def no_model_client():
    from src.api.inference_service import create_inference_app
    app = create_inference_app(checkpoint_path=None, onnx_path=None)
    return TestClient(app, raise_server_exceptions=False)


def test_health_no_model(no_model_client):
    r = no_model_client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_readiness_no_model_returns_503(no_model_client):
    r = no_model_client.get("/readiness")
    assert r.status_code == 503


def test_model_info_no_model(no_model_client):
    r = no_model_client.get("/model/info")
    assert r.status_code == 200
    assert r.json()["model_source"] == "none"
    assert r.json()["trained"] is False


def test_decide_no_model_returns_503(no_model_client):
    payload = {"fleet_size": 5, "observations": _make_obs(5), "epsilon": 0.0}
    r = no_model_client.post("/decide", json=payload)
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Service with a synthetic PyTorch checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def torch_checkpoint(tmp_path):
    """Create a minimal valid checkpoint for N=5, obs_dim=456, action_dim=4."""
    import torch
    from src.marl.qmix_network import AgentNetwork
    obs_dim, action_dim, hidden_dim, num_agents = 456, 4, 64, 5
    net = AgentNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    payload = {
        "schema_version": 2,
        "metadata": {
            "schema_version": 2,
            "num_agents": num_agents,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "agent_hidden_dim": hidden_dim,
            "checkpoint_id": "test-checkpoint-inference",
            "trained": True,
        },
        "agent_network": net.state_dict(),
    }
    path = tmp_path / "qmix_n5.pt"
    torch.save(payload, str(path))
    return path


@pytest.fixture
def torch_client(torch_checkpoint):
    from src.api.inference_service import create_inference_app, _model_info, load_model
    # Reset global state
    import src.api.inference_service as svc
    svc._active_agent = None
    svc._model_info = {
        "model_source": "none",
        "checkpoint_id": None,
        "trained": False,
        "fleet_size": None,
        "obs_dim": None,
        "action_dim": None,
        "schema_version": None,
    }
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    return TestClient(app, raise_server_exceptions=False)


def test_health_with_model(torch_client):
    r = torch_client.get("/health")
    assert r.status_code == 200


def test_readiness_with_model(torch_client):
    r = torch_client.get("/readiness")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_model_info_with_model(torch_client):
    r = torch_client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_source"] == "PyTorch"
    assert data["trained"] is True
    assert data["obs_dim"] == 456
    assert data["fleet_size"] == 5


def test_decide_valid_n5(torch_client):
    payload = {"fleet_size": 5, "observations": _make_obs(5, 456), "epsilon": 0.0}
    r = torch_client.post("/decide", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["actions"]) == 5
    assert all(0 <= a <= 3 for a in data["actions"])
    assert data["trained"] is True


def test_decide_wrong_fleet_size(torch_client):
    # 3 obs rows but fleet_size=5
    payload = {"fleet_size": 5, "observations": _make_obs(3, 456), "epsilon": 0.0}
    r = torch_client.post("/decide", json=payload)
    assert r.status_code == 422


def test_decide_wrong_obs_dim(torch_client):
    payload = {"fleet_size": 5, "observations": _make_obs(5, 50), "epsilon": 0.0}
    r = torch_client.post("/decide", json=payload)
    assert r.status_code == 422


def test_decide_with_epsilon(torch_client):
    # With epsilon=1.0 all actions should still be valid (random)
    payload = {"fleet_size": 5, "observations": _make_obs(5, 456), "epsilon": 1.0}
    r = torch_client.post("/decide", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["actions"]) == 5


def test_decide_invalid_fleet_size_zero(no_model_client):
    payload = {"fleet_size": 0, "observations": [], "epsilon": 0.0}
    r = no_model_client.post("/decide", json=payload)
    # Pydantic validation rejects fleet_size=0 (ge=1)
    assert r.status_code == 422
