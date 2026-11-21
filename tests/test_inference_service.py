"""Tests for the MARL inference service (FastAPI /decide endpoint)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError

from src.api.inference_service import (
    create_inference_app,
    DecideRequest,
    _active_agent,
    _model_info,
    load_model,
)
import src.api.inference_service as svc


def _make_obs(fleet_size: int, obs_dim: int = 456) -> list[list[float]]:
    rng = np.random.default_rng(0)
    return rng.random((fleet_size, obs_dim)).tolist()


@pytest.fixture(autouse=True)
def clean_service_state():
    """Reset global module state before and after each test."""
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
    yield
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


@pytest.fixture
def torch_checkpoint(tmp_path):
    """Create a minimal valid checkpoint for N=5, obs_dim=456, action_dim=4."""
    import torch
    from src.marl.qmix_network import AgentQNetwork
    obs_dim, action_dim, hidden_dim, num_agents = 456, 4, 64, 5
    net = AgentQNetwork(observation_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
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


def test_health_endpoint():
    app = create_inference_app(checkpoint_path=None, onnx_path=None)
    health_route = next(r for r in app.routes if r.path == "/health")
    result = health_route.endpoint()
    assert result == {"ok": True, "service": "inference"}


def test_readiness_no_model():
    from fastapi import HTTPException
    app = create_inference_app(checkpoint_path=None, onnx_path=None)
    readiness_route = next(r for r in app.routes if r.path == "/readiness")
    with pytest.raises(HTTPException) as exc_info:
        readiness_route.endpoint()
    assert exc_info.value.status_code == 503


def test_model_info_no_model():
    app = create_inference_app(checkpoint_path=None, onnx_path=None)
    info_route = next(r for r in app.routes if r.path == "/model/info")
    info = info_route.endpoint()
    assert info["model_source"] == "none"
    assert info["trained"] is False


def test_decide_no_model():
    from fastapi import HTTPException
    app = create_inference_app(checkpoint_path=None, onnx_path=None)
    decide_route = next(r for r in app.routes if r.path == "/decide")
    req = DecideRequest(fleet_size=5, observations=_make_obs(5), epsilon=0.0)
    with pytest.raises(HTTPException) as exc_info:
        decide_route.endpoint(req)
    assert exc_info.value.status_code == 503


def test_health_with_model(torch_checkpoint):
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    health_route = next(r for r in app.routes if r.path == "/health")
    result = health_route.endpoint()
    assert result["ok"] is True


def test_readiness_with_model(torch_checkpoint):
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    readiness_route = next(r for r in app.routes if r.path == "/readiness")
    result = readiness_route.endpoint()
    assert result["ok"] is True
    assert result["model_source"] == "PyTorch"


def test_model_info_with_model(torch_checkpoint):
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    info_route = next(r for r in app.routes if r.path == "/model/info")
    info = info_route.endpoint()
    assert info["model_source"] == "PyTorch"
    assert info["trained"] is True
    assert info["obs_dim"] == 456
    assert info["fleet_size"] == 5


def test_decide_valid_n5(torch_checkpoint):
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    decide_route = next(r for r in app.routes if r.path == "/decide")
    req = DecideRequest(fleet_size=5, observations=_make_obs(5, 456), epsilon=0.0)
    resp = decide_route.endpoint(req)
    assert len(resp.actions) == 5
    assert all(0 <= a <= 3 for a in resp.actions)
    assert resp.trained is True
    assert resp.model_source == "PyTorch"


def test_decide_wrong_fleet_size(torch_checkpoint):
    from fastapi import HTTPException
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    decide_route = next(r for r in app.routes if r.path == "/decide")
    req = DecideRequest(fleet_size=5, observations=_make_obs(3, 456), epsilon=0.0)
    with pytest.raises(HTTPException) as exc_info:
        decide_route.endpoint(req)
    assert exc_info.value.status_code == 422


def test_decide_wrong_obs_dim(torch_checkpoint):
    from fastapi import HTTPException
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    decide_route = next(r for r in app.routes if r.path == "/decide")
    req = DecideRequest(fleet_size=5, observations=_make_obs(5, 50), epsilon=0.0)
    with pytest.raises(HTTPException) as exc_info:
        decide_route.endpoint(req)
    assert exc_info.value.status_code == 422


def test_decide_with_epsilon(torch_checkpoint):
    app = create_inference_app(checkpoint_path=str(torch_checkpoint))
    decide_route = next(r for r in app.routes if r.path == "/decide")
    req = DecideRequest(fleet_size=5, observations=_make_obs(5, 456), epsilon=1.0)
    resp = decide_route.endpoint(req)
    assert len(resp.actions) == 5
    assert all(0 <= a <= 3 for a in resp.actions)


def test_decide_invalid_fleet_size_zero():
    with pytest.raises(ValidationError):
        DecideRequest(fleet_size=0, observations=[], epsilon=0.0)
