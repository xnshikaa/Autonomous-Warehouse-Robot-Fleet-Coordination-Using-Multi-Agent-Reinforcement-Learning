"""FastAPI decision service for trained QMIX/ONNX model inference.

Endpoints:
    GET  /health     — liveness probe
    GET  /readiness  — readiness probe (checks model loaded)
    POST /decide     — accept fleet observations, return action vector
    GET  /model/info — active model metadata
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

if HAS_FASTAPI:
    class DecideRequest(BaseModel):
        fleet_size: int = Field(..., ge=1, le=20, description="Number of active robots")
        observations: List[List[float]] = Field(
            ...,
            description="List of per-robot observation vectors, shape [fleet_size, obs_dim]",
        )
        epsilon: float = Field(
            default=0.0, ge=0.0, le=1.0, description="Epsilon for greedy exploration"
        )

    class DecideResponse(BaseModel):
        actions: List[int]
        fleet_size: int
        obs_dim: int
        model_source: str
        checkpoint_id: Optional[str]
        trained: bool

    class ModelInfoResponse(BaseModel):
        model_source: str
        checkpoint_id: Optional[str]
        trained: bool
        fleet_size: Optional[int]
        obs_dim: Optional[int]
        action_dim: Optional[int]
        schema_version: Optional[int]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class _ONNXAgent:
    """Thin wrapper around an ONNX Runtime agent network."""

    def __init__(self, path: str | Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path))
        self.input_name = self.session.get_inputs()[0].name
        self.path = str(path)

    def select_actions(self, observations: np.ndarray, epsilon: float = 0.0) -> List[int]:
        """Return argmax actions for each row in observations."""
        fleet_size = observations.shape[0]
        actions = []
        for i in range(fleet_size):
            obs = observations[i : i + 1].astype(np.float32)
            q_values = self.session.run(None, {self.input_name: obs})[0]
            if epsilon > 0.0 and np.random.random() < epsilon:
                action = np.random.randint(q_values.shape[-1])
            else:
                action = int(np.argmax(q_values))
            actions.append(action)
        return actions


class _TorchAgent:
    """Thin wrapper around a PyTorch agent network loaded from a QMIX checkpoint."""

    def __init__(self, checkpoint_path: str | Path):
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        self.metadata = payload.get("metadata", {})
        self.obs_dim: int = self.metadata.get("obs_dim", payload.get("obs_dim", 456))
        self.action_dim: int = self.metadata.get("action_dim", payload.get("action_dim", 4))
        self.fleet_size: int = self.metadata.get("num_agents", payload.get("num_agents", 5))

        # Import and build the network
        from src.marl.qmix_network import AgentQNetwork
        self.net = AgentQNetwork(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.metadata.get("agent_hidden_dim", 64),
        )
        agent_state = payload.get("agent_network") or payload.get("agent_network_state")
        if agent_state is None:
            raise ValueError("Checkpoint missing 'agent_network' state dict")
        self.net.load_state_dict(agent_state, strict=True)
        self.net.eval()
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_id = self.metadata.get("checkpoint_id", Path(checkpoint_path).stem)

    def select_actions(self, observations: np.ndarray, epsilon: float = 0.0) -> List[int]:
        """Return argmax Q-value actions for all robots."""
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        actions = []
        with torch.no_grad():
            for i in range(obs_tensor.shape[0]):
                q_values = self.net(obs_tensor[i : i + 1])
                if epsilon > 0.0 and torch.rand(1).item() < epsilon:
                    action = int(torch.randint(q_values.shape[-1], (1,)).item())
                else:
                    action = int(q_values.argmax(dim=-1).item())
                actions.append(action)
        return actions


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------

_active_agent: _ONNXAgent | _TorchAgent | None = None
_model_info: Dict[str, Any] = {
    "model_source": "none",
    "checkpoint_id": None,
    "trained": False,
    "fleet_size": None,
    "obs_dim": None,
    "action_dim": None,
    "schema_version": None,
}


def load_model(checkpoint_path: str | Path | None = None, onnx_path: str | Path | None = None) -> bool:
    """Load a model at startup. Returns True if successfully loaded."""
    global _active_agent, _model_info

    # Prefer ONNX if available
    if onnx_path and Path(onnx_path).is_file():
        try:
            agent = _ONNXAgent(onnx_path)
            _active_agent = agent
            _model_info = {
                "model_source": "ONNX",
                "checkpoint_id": Path(onnx_path).stem,
                "trained": True,
                "fleet_size": None,
                "obs_dim": None,
                "action_dim": None,
                "schema_version": 2,
            }
            logger.info("ONNX agent loaded from %s", onnx_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load ONNX model: %s", exc)

    # Fall back to PyTorch checkpoint
    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            agent = _TorchAgent(checkpoint_path)
            _active_agent = agent
            _model_info = {
                "model_source": "PyTorch",
                "checkpoint_id": agent.checkpoint_id,
                "trained": True,
                "fleet_size": agent.fleet_size,
                "obs_dim": agent.obs_dim,
                "action_dim": agent.action_dim,
                "schema_version": 2,
            }
            logger.info("PyTorch agent loaded from %s (N=%d, obs=%d)", checkpoint_path, agent.fleet_size, agent.obs_dim)
            return True
        except Exception as exc:
            logger.warning("Failed to load PyTorch checkpoint: %s", exc)

    logger.warning("No model loaded — /decide endpoint will return 503 until a model is configured")
    return False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_inference_app(
    checkpoint_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
) -> "FastAPI":
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI is required; install requirements.txt")

    app = FastAPI(
        title="Warehouse MARL Inference Service",
        description="QMIX/ONNX model inference for the autonomous warehouse robot fleet.",
        version="1.0.0",
    )

    # Resolve checkpoint paths from environment if not provided
    resolved_ckpt = checkpoint_path or os.environ.get("QMIX_CHECKPOINT")
    resolved_onnx = onnx_path or os.environ.get("QMIX_ONNX")
    load_model(resolved_ckpt, resolved_onnx)

    @app.get("/health", tags=["ops"])
    def health():
        return {"ok": True, "service": "inference"}

    @app.get("/readiness", tags=["ops"])
    def readiness():
        if _active_agent is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return {"ok": True, "model_source": _model_info["model_source"]}

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
    def model_info():
        return _model_info

    @app.post("/decide", response_model=DecideResponse, tags=["inference"])
    def decide(request: DecideRequest):
        if _active_agent is None:
            raise HTTPException(status_code=503, detail="No model loaded; configure QMIX_CHECKPOINT or QMIX_ONNX")

        obs = np.array(request.observations, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[0] != request.fleet_size:
            raise HTTPException(
                status_code=422,
                detail=f"observations shape {obs.shape} does not match fleet_size={request.fleet_size}",
            )

        # Validate obs_dim if known
        if _model_info["obs_dim"] is not None and obs.shape[1] != _model_info["obs_dim"]:
            raise HTTPException(
                status_code=422,
                detail=f"Observation dim mismatch: model expects {_model_info['obs_dim']}, got {obs.shape[1]}",
            )

        try:
            actions = _active_agent.select_actions(obs, epsilon=request.epsilon)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

        return DecideResponse(
            actions=actions,
            fleet_size=request.fleet_size,
            obs_dim=obs.shape[1],
            model_source=_model_info["model_source"],
            checkpoint_id=_model_info.get("checkpoint_id"),
            trained=_model_info.get("trained", False),
        )

    return app


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    app = create_inference_app()
    uvicorn.run(app, host="127.0.0.1", port=8001)
