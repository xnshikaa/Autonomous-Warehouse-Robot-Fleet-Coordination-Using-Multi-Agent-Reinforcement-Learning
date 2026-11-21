# Frontend–Backend Integration Report

## Overview

The frontend is a React 18 + Three.js + TypeScript application (`src/`) that connects to a Python FastAPI backend (`src/api/warehouse_backend.py`) via WebSocket at `ws://localhost:8000/ws`.

---

## Connection Architecture

```
Frontend (Vite + React)          Backend (FastAPI + uvicorn)
─────────────────────────        ─────────────────────────────────
PythonWebSocketProvider          warehouse_backend.create_app()
  │                                │
  ├─→ WebSocket connect            ├─→ /ws endpoint
  │   ws://127.0.0.1:8000/ws      │
  │                                │
  ├─→ {"type":"set_fleet_size",   ─→ _process_command_message()
  │    "size": 10}                 │   reconfigure_fleet(10)
  │                                │   environment.set_num_robots(10)
  │                                │
  ├─→ {"type":"pause"}            ─→ simulation_paused = True
  ├─→ {"type":"resume"}           ─→ simulation_paused = False
  ├─→ {"type":"set_speed","speed":2} → simulation_delay recalculated
  ├─→ {"type":"reset"}            ─→ environment.reset()
  │                                │
  ←─ state_update JSON ──────────── build_state_payload()
     (every simulation tick)
```

---

## State Payload Contract

The backend sends a `state_update` JSON message every simulation tick containing:

```typescript
{
  type: "state_update",
  timestep: number,
  episode: number,
  algorithm: "QMIX" | "DEMO",
  controllerType: "RL_BACKEND" | "DEMO_CONTROLLER",
  policy: {
    source: string,
    checkpointLoaded: boolean,
    trained: boolean,
    checkpointPath: string | null,
    modelStatus?: string,      // P6 — "trained" | "initialized" | "not_loaded"
    checkpointId?: string,     // P6 — unique checkpoint ID
    activeFleetSize?: number,  // P6 — current fleet size
  },
  config: { gridWidth, gridHeight, cellSize, layoutId, updateIntervalSeconds, speedMultiplier },
  layout: { ... },
  robots: [{ id, position, targetPosition, status, taskId, hasCargo, path, safetyAlert, ... }],
  tasks: [{ id, pickupPos, deliveryPos, assignedRobotId, status }],
  safetyZones: [{ id, name, bounds, activeWarning }],
  safetyEvents: [{ robotId, currentCell, overridden, reason, ... }],
  metrics: { activeRobots, completedTasks, safetyOverrides, ... },
  safetyOverrideActive: boolean,
  isDemoMode: boolean,
  simulationPaused: boolean,
  isConnectedToBackend: boolean,
}
```

---

## Fallback Mode

If the backend is unreachable (port 8000 not responding), `PythonWebSocketProvider` automatically:
1. Falls back to `DemoStateProvider` (BFS/greedy deterministic movement)
2. Reconnects every 5 seconds
3. Resumes backend control on successful reconnection

The frontend shows `isDemoMode: true` and `isConnectedToBackend: false` in demo mode.

---

## QMIX Policy Integration

When the backend is started with a QMIX checkpoint:

```powershell
$env:QMIX_CHECKPOINT = "artifacts/training/qmix_n5_v1/qmix.pt"
python -m uvicorn src.api.warehouse_backend:create_app --factory --port 8000
```

The backend:
1. Loads `QMIXController` with the checkpoint
2. At each simulation tick, calls `qmix_controller.select_actions()` → QMIX decisions
3. Sends `policy.trained: true`, `policy.source: "QMIX"`, `policy.checkpointId: "..."` in telemetry

Without a checkpoint:
1. Falls back to deterministic BFS/greedy demo controller
2. Sends `policy.trained: false`, `policy.source: "DEMO"` in telemetry

---

## Fleet Size Switching

The frontend sends `{"type": "set_fleet_size", "size": N}` via WebSocket.

The backend:
1. Calls `qmix_controller.reconfigure_fleet(N)` — validates P6 checkpoint compatibility
2. Calls `environment.set_num_robots(N)` — reconfigures the warehouse simulation
3. Sends back updated state with new fleet

If reconfiguration fails (no compatible checkpoint for the requested fleet size), the backend returns a `command_error` message and **preserves the previous valid controller**.

---

## Inference Service (/decide endpoint)

An additional FastAPI service (`src/api/inference_service.py`) exposes a REST `/decide` endpoint for external integration:

```
POST http://localhost:8001/decide
{
  "fleet_size": 5,
  "observations": [[...456 floats...], ...×5],
  "epsilon": 0.0
}

→ {"actions": [2, 0, 3, 1, 2], "fleet_size": 5, "obs_dim": 456, 
   "model_source": "PyTorch", "checkpoint_id": "...", "trained": true}
```

```powershell
# Start inference service
$env:QMIX_CHECKPOINT = "artifacts/training/qmix_n5_v1/qmix.pt"
python -m uvicorn src.api.inference_service:create_inference_app --factory --port 8001
```

---

## Testing

```powershell
# Backend integration tests
python -m pytest tests/test_warehouse_backend.py -v

# Inference service tests
python -m pytest tests/test_inference_service.py -v

# Frontend tests
npm run test:frontend -- --run

# Full TypeScript check
npx tsc --noEmit

# Production build
npm run build
```

---

## Known Integration Constraints

| Issue | Status | Notes |
|---|---|---|
| Backend requires port 8000 | ✅ | Configurable via uvicorn `--port` |
| Frontend fallback auto-reconnects | ✅ | 5s reconnect timer in `PythonWebSocketProvider` |
| Fleet-size change requires matching checkpoint | ✅ | P6 reconfigure_fleet() validates compatibility |
| QMIX decisions require loaded checkpoint | ✅ | Falls back to BFS demo if no checkpoint |
| PyBullet 3D bridge unavailable on Windows | 🔄 | Software physics fallback; frontend 3D visualization works |
| ONNX inference service | ✅ | Available at port 8001 |
