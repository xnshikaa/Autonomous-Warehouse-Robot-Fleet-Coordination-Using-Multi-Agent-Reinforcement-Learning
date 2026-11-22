# Frontend–Backend Integration Report

## Overview

The React and Three.js frontend connects to the FastAPI WebSocket backend at
`ws://127.0.0.1:8000/ws`. Set `VITE_WAREHOUSE_WS_URL` when building for a
different WebSocket address.

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
foreach ($n in 5,10,20) {
    $selection = Get-Content -Raw "results/project25_validation/qmix_n$n.json" | ConvertFrom-Json
    Set-Item -Path "Env:QMIX_CHECKPOINT_N$n" -Value $selection.selected_checkpoint
}
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
The client sends only the fleet size. Configure checkpoint paths on the backend.

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

The production TypeScript WebSocket provider connected to a temporary Uvicorn
server over localhost. It received trained QMIX telemetry on each fleet change
in the sequence 5 → 10 → 20 → 5 and verified pause, speed, resume and reset.
The Vite production build and all six frontend connection assertions passed.
The Windows desktop composition error prevented a browser-rendered visual check.

```powershell
# Backend integration tests
python -m pytest tests/test_warehouse_backend.py tests/test_live_telemetry_integration.py -v

# Inference service tests
python -m pytest tests/test_inference_service.py -v

# Frontend tests
npm run test:frontend

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

## Docker verification procedure

Docker Desktop Linux engine verification is pending. In this session `docker
version` found the client in context `desktop-linux` but returned
`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`.
`docker desktop start --timeout 45` returned
`Docker Desktop is still starting: context deadline exceeded`.
On a machine with a running Docker Linux engine, run these commands from the
repository root to load the verified N=5 checkpoint and test actual inference.

```powershell
docker version
docker build --tag project25-inference .
docker run --detach --name project25-inference-check --publish 127.0.0.1:8001:8001 `
    --env QMIX_CHECKPOINT=/app/artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt `
    project25-inference
try {
    $deadline = (Get-Date).AddMinutes(3)
    do {
        $status = docker inspect --format '{{.State.Health.Status}}' project25-inference-check
        if ($status -eq 'healthy') { break }
        if ($status -eq 'unhealthy' -or (Get-Date) -gt $deadline) {
            docker logs project25-inference-check
            throw "Container health status is $status"
        }
        Start-Sleep -Seconds 3
    } while ($true)
    $base = 'http://127.0.0.1:8001'
    $health = Invoke-RestMethod "$base/health"
    $ready = Invoke-RestMethod "$base/readiness"
    $model = Invoke-RestMethod "$base/model/info"
    if (!$health.ok -or !$ready.ok -or !$model.trained -or $model.fleet_size -ne 5 -or $model.obs_dim -ne 456) {
        throw 'Expected trained N=5 checkpoint with 456 observations'
    }
    $obs = [double[]]::new(456)
    $request = @{fleet_size=5; observations=@($obs,$obs,$obs,$obs,$obs); epsilon=0} | ConvertTo-Json -Compress -Depth 4
    $decision = Invoke-RestMethod "$base/decide" -Method Post -ContentType 'application/json' -Body $request
    if (!$decision.trained -or $decision.actions.Count -ne 5 -or @($decision.actions | Where-Object { $_ -lt 0 -or $_ -gt 3 }).Count) {
        throw 'Invalid inference response'
    }
    $model
    $decision
} finally {
    docker stop project25-inference-check
    docker rm project25-inference-check
}
```

## Manual browser demonstration

Visual verification and screenshots are pending. Desktop capture returned
`DwmFlush after showing cursor overlay: Desktop composition is disabled (0x80263001)`.
The previous live check exercised the frontend provider with real WebSocket traffic.

1. From the repository root, run the three-checkpoint setup shown above and start
   Uvicorn on port 8000. In another terminal run `npm.cmd run dev -- --host 127.0.0.1`.
2. Open the Vite URL printed in the terminal. Let the intro finish and confirm
   that the backend connection indicator and `QMIX / TRAINED CHECKPOINT` appear.
   The simulation starts running on connection.
3. Select **5**, **10**, **20**, then **5**. At each size, pause and compare the
   visible fleet size and counters with `Invoke-RestMethod http://127.0.0.1:8000/state`.
   Check `policy.activeFleetSize`, `policy.trained`, `robots.Count`,
   `metrics.completedTasks`, and `simulationPaused`.
4. Click **RUN** and verify the tick advances, then click the reset icon and
   confirm that the episode increments. Capture a screenshot at each fleet size
   with the warehouse view and trained-policy indicator visible.
5. The browser's live controller is QMIX. IQL and Rule-Based are available
   through the benchmark runner. Show their matched benchmark results during
   the demonstration.
