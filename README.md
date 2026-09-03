# Autonomous Warehouse Robot Fleet Coordination Visualizer (MARL / QMIX)

A premium, research-grade 3D visualization layer built for university project demonstrations on **Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning (QMIX)**.

---

## 🌟 Visual Identity & Design Principles

- **Industrial Robotics Lab Aesthetic**: Clean, dark polished concrete floor, instanced modular shelving racks, painted yellow hazard borders, defined storage aisles, conveyor delivery bays, inductive charging docks, and translucent human safety zones.
- **Decoupled Architecture**: Strict state separation (`WarehouseState`). Three.js renders state snapshots received from either a local **DemoMode** simulator or a live **Python WebSocket API** (`ws://localhost:8000/ws`).
- **Research Transparency**: High-performance multi-robot fleet rendering (N=5, 10, 20), animated 3D path line ribbons, live engineering control dashboard, interactive robot inspector drawers, and presentation mode.

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────────┐
│             Python RL Environment                      │
│      (Gym / PyMARL / QMIX / IQL / Greedy+A*)           │
└───────────────────────────┬────────────────────────────┘
                            │ WebSocket JSON Telemetry
                            ▼
┌────────────────────────────────────────────────────────┐
│            PythonWebSocketProvider                     │
│  (Subscribes to ws://localhost:8000/ws, with automatic │
│   fallback to DemoStateProvider when offline)          │
└───────────────────────────┬────────────────────────────┘
                            │ State Snapshot (WarehouseState)
                            ▼
┌────────────────────────────────────────────────────────┐
│            Three.js Visualization Layer                │
│  ├── WarehouseFloor (20x20 Grid, Aisles & Markings)   │
│  ├── ShelfSystem (Instanced Racks & Storage Crates)   │
│  ├── ZoneSystem (Dispatch, Charging, Human Zone)      │
│  ├── RobotSystem (AMR Fleet R01-R20 & LiDAR Towers)    │
│  ├── PathSystem (Glowing Directional Route Lines)     │
│  ├── TaskSystem (Pickup & Delivery Beacons)           │
│  └── SafetySystem (Human Zone Overrides & Collisions) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Industrial Engineering Control HUD (React)      │
│  ├── Live Dashboard (Algorithm, Fleet, Timestep, Ep)   │
│  ├── Robot & Task Inspector Panels                     │
│  ├── Multi-Camera Viewports (Overview, Top-Down, Free) │
│  └── Presentation Mode & Replay Speed Controls         │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
npm install

# 2. Start local development server
npm run dev

# 3. Type check & production build verification
npx tsc --noEmit
npx vite build
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🎮 Demo & Keyboard Controls

- **Camera Viewports**:
  - `OVERVIEW`: Angled high-bay perspective of the complete 20x20 warehouse layout.
  - `TOP-DOWN`: Tactical 2D overhead camera view for multi-agent coordination analysis.
  - `FREE ORBIT`: Left-click drag to rotate, right-click drag to pan, scroll to zoom.
- **Path Ribbon Visibility**:
  - Click `PATHS: ALL / SELECTED / OFF` in top right toolbar to cycle route overlay modes.
- **Robot Inspection**:
  - Left-click any AMR robot (`R01`–`R20`) in the 3D scene to open its inspector panel and click `TRACK ROBOT CAMERA`.
- **Presentation Mode**:
  - Click `PRESENTATION MODE` button in the top right header to toggle distraction-free widescreen view.
- **Simulation Replay Controls**:
  - Click `PLAY / PAUSE`, `RESET`, or `0.5x / 1x / 2x / 5x` speed multiplier in the bottom control bar.

---

## 📡 Python WebSocket Backend API Specification

The visualizer connects to `ws://localhost:8000/ws` and expects JSON payloads conforming to the `WarehouseState` schema:

```json
{
  "type": "state_update",
  "timestep": 1432,
  "episode": 812,
  "algorithm": "QMIX",
  "robots": [
    {
      "id": "R01",
      "position": [5, 7],
      "targetPosition": [12, 14],
      "rotation": 1.57,
      "status": "MOVING",
      "taskId": "T04",
      "hasCargo": true,
      "battery": 92,
      "path": [[5, 7], [6, 7], [7, 7], [8, 7], [9, 7]],
      "safetyAlert": false
    }
  ],
  "tasks": [
    {
      "id": "T04",
      "pickupPos": [5, 7],
      "pickupShelfId": "S14",
      "deliveryPos": [3, 18],
      "deliveryZoneId": "D01",
      "assignedRobotId": "R01",
      "status": "IN_PROGRESS"
    }
  ],
  "safetyZones": [],
  "recentCollisions": [],
  "metrics": {
    "activeRobots": 10,
    "totalRobots": 10,
    "activeTasks": 7,
    "completedTasks": 142,
    "throughputPerHour": 180,
    "averageDeliveryTimeSec": 12.4,
    "totalCollisions": 0,
    "safetyOverrides": 2,
    "timestep": 1432,
    "episode": 812
  },
  "safetyOverrideActive": false
}
```

---

## 🔗 Connecting to Python RL (QMIX) Backend Later

When ready to connect your Python Gym/PyMARL QMIX environment:

1. Create a simple FastAPI WebSocket endpoint in Python:
   ```python
   from fastapi import FastAPI, WebSocket
   import asyncio

   app = FastAPI()

   @app.websocket("/ws")
   async def websocket_endpoint(websocket: WebSocket):
       await websocket.accept()
       while True:
           # Get current state snapshot from Python QMIX environment
           state = env.get_warehouse_state()
           await websocket.send_json(state)
           await asyncio.sleep(0.1) # 10 Hz telemetry sync
   ```
2. Start the Python server on port 8000: `uvicorn main:app --port 8000`.
3. The Three.js frontend will automatically connect to `ws://localhost:8000/ws` and transition from `DEMO MODE` to `LIVE BACKEND`.

---

## ⚡ Technical Performance Optimizations

- **`InstancedMesh`**: Used for all storage rack tiers and crates, reducing draw calls from hundreds down to 2 draw calls.
- **Pixel Ratio Capping**: WebGLRenderer pixel ratio capped at `Math.min(window.devicePixelRatio, 2)` for high-DPI displays.
- **GSAP Tweens**: Smooth position & angular rotation interpolation prevents frame stuttering during grid transitions.

---

## 🎯 Recommended Next Step

Build the Python Gym / PyMARL environment backend with QMIX action selection and broadcast telemetry snapshots over `ws://localhost:8000/ws` to visualize live RL training/evaluation runs!

