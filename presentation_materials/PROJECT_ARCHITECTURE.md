# Project 25 — System Architecture & Workflow Specification

This document provides the complete, factually verified architecture and execution flow for **Project 25: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning (QMIX)**.

---

## 1. Component Implementation Inventory

| Component | Purpose | Actual Implementation | Key Files | Verified Status |
|---|---|---|---|---|
| **Warehouse Simulation & GridWorld** | Discrete 20×20 warehouse grid simulation with aisles, shelf racks, and human corridors | Custom Python environment with deterministic movement dynamics, boundary enforcement, and 4-action directional execution | [src/ai/environment.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/ai/environment.py) | **VERIFIED** |
| **Warehouse Layout Library** | Validated static geometries, obstacle masks, corridor zones, and robot start locations | Immutable data-driven layout classes (`WarehouseLayout`, `LayoutLibrary`) | [src/warehouse/layouts.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/warehouse/layouts.py) | **VERIFIED** |
| **PyBullet 3D Simulation** | Physics-based simulation bridge for continuous motion validation and collision dynamics | PyBullet warehouse bridge with software/headless fallback | [src/simulation/pybullet_warehouse.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/simulation/pybullet_warehouse.py) | **VERIFIED** |
| **Pre-Action Hard Safety Layer** | Non-bypassable deterministic safety gate enforcing human exclusion corridor boundaries | Intercepts proposed actions before environment state mutation; overrides unsafe moves with `NO_OP` | [src/safety/safety_layer.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/safety/safety_layer.py), [src/safety/corridor.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/safety/corridor.py) | **VERIFIED** (0 intrusions in 20 episodes) |
| **Same-Cell Contention & Conflict Resolution** | Collision avoidance for multi-agent moves and position swaps | Deterministic cell reservation: stationary occupant retains cell; incoming contenders blocked; empty cells use lowest-ID priority | [src/ai/environment.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/ai/environment.py#L338-L365) | **VERIFIED** |
| **Local Observation Encoder** | Extracts per-agent state representation | 456-dimensional vector: 45-D local 3×3 occupancy, 3-D task geometry, 2-D flags, 2-D normalized coordinates, 400-D static map, 4-D BFS path costs | [src/marl/observation_encoder.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/observation_encoder.py) | **VERIFIED** |
| **Centralized Global State Encoder** | Centralized state representation for hypernetwork mixer | $11 \times N$ dimensional vector: 3 features per robot (`norm_x`, `norm_y`, `carrying`) + 8 features per task | [src/ai/state.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/ai/state.py), [src/marl/environment_adapter.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/environment_adapter.py) | **VERIFIED** |
| **Agent Neural Network** | Computes individual action-utilities $Q_i(o_i, a_i)$ | Homogeneous shared 3-layer MLP: `Linear(456, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 4)` | [src/marl/qmix_network.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/qmix_network.py#L15-L50) | **VERIFIED** |
| **QMIX Centralized Mixer** | Monotonically mixes individual agent utilities into $Q_{tot}$ | Hypernetworks generate weights $W_1 \in \mathbb{R}^{N \times 32}, W_2 \in \mathbb{R}^{32 \times 1}$ conditioned on state $s$, passed through absolute value activation ($\|W\|$) | [src/marl/qmix_network.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/qmix_network.py#L55-L135) | **VERIFIED** |
| **Multi-Agent Replay Buffer** | Stores transition tuples $(s, \mathbf{o}, \mathbf{a}, r, s', \mathbf{o}', d)$ | Fixed-capacity circular buffer with uniform transition sampling | [src/marl/replay_buffer.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/replay_buffer.py) | **VERIFIED** |
| **QMIX Learner & Trainer** | Centralized loss calculation, backpropagation, and target updates | PyTorch Adam optimizer ($lr=10^{-4}$), Smooth L1 TD loss, gradient clipping ($max\_norm=10.0$), periodic target synchronization | [src/marl/qmix_learner.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/qmix_learner.py), [src/marl/qmix_trainer.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/qmix_trainer.py) | **VERIFIED** |
| **Checkpoint Management & Schema V2** | Robust serialization with strict architectural metadata validation | Enforces fleet-size match ($N=5, 10, 20$), layer dimension checks, and state-dict verification | [python_infra/checkpoint_manager.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/python_infra/checkpoint_manager.py), [src/marl/checkpoint_validator.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/checkpoint_validator.py) | **VERIFIED** (23/23 tests pass) |
| **FastAPI Backend & WebSocket Stream** | Real-time telemetry streaming and control plane | Async WebSocket endpoint (`/ws`), REST health and state endpoints (`/health`, `/readiness`, `/state`) | [src/api/warehouse_backend.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/api/warehouse_backend.py) | **VERIFIED** |
| **REST Inference Service** | High-performance model-serving endpoint | `/decide` endpoint accepting observation batches, producing action decisions via ONNX Runtime or PyTorch | [src/api/inference_service.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/api/inference_service.py) | **VERIFIED** |
| **React 18 & Three.js 3D Frontend** | Hardware-accelerated 3D warehouse visualization | Modular 3D scene (AMR meshes, racks, path traces, beacons), camera presets (Overview, Top-Down, Robot Follow), live telemetry HUD | [src/App.tsx](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/App.tsx), [src/scene/WarehouseSceneController.ts](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/scene/WarehouseSceneController.ts) | **VERIFIED** |
| **Independent Q-Learning (IQL)** | Non-cooperative baseline algorithm | Decentralized training with $N$ independent Q-networks without a mixing network | [src/marl/iql_network.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/iql_network.py), [src/marl/iql_trainer.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/iql_trainer.py) | **VERIFIED** |
| **Rule-Based Baseline** | Deterministic heuristic benchmark | Greedy nearest-task allocation with reservation-aware Breadth-First Search (BFS) pathfinding | [src/ai/live_navigation.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/ai/live_navigation.py), [src/evaluation/trace_generator.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/evaluation/trace_generator.py) | **VERIFIED** |
| **MLflow Experiment Tracking** | Metrics logging, artifact management, and parameter recording | SQLite-backed MLflow database (`mlflow.db`) tracking runs across experiments | [src/monitoring/experiment_tracker.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/monitoring/experiment_tracker.py) | **VERIFIED** |
| **Optuna Hyperparameter Tuning** | Automated hyperparameter optimization | Study framework with composite multi-objective scalarization ($Completion \times 100 - Conflicts \times 2$) | [src/marl/optuna_study.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/optuna_study.py) | **VERIFIED** |
| **Evidently AI Drift Monitoring** | Production data drift and distribution shift monitoring | Population Stability Index (PSI) and Wasserstein distance calculations on observation distributions | [src/monitoring/drift_monitor.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/monitoring/drift_monitor.py) | **VERIFIED** |
| **CodeCarbon Energy Tracking** | Environmental impact and compute carbon tracking | Monitors emissions and kilowatt-hours consumed during training episodes | [src/monitoring/carbon_tracker.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/monitoring/carbon_tracker.py) | **VERIFIED** |
| **ONNX Export & Parity Check** | Model optimization for edge deployment | Exports PyTorch `AgentQNetwork` to ONNX; verifies output tensor parity ($max\_diff < 10^{-6}$) | [src/marl/export_onnx.py](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/src/marl/export_onnx.py) | **VERIFIED** ($2.39 \times 10^{-7}$) |
| **Cloud Deployment** | Cloud edge delivery of 3D control room | Production deployment hosted on Vercel with single-page routing | [vercel.json](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/vercel.json) | **VERIFIED** (Live at [marl-qmix-warehouse-3d.vercel.app](https://marl-qmix-warehouse-3d.vercel.app)) |

---

## 2. Centralized Training with Decentralized Execution (CTDE) Architecture

```mermaid
flowchart TD
    subgraph Environment ["Warehouse Simulation Environment (20x20 Grid)"]
        State["Global State s ∈ ℝ^(11×N)"]
        Obs1["Obs o_1 ∈ ℝ^456"]
        Obs2["Obs o_2 ∈ ℝ^456"]
        ObsN["Obs o_N ∈ ℝ^456"]
    end

    subgraph DecentralizedExecution ["Decentralized Execution (Per Robot)"]
        AgentNet1["Agent Q-Network 1<br/>Linear(456→64→64→4)"]
        AgentNet2["Agent Q-Network 2<br/>Linear(456→64→64→4)"]
        AgentNetN["Agent Q-Network N<br/>Linear(456→64→64→4)"]
        
        Q1["Q_1(o_1, ·)"]
        Q2["Q_2(o_2, ·)"]
        QN["Q_N(o_N, ·)"]

        Act1["Action a_1 = argmax Q_1"]
        Act2["Action a_2 = argmax Q_2"]
        ActN["Action a_N = argmax Q_N"]
    end

    Obs1 --> AgentNet1 --> Q1 --> Act1
    Obs2 --> AgentNet2 --> Q2 --> Act2
    ObsN --> AgentNetN --> QN --> ActN

    subgraph SafetyGate ["Pre-Action Deterministic Safety Gate"]
        CheckCorridor{"Human Corridor /<br/>Obstacle Check"}
        ResolveContention{"Same-Cell Contention /<br/>Swap Check"}
        SafeActions["Committed Actions a*"]
    end

    Act1 & Act2 & ActN --> CheckCorridor --> ResolveContention --> SafeActions
    SafeActions -->|"Execute Step"| Environment

    subgraph CentralizedTraining ["Centralized Training (Learner Only)"]
        ReplayBuffer[("Multi-Agent Replay Buffer<br/>(Capacity: 5000)")]
        
        HyperMixer["QMIX Mixing Network<br/>Hypernetworks: W1, b1, W2, b2<br/>Conditioned on State s<br/>Constraint: ∂Q_tot / ∂Q_i ≥ 0"]
        
        Qtot["Centralized Joint Utility Q_tot"]
        TargetNet["Target Agent Networks & Mixer<br/>(Updated every 200 episodes)"]
        TDLoss["Smooth L1 TD Loss<br/>L(θ) = (r + γ max Q_tot' - Q_tot)^2"]
    end

    Environment -->|"Store Transitions (s, o, a, r, s', o', d)"| ReplayBuffer
    ReplayBuffer -->|"Sample Minibatch (B=32)"| HyperMixer
    State -.->|"Conditioning State s"| HyperMixer
    Q1 & Q2 & QN -.->|"Chosen Action Q_i"| HyperMixer
    HyperMixer --> Qtot
    Qtot --> TDLoss
    TargetNet --> TDLoss
    TDLoss -->|"Backpropagation & Adam Optimizer"| AgentNet1 & AgentNet2 & AgentNetN
    TDLoss -->|"Backprop Hypernetworks"| HyperMixer
```

---

## 3. End-to-End System Deployment Architecture

```mermaid
flowchart LR
    subgraph ClientLayer ["Client Presentation Layer (Browser / Cloud)"]
        WebBrowser["React 18 Web Client<br/>(Hosted on Vercel)"]
        ThreeJS["Three.js 3D Viewport<br/>AMR Meshes, Racks, Beacons"]
        Dashboard["Telemetry HUD & KPI Metrics<br/>Algorithm Selector, Fleet Size"]
        WebBrowser --- ThreeJS
        WebBrowser --- Dashboard
    end

    subgraph NetworkLayer ["Real-Time Communication Layer"]
        WSServer["WebSocket Protocol<br/>ws://127.0.0.1:8000/ws<br/>Bi-directional Telemetry Stream"]
        RESTAPI["FastAPI HTTP Endpoints<br/>/health, /readiness, /state<br/>/decide (REST Inference)"]
    end

    Dashboard <-->|"Commands (Start, Pause, Reset, Fleet)"| WSServer
    ThreeJS <---|JSON Robot Positions & Status| WSServer
    Dashboard <---|Metrics & Checkpoint Status| RESTAPI

    subgraph BackendLayer ["Python Application Backend (FastAPI / PyTorch)"]
        BackendController["Warehouse Backend Service<br/>(src/api/warehouse_backend.py)"]
        InferenceService["ONNX / PyTorch Inference Engine<br/>(src/api/inference_service.py)"]
        P6Validator["P6 Checkpoint Validator<br/>(Schema V2 Metadata Verification)"]
    end

    WSServer <--> BackendController
    RESTAPI <--> InferenceService
    BackendController --> P6Validator

    subgraph ModelLayer ["Execution & Policy Layer"]
        QMIXController["QMIX Inference Controller<br/>checkpoints/qmix_n5_v1/qmix.pt<br/>checkpoints/qmix_n10_v1/qmix.pt<br/>checkpoints/qmix_n20_v1/qmix.pt"]
        IQLController["IQL Controller<br/>checkpoints/iql_n5_v1/iql.pt<br/>checkpoints/iql_n10_v1/iql.pt<br/>checkpoints/iql_n20_v1/iql.pt"]
        RuleBasedNavigator["WarehouseNavigator<br/>Deterministic BFS Reservation Policy"]
    end

    BackendController -->|"Selected Policy"| QMIXController
    BackendController -->|"Selected Policy"| IQLController
    BackendController -->|"Selected Policy"| RuleBasedNavigator

    subgraph SimulationCore ["Simulation Core & Physics"]
        EnvCore["WarehouseEnvironment (20x20)<br/>Episode Manager, Task Queue"]
        SafetyGateCore["SafetyLayer & CorridorModule<br/>Hard Exclusion Enforcement"]
    end

    QMIXController & IQLController & RuleBasedNavigator --> EnvCore
    EnvCore <--> SafetyGateCore
    EnvCore -->|"State Updates (10 Hz)"| BackendController
```
