# Model Card — QMIX Warehouse Fleet Coordination Policy

## Model Details

| Field | Value |
|---|---|
| Model name | QMIX Cooperative Q-Mixing Network |
| Version | 1.0 |
| Architecture | CTDE — Centralized Training, Decentralized Execution |
| Algorithm | QMIX (Rashid et al., 2018) |
| Framework | PyTorch (custom implementation — no EPyMARL dependency) |
| Observation encoder | Map-aware local occupancy + BFS path-cost encoder |
| Observation dim | 456 (45-D local occupancy grid + path costs × 9 directions + robot/task features) |
| Action space | Discrete 4 (UP, DOWN, LEFT, RIGHT) |
| State dim (mixer) | 11 × N (3 robot features + 8 task features per robot) |
| Supported fleet sizes | N = 5, 10, 20 |
| Checkpoint schema | Version 2 (includes full architecture metadata for compatibility validation) |

## Intended Use

This model is designed for **research and demonstration** of cooperative multi-agent reinforcement learning in an autonomous warehouse robot fleet setting. It is **not intended for deployment in physical warehouses or with real robots**.

Intended use cases:
- Academic research into MARL algorithms for robot fleet coordination
- Comparison of QMIX, IQL, and rule-based nearest-task-assignment baselines
- Demonstration of CTDE (Centralized Training, Decentralized Execution) architecture

## Training Data & Environment

- **Environment**: Custom 20×20 grid-world warehouse simulation
- **Layout**: Shelf rows, pickup locations, dispatch zones, human-worker exclusion corridors
- **Safety**: Hard safety layer enforces zero corridor intrusions (action override)
- **Reward**: Cooperative task completion reward + per-robot collision penalty
- **Training seeds**: 42 (N=5), 43 (N=10), 44 (N=20)
- **Evaluation seeds**: Held-out (separated from training seeds)

## Architecture Description

```
Input: obs_i ∈ ℝ^456  (per robot i)
│
├── AgentNetwork (shared weights across all robots)
│   Linear(456→64) → ReLU → Linear(64→4) → Q(s,a)_i
│
└── QMixer (centralized, state-conditioned)
    state s ∈ ℝ^(11×N)
    Hypernetwork → W1(N×32), b1(32,), W2(32×1), b2()
    → monotonic mixing: Q_total = g(Q_1,...,Q_N; s)
```

The agent network is **shared** across all robots (parameter efficient, homogeneous fleet). The mixer is **fleet-size dependent** — a N=5 mixer cannot be loaded into a N=10 architecture (enforced by P6 checkpoint validator).

## Performance (Preliminary — Training in Progress)

> [!NOTE]
> Performance metrics are updated after full training runs complete. The values below reflect the evaluation pipeline and will be populated with actual results.

Target NFRs:
- **NFR-1**: QMIX ≥80% task-completion rate at N=5 (20 held-out episodes)
- **NFR-3**: Zero corridor intrusions across 20 safety episodes
- **NFR-4**: QMIX collision rate ≤50% of IQL at N=10
- **NFR-5**: Bootstrap 95% CI for all 3 algorithms × 3 fleet sizes

## Ethical Considerations

- **Physical safety**: The hard safety layer enforces corridor exclusion zones at the environment level, independent of learned policy. No robot action can override this.
- **Scope**: Software simulation only; no deployment on physical hardware.
- **Transparency**: All checkpoints include full architecture metadata (Schema V2). Synthetic/untrained checkpoints are explicitly labeled.
- **Fairness**: QMIX and IQL evaluations use identical task seeds (matched episodes) for fair comparison.
- **Carbon tracking**: Training energy consumption is tracked via CodeCarbon and a physical AMR energy model.

## Limitations

- CPU-only training (no GPU detected on training machine)
- PyBullet 3D physics unavailable without MSVC 14+ (software physics fallback used)
- No transfer learning; each fleet size trained independently
- Grid-world approximation; real warehouse dynamics differ significantly
- No heterogeneous robot types or continuous-action control

## Checkpoint Versioning

All checkpoints use **Schema Version 2** and include:
- Number of agents, observation dim, action dim, state dim
- Agent and mixer architecture identifiers
- Training episode count, seed, algorithm config
- Unique `checkpoint_id` for traceability

Incompatible checkpoints (wrong N, obs_dim, or action_dim) are **rejected with a clear error** — no silent partial loading.

## References

- Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. ICML 2018.
- Sunehag, P., et al. (2018). Value-Decomposition Networks For Cooperative Multi-Agent Learning. AAMAS 2018.
