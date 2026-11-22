# Project 25 verification

Branch `integration/final-demo` was three commits ahead of its upstream at takeover. The working tree contained changes to benchmark reporting and QMIX telemetry, plus untracked training runs, tests and the nine-configuration benchmark. The existing changes and checkpoints were preserved. The available Codex chat history contained no separate Project 25 execution, so recovery relied on Git, saved manifests, checkpoint content, logs and evaluation artifacts.

## Recovery inventory

| Area | Evidence and outcome |
|---|---|
| Recovered training | QMIX N=5, 10 and 20 each completed 1,000 episodes. IQL N=5, 10 and 20 each completed 500. The final models passed SHA-256 verification, optimizer-step checks, and inference through the paths used in evaluation. |
| Interrupted work | The first checkpoint inventory omitted final IQL and N=20 artifacts, and the newer benchmark had not used a frozen scenario manifest. Training itself had completed, including the three runs mentioned in the handoff. |
| Baseline | Rule-Based runs its deterministic policy without a learned checkpoint. |
| Historical tracking | The six manifests record configuration, seeds, losses and rewards. Historical wall-clock durations and MLflow run IDs could not be established. The current training CLI records both. |
| Older evaluation | The two original P2 frozen benchmarks remain in the repository. The higher-scoring Cycle 2 run references checkpoints absent from this checkout, so its headline scores describe that recorded run. |
| Prior full benchmark | The seed-999 run includes all nine configurations. Its N=20 snapshots completed zero tasks for both learned policies and its scenarios were used as exposed evidence. |
| Live execution | React and Three.js use the FastAPI WebSocket simulation. The trained demo loads a compatible QMIX snapshot for each fleet size and preserves the active model when a requested checkpoint is missing. |

## Verification

- The full Python suite passed 281 tests. The frontend passed all six connection assertions and built successfully with Vite.
- The compiled frontend provider connected to Uvicorn and exercised trained QMIX at all three fleet sizes over a live WebSocket. Windows desktop composition prevented the visual browser check.
- The bundled ONNX export matched the QMIX N=5 episode-600 snapshot with a maximum absolute difference of 2.39e-7. The inference endpoint returned the same actions with both model formats for a five-robot request.
- Short QMIX and IQL CLI smoke runs each performed optimizer updates and recorded a finished MLflow run with duration and episode metrics.
- The Docker Linux engine was unavailable in this session, leaving the container build unverified.

## Acceptance status

The fresh benchmark evaluated all nine algorithm and fleet combinations in 180
episodes. QMIX reached 78% at N=5 against the 80% target. The corridor gate
recorded zero intrusions in 20 attempts. Both learned policies recorded zero
physical contacts at N=10, leaving the collision-rate ratio undefined. The
N=20 learned policies completed four deliveries combined across 800 tasks.

See the [checkpoint inventory](CHECKPOINT_INVENTORY.md), [evaluation report](FINAL_EVALUATION_REPORT.md), [frontend integration report](FRONTEND_BACKEND_INTEGRATION_REPORT.md), and [reproduction guide](TRAINING_AND_REPRODUCTION_GUIDE.md).
