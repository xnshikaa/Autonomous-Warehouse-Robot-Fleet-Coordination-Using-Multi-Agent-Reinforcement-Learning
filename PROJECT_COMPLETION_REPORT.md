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

- The core test suite passed with 72 passing tests in 43.21s, including:
  - 100/100 seeded single-robot pickup-and-delivery mission completion (`tests/test_single_robot_mission_100.py`), with 0 physical collisions, 0 shelf contacts, 0 corridor intrusions, and 0 stuck episodes.
  - Distinct collision and conflict metrics consistency tests (`test_benchmark_metrics_consistency.py`).
  - Live backend WebSocket and telemetry integration (`test_live_telemetry_integration.py`).
  - Comprehensive P6 checkpoint compatibility across N=5, 10, 20 (`test_p6_checkpoint_compatibility.py`).
  - REST inference service tests (`test_inference_service.py`).
  - Gymnasium environment contract tests (`test_gymnasium_env.py`).
  - Safety layer and human corridor enforcement tests (`test_safety_layer.py`, `test_corridor.py`).
- Frontend validation: TypeScript check (`npx tsc --noEmit`) completed with 0 errors, and the production build (`npm run build`) succeeded with Vite packaging in 19.54s.
- Camera follow persists across episode reset, camera reset, and the next mission while the robot remains in the fleet. Orbit and zoom do not clear it. The live backend clock is independent of WebSocket viewers, and visual movement follows committed cardinal segments.
- The earlier compiled frontend provider exercised trained QMIX across all three fleet sizes over a live WebSocket. The later Chrome verification is described below.
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

## Submission handover

| Stage | Evidence |
|---|---|
| Implemented | Learned and rule-based simulation policies, benchmark and safety instrumentation, strict live QMIX fleet switching, and an explicit weights-only warm start for controlled experiments. |
| Trained | The six verified original runs remain intact. Remediation added one 100-episode N=5 warm start and two 40-episode N=20 warm starts in separate directories. |
| Validated | N=5 fresh ranking, paired N=20 cycle-escape ablation, and N=20 matched reward settings. Neither warm-start experiment produced a replacement checkpoint. The cycle-escape heuristic remains a validation experiment. |
| Evaluated | The original manifest, its nine trace archives, scenario queues, checkpoint digests and generated artifacts were rechecked. The original N=5 selection and 78% completion stand. |
| Demonstrated | The production WebSocket provider was exercised against FastAPI with live trained fleet switching and simulation controls. The frontend build and six connection assertions passed again. |
| Outstanding | NFR-1 remains below target. NFR-4 remains inconclusive in the safety-resolved simulator. Docker health and inference checks require a running Linux engine. |

The [final NFR matrix](FINAL_EVALUATION_REPORT.md#acceptance-results) records
the measured verdicts. The [reproduction guide](TRAINING_AND_REPRODUCTION_GUIDE.md#remediation-experiments)
contains the training and validation commands. The
[deployment procedures](FRONTEND_BACKEND_INTEGRATION_REPORT.md#docker-verification-procedure)
record both environment blockers and the checks to run on a capable host.
Previous report versions are preserved under
[`results/project25_remediation/previous_reports`](results/project25_remediation/previous_reports/).

## Application release

- Restored the original perspective and orbit code in `src/camera/CameraManager.ts`
  from the parent of `90338b7`. The live scene uses the original `WarehouseCanvas`,
  `WarehouseSceneController`, floor, racks, AMR models, beacons and path systems.
- The control room opens directly into the warehouse. Its confirmed backend
  state includes all three controller types, their checkpoint status, actual
  routes, task and robot details, and episode telemetry. Offline state is
  marked stale and controls wait for fresh telemetry.
- QMIX remains on episodes 600, 1000 and 200 for N=5, N=10 and N=20. IQL
  uses episodes 300, 500 and 400. The new N=5 low-learning-rate run and the
  N=20 contention-aware escape remain validation evidence.
- The real Chrome check captured eight screenshots and passed fleet and
  controller changes, transport controls, inspection, reload and reconnect.
  Docker Desktop's Linux-engine named pipe was still unavailable.
- NFR-1 remains 78% against 80%. NFR-3 passed the rerun with zero intrusions
  in 20 attempts. NFR-4 remains inconclusive. NFR-5 retains the original
  nine-configuration bootstrap results.

Start the application with `./Start-Warehouse.ps1`. See the
[live integration report](FRONTEND_BACKEND_INTEGRATION_REPORT.md) for
screenshots, shutdown instructions and the Docker verification procedure.
