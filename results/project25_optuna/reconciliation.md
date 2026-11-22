# Project 25 QMIX results reconciliation

The current N=5 demo checkpoint completed **78 of 100 tasks** across 20 frozen episodes, with a 95% episode-bootstrap completion interval of **70–86%**, a mean of **34.78 steps** for completed deliveries, and **zero observed physical robot collisions**. It averaged **1.10 blocked robot collision intent groups per episode**. The 80% completion requirement was missed by two tasks. The same frozen suite measured **74/200** at N=10 and **3/400** at N=20. These figures apply to the selected checkpoint at each fleet size. See the [frozen report](../project25_holdout/report.md) and [benchmark manifest](../project25_holdout/manifest.json).

A fresh replay through the current inference path reproduced the archived N=5 task snapshots, actions, step counts, and safety events on all 20 frozen episodes. This replay checked the already selected incumbent only. See [replay verification](frozen_replay_verification.json).

## Historical figures

The checked-out project, local Git history searched for the quoted figures, validation records, Optuna storage, MLflow completion metrics, and benchmark traces yielded no attributable **81%** or **82%** QMIX completion point estimate. The N=5 episode-400 checkpoint did reach an **82% upper confidence bound** on exposed validation, with a point estimate of **33/50 (66%)**. The 2026-11 Cycle 2 N=10 holdout also has **82% as the lower confidence bound**, with a point estimate of **177/200 (88.5%)**. Either is an example of a recorded 82% value. The available records do not identify the source of the teammate’s figure.

| N=5 benchmark | Delivered / assigned | Evaluation seeds | Scenario provenance |
|---|---:|---|---|
| Current frozen holdout | 78/100 | 816597421–816597440 | Explicit queues in `results/project25_holdout/scenario_manifest.json` |
| Historical P2 benchmark | 83/100 | 20261120–20261139 | Seed-generated queues retained in the benchmark traces. Exposed benchmark with no separate frozen scenario manifest. |
| Historical November holdout | 83/100 | 976304181–976304200 | Explicit queues in `results/evaluation/p2_final_holdout_2026_11/scenario_manifest.json` |
| Historical Cycle 2 holdout | 86/100 | 20261121–20261140 | Explicit queues in `results/evaluation/p2_final_holdout_cycle2/scenario_manifest.json`. Nineteen of its twenty N=5 task queues are identical to queues in the exposed P2 benchmark. |

Each row used 20 N=5 episodes and a 100-step cap. Counts were recomputed from initial assignments and final task snapshots. The published results come from different checkpoint and scenario combinations. They cannot estimate a paired improvement over the current 78/100 figure.

| Benchmark | Training run and episode | Training seed | Checkpoint SHA-256 |
|---|---|---:|---|
| Current frozen holdout | `qmix_n5_v1`, episode 600 | 42 | `89d4a600569b92e68dfa888cd0ff081539c0417b30772ff4c15b1ebcf8309fe8` |
| Historical P2 benchmark | `p1_pathcost_n5_refine500`, episode 3600 | 200000 | `2570f9d2946683f95fb1f4570180ea28083dd82ce4e3480038741250ac3ff69c` |
| Both historical P2 holdouts | `p2_stable_n5_seed650005`, episode 3400 | 650005 | `fa2567f9bb27ee4cb798fdabb165572219e3121eb64d02db055bec04f5d4393c` |

The current checkpoint exists on disk and matches the recorded hash. Both historical checkpoints are absent at their recorded paths. A SHA-256 inventory of the local checkpoint files and MLflow artifacts found no matching copy. The preserved trace hashes and manifest records support the historical counts and checkpoint references. Exact file paths, scenario manifest IDs, episode seeds, and hash checks are recorded in [historical evidence](historical_evidence.json). See the [fresh on-disk checkpoint inventory](checkpoint_inventory.json), the [prior checkpoint report](../../CHECKPOINT_INVENTORY.md), and [verification records](../../artifacts/checkpoint_verification.json) for the N=5, 10, and 20 QMIX and IQL checkpoints.

The pre-existing Optuna SQLite study contained three complete unit-test trials. The legacy objective scored training episodes and suggested a mixer width that the trainer ignored. The original MLflow database contained 13 runs across four nonempty experiments. Its metrics had no matching 81% or 82% completion entry. The existing production training runs retain their original settings and provenance.

## Optuna protocol

The new study screens N=5 QMIX warm starts. Each trial starts from the same episode-600 agent weights and initializes a fresh mixer, optimizer and replay buffer. It trains for four episodes of up to 100 steps using the existing map-aware warehouse environment, 456 observation features per robot and a 55-feature centralized state. The common training seed is 254005, with task seeds 254005–254008, and the first six screening scenarios use task seeds 254105–254110. The independently frozen 20-episode confirmation set uses 255105–255124. Confirmation evaluates one winner and the incumbent on the same queues.

The score is completed tasks minus two bounded penalties. One penalty is 0.49 times safety conflicts divided by one plus conflicts. The other is 0.49 times mean completed delivery steps divided by 100, using 100 when there is no completed delivery. This gives each completed task greater weight than both penalties combined. Physical robot collisions are recorded separately. Rewards used in validation use the repository's default reward configuration.

| Parameter | Search range |
|---|---|
| Learning rate | 0.00003–0.001, log scale surrounding the original 0.0001 |
| Mixer width | 16, 32, or 64, around the original 32 |
| Discount factor | 0.95–0.999, including the neighborhood of the original 0.99 |
| Collision penalty | −20 to −2, surrounding the configured −10 |
| Delivery reward | 5–20, surrounding the configured 10 |

Sequential Optuna TPE uses 15 startup trials and a sampler seed of 254205 plus the trial number. Pruning is disabled. All complete trials have a checkpoint, a training history, and a compressed validation trace. SQLite maintains trial states, including interrupted trials. The study resumes at the next trial number after an interruption. It uses the original frozen holdout for neither screening nor confirmation. The short training budget and one training seed limit what can be inferred about training from scratch or across seeds.

Reproduce or resume from the repository root with `python -m src.marl.project25_optuna`. The same command defaults to a target of 100 completed trials.

An independent three-trial run reopened its SQLite study and appended trial four. All four corresponding parameter sets, objectives, training losses, and network tensors matched the production study. Checkpoint bytes differ because serialization includes checkpoint creation metadata. See [reproducibility check](reproducibility_check.json). All 291 Python tests passed. Application Control blocked the compiled SQLAlchemy extension on this workstation. The study and full tests used SQLAlchemy’s bundled pure-Python modules on an isolated import path.

## Completed study and demo decision

Optuna completed **100 trials**. All checkpoint hashes and validation counts were verified, and MLflow recorded 100 trial scores under run `a795b720c8894037961aaac8953b58bb`. Trial 47 led screening at **22/30** assigned deliveries against the incumbent’s **16/30** on the same six queues. Its four training episodes produced 369 optimizer updates. Its checkpoint SHA-256 is `9c54f52ab4d75bdba4d34c1b0171dc52ad42af3ce859f3918fa0903edbcd1acd`.

| N=5 policy on the same 20 confirmation episodes | Delivered / assigned | 95% episode-bootstrap CI | Mean delivery steps | Safety conflict events | Physical robot collisions |
|---|---:|---:|---:|---:|---:|
| Incumbent episode 600 | 69/100 | 56–81% | 34.01 | 167 | 0 |
| Optuna trial 47 | 56/100 | 44–67% | 37.53 | 57 | 0 |

The paired completion difference was **−13 percentage points** for trial 47, with a **95% interval of −28 to +2 points** and a two-sided paired permutation p-value of **0.141**. Trial 47 had fewer robot-level safety interventions, and the evidence does not establish a completion improvement. Keep the episode-600 checkpoint for the N=5 demo. Its 78/100 figure belongs to the original frozen holdout. The 69/100 figure comes from the separate confirmation queues and shows that measured completion varies by scenario set.

**Suggested presentation wording.** “We ran 100 Optuna trials that warm-started the same five-robot QMIX policy. The screening winner delivered 22 of 30 tasks, then delivered 56 of 100 on a separate matched validation set, where our existing checkpoint delivered 69. We kept the existing checkpoint. It delivered 78 of 100 on the original frozen holdout with zero observed physical collisions, two tasks short of the 80% target.”

[Optimization history](optimization_history.png) · [Hyperparameter importance](hyperparameter_importance.png) · [Best-trial parameters](best_trial.md) · [All trials](trials.csv) · [Paired confirmation and verification](study_verification.json)
