# Complete Checkpoint & Model Inventory

Audited catalog of all reinforcement learning checkpoints across the project repository, tracking algorithm architecture, fleet dimensions, training provenance, SHA-256 integrity, and evaluation lifecycle status.

## Verified Final Training Runs

All six final policies passed saved checksum verification, episode-history checks, optimizer counter checks and action selection through the evaluation inference path.
Rule-Based uses no learned checkpoint.
Final weights match each run's last snapshot. Full hashes, history paths, schema versions and optimizer counters are in [checkpoint_verification.json](artifacts/checkpoint_verification.json).

| Algorithm | N | Observation | Global state | Episodes | Environment steps | Optimizer updates (total) |
|---|---:|---:|---:|---:|---:|---:|
| QMIX | 5 | 456 | 55 | 1,000 | 98,919 | 98,888 |
| IQL | 5 | 456 | n/a | 500 | 43,747 | 218,735 |
| QMIX | 10 | 456 | 110 | 1,000 | 99,983 | 99,952 |
| IQL | 10 | 456 | n/a | 500 | 50,000 | 500,000 |
| QMIX | 20 | 456 | 220 | 1,000 | 100,000 | 99,969 |
| IQL | 20 | 456 | n/a | 500 | 50,000 | 1,000,000 |

The historical manifests do not record wall-clock duration or an MLflow run ID. IQL update totals sum independent optimizers across the fleet.

The trained live demo uses the three QMIX snapshots recorded in [the validation selections](results/project25_validation/).

## Inventory Table

| Algorithm | N | Obs Dim | State Dim | Episodes | Updates | Run ID | File / Directory Path | SHA-256 (Prefix) | Evaluation Status |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| IQL | 10 | 456 | n/a (decentralized) | 500 | 500,000 total | untracked | `artifacts/training/iql_n10_v1/agents` | `7e5ac5f7f894146d...` | **evaluated** |
| IQL | 10 | 456 | n/a (decentralized) | 100 | 100,000 total | untracked | `artifacts/training/iql_n10_v1/snapshots/episode_00100` | `edc95cfc060e3be9...` | **evaluated** |
| IQL | 10 | 456 | n/a (decentralized) | 200 | 200,000 total | untracked | `artifacts/training/iql_n10_v1/snapshots/episode_00200` | `231ab20a8ca2f941...` | **trained but unevaluated** |
| IQL | 10 | 456 | n/a (decentralized) | 300 | 300,000 total | untracked | `artifacts/training/iql_n10_v1/snapshots/episode_00300` | `b21d6031649e1082...` | **evaluated** |
| IQL | 10 | 456 | n/a (decentralized) | 400 | 400,000 total | untracked | `artifacts/training/iql_n10_v1/snapshots/episode_00400` | `633c336792bd8941...` | **trained but unevaluated** |
| IQL | 10 | 456 | n/a (decentralized) | 500 | 500,000 total | untracked | `artifacts/training/iql_n10_v1/snapshots/episode_00500` | `7e5ac5f7f894146d...` | **evaluated** |
| IQL | 20 | 456 | n/a (decentralized) | 500 | 1,000,000 total | untracked | `artifacts/training/iql_n20_v1/agents` | `f0ed1bb97e3dd354...` | **trained but unevaluated** |
| IQL | 20 | 456 | n/a (decentralized) | 100 | 200,000 total | untracked | `artifacts/training/iql_n20_v1/snapshots/episode_00100` | `e9a4402a57809d66...` | **evaluated** |
| IQL | 20 | 456 | n/a (decentralized) | 200 | 400,000 total | untracked | `artifacts/training/iql_n20_v1/snapshots/episode_00200` | `5cbd0279cfb9f744...` | **trained but unevaluated** |
| IQL | 20 | 456 | n/a (decentralized) | 300 | 600,000 total | untracked | `artifacts/training/iql_n20_v1/snapshots/episode_00300` | `f8e375bf87f0f701...` | **trained but unevaluated** |
| IQL | 20 | 456 | n/a (decentralized) | 400 | 800,000 total | untracked | `artifacts/training/iql_n20_v1/snapshots/episode_00400` | `f72cfd757488110f...` | **evaluated** |
| IQL | 20 | 456 | n/a (decentralized) | 500 | 1,000,000 total | untracked | `artifacts/training/iql_n20_v1/snapshots/episode_00500` | `f0ed1bb97e3dd354...` | **trained but unevaluated** |
| IQL | 5 | 456 | n/a (decentralized) | 500 | 218,735 total | untracked | `artifacts/training/iql_n5_v1/agents` | `f7ceae66249219c0...` | **trained but unevaluated** |
| IQL | 5 | 456 | n/a (decentralized) | 100 | 50,000 total | untracked | `artifacts/training/iql_n5_v1/snapshots/episode_00100` | `8461432a374f3bec...` | **trained but unevaluated** |
| IQL | 5 | 456 | n/a (decentralized) | 200 | 98,915 total | untracked | `artifacts/training/iql_n5_v1/snapshots/episode_00200` | `8156e0d47e859c47...` | **evaluated** |
| IQL | 5 | 456 | n/a (decentralized) | 300 | 141,760 total | untracked | `artifacts/training/iql_n5_v1/snapshots/episode_00300` | `2e62a512ae4aa57c...` | **evaluated** |
| IQL | 5 | 456 | n/a (decentralized) | 400 | 180,610 total | untracked | `artifacts/training/iql_n5_v1/snapshots/episode_00400` | `20090116822e8b49...` | **trained but unevaluated** |
| IQL | 5 | 456 | n/a (decentralized) | 500 | 218,735 total | untracked | `artifacts/training/iql_n5_v1/snapshots/episode_00500` | `f7ceae66249219c0...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 1000 | 99952 | untracked | `artifacts/training/qmix_n10_v1/qmix.pt` | `364c0885cd6e90cf...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 100 | 9969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00100/checkpoint_state.pt` | `e63fa7a234cc6efa...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 200 | 19969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00200/checkpoint_state.pt` | `30cd5b21a7efcc69...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 300 | 29969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00300/checkpoint_state.pt` | `257c8cc37d6c7505...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 400 | 39969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00400/checkpoint_state.pt` | `45f6d566083adce7...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 500 | 49969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00500/checkpoint_state.pt` | `8d1d4521d327f5e4...` | **evaluated** |
| QMIX | 10 | 456 | 110 | 600 | 59969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00600/checkpoint_state.pt` | `aafe0d43f5528b1e...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 700 | 69969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00700/checkpoint_state.pt` | `e274a24bc1d93571...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 800 | 79969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00800/checkpoint_state.pt` | `1deeed05492031d7...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 900 | 89969 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_00900/checkpoint_state.pt` | `e0660ed08eaf201a...` | **trained but unevaluated** |
| QMIX | 10 | 456 | 110 | 1000 | 99952 | untracked | `artifacts/training/qmix_n10_v1/snapshots/episode_01000/checkpoint_state.pt` | `43e30259e9de5df5...` | **evaluated** |
| QMIX | 20 | 456 | 220 | 1000 | 99969 | untracked | `artifacts/training/qmix_n20_v1/qmix.pt` | `48224fec0467a955...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 100 | 9969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00100/checkpoint_state.pt` | `a41f89e2a781a340...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 200 | 19969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00200/checkpoint_state.pt` | `16421fd30302cb9b...` | **evaluated** |
| QMIX | 20 | 456 | 220 | 300 | 29969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00300/checkpoint_state.pt` | `c2ea7ad028e8bec5...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 400 | 39969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00400/checkpoint_state.pt` | `4e62c739934f114a...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 500 | 49969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00500/checkpoint_state.pt` | `eeb4ee650667dc2b...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 600 | 59969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00600/checkpoint_state.pt` | `9cdbbe9a5cd113b9...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 700 | 69969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00700/checkpoint_state.pt` | `359e5f259d8ea735...` | **evaluated** |
| QMIX | 20 | 456 | 220 | 800 | 79969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00800/checkpoint_state.pt` | `a22631e059071903...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 900 | 89969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_00900/checkpoint_state.pt` | `f855723b9ac39029...` | **trained but unevaluated** |
| QMIX | 20 | 456 | 220 | 1000 | 99969 | untracked | `artifacts/training/qmix_n20_v1/snapshots/episode_01000/checkpoint_state.pt` | `4c3646699b1c28af...` | **trained but unevaluated** |
| QMIX | 5 | 50 | 250 | 60 | 2337 | untracked | `artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt` | `0cc25c47d89ed57e...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 50 | 2469 | untracked | `artifacts/training/qmix_n5_smoke/qmix.pt` | `3ce8f03a39efa021...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 10 | 469 | untracked | `artifacts/training/qmix_n5_smoke/snapshots/episode_00010/checkpoint_state.pt` | `7858afa5ce1dbbe7...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 20 | 969 | untracked | `artifacts/training/qmix_n5_smoke/snapshots/episode_00020/checkpoint_state.pt` | `bee8262fb480458f...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 30 | 1469 | untracked | `artifacts/training/qmix_n5_smoke/snapshots/episode_00030/checkpoint_state.pt` | `135116649af7e581...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 40 | 1969 | untracked | `artifacts/training/qmix_n5_smoke/snapshots/episode_00040/checkpoint_state.pt` | `924fb11c3c0815bc...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 50 | 2469 | untracked | `artifacts/training/qmix_n5_smoke/snapshots/episode_00050/checkpoint_state.pt` | `8add6ed7e27ffcaa...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 1000 | 98888 | untracked | `artifacts/training/qmix_n5_v1/qmix.pt` | `3d3bbb21daba4ad4...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 100 | 9969 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00100/checkpoint_state.pt` | `067486b42c2b70da...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 200 | 19969 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00200/checkpoint_state.pt` | `29d53e1ba2a58a08...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 300 | 29969 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00300/checkpoint_state.pt` | `62afc59eade38165...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 400 | 39969 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00400/checkpoint_state.pt` | `16803669fd8c6e35...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 500 | 49956 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00500/checkpoint_state.pt` | `b5339de9c6d8691d...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 600 | 59860 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt` | `89d4a600569b92e6...` | **evaluated** |
| QMIX | 5 | 456 | 55 | 700 | 69357 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00700/checkpoint_state.pt` | `118c3d597d35b735...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 800 | 79100 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00800/checkpoint_state.pt` | `e30d26f48c689fbd...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 900 | 89041 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_00900/checkpoint_state.pt` | `67a743f0a5b06465...` | **trained but unevaluated** |
| QMIX | 5 | 456 | 55 | 1000 | 98888 | untracked | `artifacts/training/qmix_n5_v1/snapshots/episode_01000/checkpoint_state.pt` | `40495fe6e7e9fbe5...` | **trained but unevaluated** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/07f6f4cbf7104f3189b105cb26c93a3f/best/checkpoint_state.pt` | `38177c1f49fe2548...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/07f6f4cbf7104f3189b105cb26c93a3f/checkpoint_1/checkpoint_state.pt` | `38177c1f49fe2548...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/07f6f4cbf7104f3189b105cb26c93a3f/latest/checkpoint_state.pt` | `38177c1f49fe2548...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/0f83bb06e62546caaeef5c8c0f49023f/best/checkpoint_state.pt` | `cfd4ebad73875a5a...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/0f83bb06e62546caaeef5c8c0f49023f/checkpoint_5/checkpoint_state.pt` | `cfd4ebad73875a5a...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/0f83bb06e62546caaeef5c8c0f49023f/latest/checkpoint_state.pt` | `cfd4ebad73875a5a...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/139a8be38f7b4d56af39f06fd0dd370f/best/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/139a8be38f7b4d56af39f06fd0dd370f/checkpoint_5/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/139a8be38f7b4d56af39f06fd0dd370f/latest/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/4a27de20d7474ac78cd6a33a03de2347/best/checkpoint_state.pt` | `7f693ec73a8325fc...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/4a27de20d7474ac78cd6a33a03de2347/checkpoint_1/checkpoint_state.pt` | `7f693ec73a8325fc...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/4a27de20d7474ac78cd6a33a03de2347/latest/checkpoint_state.pt` | `7f693ec73a8325fc...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/4cbc0c98341f4885ae830bc037ee5c5e/best/checkpoint_state.pt` | `fe24472f1482789e...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/4cbc0c98341f4885ae830bc037ee5c5e/checkpoint_1/checkpoint_state.pt` | `fe24472f1482789e...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/4cbc0c98341f4885ae830bc037ee5c5e/latest/checkpoint_state.pt` | `fe24472f1482789e...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/52c7cececd31488c8699b4b85e85f3d2/best/checkpoint_state.pt` | `cfd4ebad73875a5a...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/52c7cececd31488c8699b4b85e85f3d2/checkpoint_5/checkpoint_state.pt` | `cfd4ebad73875a5a...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/52c7cececd31488c8699b4b85e85f3d2/latest/checkpoint_state.pt` | `cfd4ebad73875a5a...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/674d769582c5470eba4beb9d43acd1ba/best/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/674d769582c5470eba4beb9d43acd1ba/checkpoint_5/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/674d769582c5470eba4beb9d43acd1ba/latest/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/781d2a094a1d4c3284df3d756483f946/best/checkpoint_state.pt` | `9ba657472359d16e...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/781d2a094a1d4c3284df3d756483f946/checkpoint_1/checkpoint_state.pt` | `9ba657472359d16e...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/781d2a094a1d4c3284df3d756483f946/latest/checkpoint_state.pt` | `9ba657472359d16e...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/9bf13b048a124d81bcead2d9fcf6f0b3/best/checkpoint_state.pt` | `06f818e723179555...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/9bf13b048a124d81bcead2d9fcf6f0b3/checkpoint_1/checkpoint_state.pt` | `06f818e723179555...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/9bf13b048a124d81bcead2d9fcf6f0b3/latest/checkpoint_state.pt` | `06f818e723179555...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/b12fb29d4e9041f09aafac0b6235b2e4/best/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/b12fb29d4e9041f09aafac0b6235b2e4/checkpoint_5/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/b12fb29d4e9041f09aafac0b6235b2e4/latest/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/c2e73785c9b645c48b6e652be07147f8/best/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/c2e73785c9b645c48b6e652be07147f8/checkpoint_5/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/c2e73785c9b645c48b6e652be07147f8/latest/checkpoint_state.pt` | `b1420bb065401ad7...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: 'num_agents')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/f0419714129c4c788235cdde10bfe563/best/checkpoint_state.pt` | `c0a20dc1c537b4c2...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/f0419714129c4c788235cdde10bfe563/checkpoint_1/checkpoint_state.pt` | `c0a20dc1c537b4c2...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |
| QMIX (corrupt/unsupported) | unknown | unknown | unknown | unknown | 0 | none | `artifacts/checkpoints/qmix/f0419714129c4c788235cdde10bfe563/latest/checkpoint_state.pt` | `c0a20dc1c537b4c2...` | **unsupported (Checkpoint metadata is corrupt or missing required fields: int() argument must be a string, a bytes-like object or a real number, not 'NoneType')** |

## Status Taxonomy
- **`evaluated`**: Referenced in an existing benchmark manifest. See the benchmark for scenario provenance.
- **`trained but unevaluated`**: Positive optimizer counters with no matching benchmark reference.
- **`no optimizer updates verified`**: No positive training counter was recovered.

## Reproducibility and Validation Rules
1. Checkpoints with incompatible fleet sizes (N=5 vs N=10 vs N=20) or observation dimensions are strictly rejected by `checkpoint_validator.py`.
2. No checkpoint is silently reinitialized or partially loaded (`strict=True` is enforced across all loaders).
3. Models evaluated on held-out test scenarios are strictly selected using the separate validation set (`select_checkpoint.py`).