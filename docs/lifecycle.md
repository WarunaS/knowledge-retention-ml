# ML lifecycle -> this repo

| # | Stage | Code | Notebook |
|---|---|---|---|
| 1 | Define problem | `docs/01_problem.md` | - |
| 2 | Collect | `krs/collect/` | `02_collect` |
| 3 | Clean / prepare | `krs/clean/` | `03_clean` |
| 4 | Explore | `krs/explore.py` | `04_explore` |
| 4b | Label | `krs/label/` | `05_label` |
| 6 | Split (before features!) | `krs/split.py` | `06_split` |
| 5 | Feature engineering | `krs/features.py` | `07_features_baselines` |
| 7-8 | Choose / train | `krs/models/` | `07_…`, `08_train` |
| 9 | Evaluate | `krs/evaluate/` | `09_evaluate` |
| 10 | Tune (validation split only) | `krs/models/` | `08_train` |
| 11 | Track | `krs/tracking.py` | - |
| 12 | Register / version | `krs/registry.py` | `12_register` |
| 13 | Deploy | **skipped** | - |
| 14 | Inference (offline batch) | `krs/inference.py` | `14_inference` |
| 15-18 | Monitor / drift / retrain | X1 transfer test only: `krs/evaluate/transfer.py` | `16_transfer_test` |
| - | Application plane | `krs/app/` | later |

Rules: dedup before split · split by conversation · fit features on train only ·
tune on validation only · report on gold only · X1 runs once.
