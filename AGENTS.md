# AGENTS.md

Guidance and verified operational invariants for AI agents working in `CDUM_UPGRADE`.

---

## 1. Environment & Python Runtime

- **Active Conda Environment**: Use `ml_env` (`/home/ducvu0904/miniconda3/envs/ml_env/bin/python`).
  - Do **not** use default system/base `python` (`Python 3.13`); it lacks `yaml` (`PyYAML`), `optuna`, and other project dependencies.
  - To activate: `conda activate ml_env`
- **GPU & Determinism**:
  - Environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` is required for PyTorch deterministic operations and is pre-set in scripts.
  - PyTorch version: `2.11.0+cu130` with CUDA enabled.

---

## 2. Developer Commands

### Smoke & Unit Tests
Run tests using `ml_env`:
```bash
# 1. Core CPM unit test suite (encoder, refine, experts, towers, trainer, discretization)
python CDUM/smoke_test.py

# 2. Dynamic fusion variant smoke test (shapes, algebra, tower hooks, grad isolation)
python CDUM/smoke_test_dynamic_fusion.py

# 3. Integration test suite (CLI overrides, YAML config merging, model dispatch, evaluate)
python experiment/smoke_test_dynamic_fusion.py
```

### Training & Evaluation
```bash
# Full multi-seed run via config (seeds 1..5)
python experiment/main.py --config experiment/config.yaml

# Fast debug run (subsample data, 1 epoch, single seed)
python experiment/main.py --config experiment/config.yaml --max_samples 1000 --epochs 1 --seed 1 --results_dir /tmp/debug_results

# Explicit model selection ('cdum' [or alias 'cpm'], 'cpm_dynamic_fusion')
python experiment/main.py --model cpm_dynamic_fusion --router_hidden_dim 64 --seeds 1 2 3

# Standalone evaluation on test set across seeds
python experiment/evaluate.py --config experiment/config.yaml --all_seeds --checkpoint_type best_auuc

# Evaluate a specific single checkpoint
python experiment/evaluate.py --checkpoint results/cdum/seed_1/best_checkpoint.pth --model cdum
```

### Hyperparameter Tuning (Optuna)
```bash
# Optuna tuning across tuning seeds [10, 11, 12]
python experiment/tune_cdum.py --n-trials 30 --seeds 10 11 12

# Fast smoke run for tuning
python experiment/tune_cdum.py --n-trials 2 --max-samples 2000 --epochs 2 --output-dir /tmp/tune_debug
```

### Checkpoint Representation Inspection
```bash
# Inspect gate weights, representation differences, and indicators without retraining
python CDUM/inspect_checkpoint.py --config results/cdum/config.json --results_dir results/cdum --seeds 1 2 3 4 5
```

---

## 3. Architecture & Operational Invariants

- **Hadamard Mask Constraint**: `tower_hidden_dim` **must equal** `refine_dim` (default `32`). `TreatmentTower` computes element-wise product `h * e_ind`, which fails if dimensions diverge.
- **Treatment Embedding Scaling**: `treatment_dim` defaults to `embedding_dim * 4` (i.e. `32 * 4 = 128`).
- **VALOR Branch Invariant (`cpm_dynamic_fusion`)**: The VALOR interaction branch receives raw pre-refine treatment embedding `e_t` (`Linear(e_t)`), **never** `e_guidance` or `e_indicator`.
- **Factual-Outcome Loss**: `CPMTrainer` optimizes **only** factual Huber loss (`y_factual` vs observed outcome `y`). Potential outcomes (`y0`, `y1`) and uplift (`y1 - y0`) are calculated solely for evaluation/AUUC metrics, never directly backpropagated.
- **LR Scheduling**: `ReduceLROnPlateau` always steps on `val_loss`, even when `monitor_metric` is `val_auuc`.
- **Optuna Tuning Seed Separation**:
  - Tuning seeds (`[10, 11, 12]`) must **never overlap** with final evaluation seeds (`[1, 2, 3, 4, 5]`). `tune_cdum.py` validates this and raises an error on overlap.
  - Tuning uses `train` and `val` splits only; test split is never touched during Optuna trials.

---

## 4. Data & Preprocessing Gotchas

- **Data Directory**: Default is `/home/ducvu0904/Documents/dataset/Criteo`.
  - Binary tensor splits (`train_criteo.pt`, `val_criteo.pt`, `test_criteo.pt`) are automatically detected and memory-mapped (`mmap=True`), taking priority over `.csv`.
- **Equidistant Discretization (`OfficialCPMBucketer`)**:
  - Continuous features `f0..f11` are bucketed to `[0, 100]` (101 bins): `scaled = x_j / denominator_j * 100`.
  - **Rule**: Denominators `denominator_j = max(1e-8, max(train_feature_j))` must be computed **only** on the training split. Never fit on val or test splits.
  - `prepare_loaders_for_model` wraps raw loaders with `BucketedDataLoader` for models requiring integer bucket IDs (`cdum`, `cpm`, `cpm_dynamic_fusion`).

---

## 5. Configuration & Result Artifacts

- **Config Precedence**: CLI explicit options > YAML config (`experiment/config.yaml`) > Parser defaults.
  - Passing an option via CLI overrides YAML even if its value equals the argument parser default.
- **Artifact Locations**:
  - Training runs output to `results/<model>/`:
    - `seed_<seed>/best_checkpoint.pth` (matches `--monitor_metric`)
    - `seed_<seed>/best_auuc_checkpoint.pth`
    - `seed_<seed>/best_loss_checkpoint.pth`
    - `seed_<seed>/last_checkpoint.pth`
    - `config.json` (saved experiment hyperparameters)
    - `metrics.json` (per-seed metrics and mean ± std)
    - `results/summary.csv` (aggregated table across models; thread-safe via `filelock`).
  - Optuna runs output to `results/optuna/` (`cdum_trials.csv`, `cdum_best_config.json`, `cdum_study_summary.json`).
