#!/usr/bin/env python3
"""
tune_cdum.py — Optuna Hyperparameter Tuning Pipeline for CPM + VALOR + Dynamic Fusion.
====================================================================================

Objective:
    Find the peak validation performance of CPM + VALOR + Dynamic Fusion
    via a coarse search over router capacity, branch bottlenecks, and optimization.

Separation of Concerns:
    - Model:                  CPMDynamicFusion (tune ONLY the new model)
    - Tuning seeds:           [10, 11, 12]  (completely separate from final eval seeds 1-5)
    - Epoch / checkpoint:     argmin(val_loss) per seed
    - Hyperparameter goal:    maximize mean(val_auuc) across tuning seeds at best-val-loss checkpoints
    - Test set:               NEVER loaded, evaluated, or touched during Optuna tuning
    - Storage:                In-memory study only (NO SQLite / database)
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from CDUM.variants import CPMDynamicFusion
from CDUM.trainer import CPMTrainer
from preprocess.data_loader import get_dataloaders
from preprocess.cpm_processor import OfficialCPMBucketer
from experiment.main import BucketedDataLoader, set_seed
from experiment.cdum_search_space import SEARCH_SPACE, FIXED_PARAMS, sample_cdum_params

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tune_cdum")

FINAL_EVAL_SEEDS = {1, 2, 3, 4, 5}


def validate_seeds(seeds: List[int]) -> None:
    """Ensure tuning seeds NEVER overlap with final evaluation seeds 1-5."""
    overlap = set(seeds).intersection(FINAL_EVAL_SEEDS)
    if overlap:
        raise ValueError(
            f"Constraint violation: Tuning seeds {seeds} contain reserved final evaluation seeds: {sorted(overlap)}. "
            f"Optuna tuning MUST use completely separate seeds (default: [10, 11, 12])."
        )


class OptunaTrialRecorder:
    """Manages writing trial metadata and updating results/optuna/cdum_trials.csv."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        self.columns = [
            "trial_number",
            "state",
            "router_hidden_dim",
            "valor_hidden_dim",
            "expert_hidden_dim",
            "lr",
            "weight_decay",
            "seed10_best_epoch",
            "seed10_val_loss",
            "seed10_val_auuc",
            "seed11_best_epoch",
            "seed11_val_loss",
            "seed11_val_auuc",
            "seed12_best_epoch",
            "seed12_val_loss",
            "seed12_val_auuc",
            "completed_seeds",
            "mean_val_auuc",
            "std_val_auuc",
        ]
        if not os.path.exists(self.csv_path):
            df_init = pd.DataFrame(columns=self.columns)
            df_init.to_csv(self.csv_path, index=False)

    def record_trial(
        self,
        trial_number: int,
        state: str,
        params: Dict[str, Any],
        seed_records: Dict[int, Dict[str, Any]],
        mean_val_auuc: Optional[float],
        std_val_auuc: Optional[float],
    ) -> None:
        """Append or update trial row in CSV."""
        row_dict = {
            "trial_number": trial_number,
            "state": state,
            "router_hidden_dim": params.get("router_hidden_dim"),
            "valor_hidden_dim": params.get("valor_hidden_dim"),
            "expert_hidden_dim": params.get("expert_hidden_dim"),
            "lr": params.get("lr"),
            "weight_decay": params.get("weight_decay"),
            "completed_seeds": len(seed_records),
            "mean_val_auuc": round(mean_val_auuc, 6) if mean_val_auuc is not None else None,
            "std_val_auuc": round(std_val_auuc, 6) if std_val_auuc is not None else None,
        }

        # Populate per-seed columns for standard seeds 10, 11, 12
        for s in [10, 11, 12]:
            rec = seed_records.get(s, {})
            row_dict[f"seed{s}_best_epoch"] = rec.get("best_epoch")
            row_dict[f"seed{s}_val_loss"] = round(rec["val_loss"], 6) if "val_loss" in rec else None
            row_dict[f"seed{s}_val_auuc"] = round(rec["val_auuc"], 6) if "val_auuc" in rec else None

        # Also handle any arbitrary seed mapping if non-standard seeds were passed
        if set(seed_records.keys()) != {10, 11, 12} and len(seed_records) > 0:
            for i, (s_val, rec) in enumerate(sorted(seed_records.items())):
                prefix = f"seed{10 + i}"
                row_dict[f"{prefix}_best_epoch"] = rec.get("best_epoch")
                row_dict[f"{prefix}_val_loss"] = round(rec["val_loss"], 6) if "val_loss" in rec else None
                row_dict[f"{prefix}_val_auuc"] = round(rec["val_auuc"], 6) if "val_auuc" in rec else None

        df_new = pd.DataFrame([row_dict])
        if os.path.exists(self.csv_path):
            try:
                df_existing = pd.read_csv(self.csv_path)
                # Remove prior entry for same trial_number if present
                df_existing = df_existing[df_existing["trial_number"] != trial_number]
                if df_existing.empty:
                    df_combined = df_new
                else:
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(self.csv_path, index=False)
            except Exception:
                df_new.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.csv_path, index=False)


def run_tuning(args: argparse.Namespace) -> optuna.Study:
    """Run Optuna study for CPM + VALOR + Dynamic Fusion with explicit seed-level pruning."""
    validate_seeds(args.seeds)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("============================================================")
    logger.info("  OPTUNA CPM + VALOR + DYNAMIC FUSION TUNING (Criteo)")
    logger.info("============================================================")
    logger.info("Model:                     CPMDynamicFusion")
    logger.info("Tuning Seeds:              %s", args.seeds)
    logger.info("Device:                    %s", device)
    logger.info("Epochs per seed:           %d (Early stopping patience: %d)", args.epochs, args.early_stopping)
    logger.info("Monitor metric:            %s", args.monitor)
    logger.info("Batch size:                %d", args.batch_size)
    logger.info("Requested Trials:          %d", args.n_trials)
    logger.info("Pruner Startup Trials:     %d", args.startup_trials)
    logger.info("Output Directory:          %s", args.output_dir)
    logger.info("Max samples (subset):      %s", args.max_samples if args.max_samples else "Full Dataset")
    logger.info("============================================================")

    # ── 1. Load Data (TRAIN and VAL ONLY — NEVER TEST) ───────────────────────
    logger.info("[Data] Loading pre-split Criteo dataset (train & val only)...")
    train_path = os.path.join(args.data, "train_criteo.pt")
    val_path = os.path.join(args.data, "val_criteo.pt")
    if not os.path.exists(train_path):
        train_path = os.path.join(args.data, "train.pt")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.data, "val.pt")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=None,
        train_path=train_path,
        val_path=val_path,
        test_path=None,
        batch_size=args.batch_size,
        label_col="visit",
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seeds[0],
    )
    logger.info(
        "DataLoaders ready | train batches: %d | val batches: %d (Test set completely excluded)",
        len(train_loader),
        len(val_loader),
    )

    # ── 2. Official CPM Bucketer Preprocessing ────────────────────────────────
    logger.info("[Preprocessing] Fitting OfficialCPMBucketer on train continuous features...")
    bucketer = OfficialCPMBucketer(num_bins=101, dim=100)
    if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "X"):
        x_train_mat = train_loader.dataset.X
        if not isinstance(x_train_mat, torch.Tensor):
            x_train_mat = torch.as_tensor(x_train_mat)
        if x_train_mat.is_floating_point():
            bucketer.fit(x_train_mat)

    b_train_loader = BucketedDataLoader(train_loader, bucketer)
    b_val_loader = BucketedDataLoader(val_loader, bucketer)
    raw_train_loader = train_loader

    # ── 3. CSV Recorder & Checkpoint Directories ──────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "cdum_trials.csv")
    recorder = OptunaTrialRecorder(csv_path)

    # ── 4. Define Objective Function ──────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters from coarse search space
        params = sample_cdum_params(trial)
        router_hidden_dim = params["router_hidden_dim"]
        valor_hidden_dim = params["valor_hidden_dim"]
        expert_hidden_dim = params["expert_hidden_dim"]
        lr = params["lr"]
        weight_decay = params["weight_decay"]

        trial_dir = os.path.join(args.output_dir, "trials", f"trial_{trial.number:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        logger.info(
            "\n>>> [Trial %03d/%03d] Testing configuration (CPM + VALOR + Dynamic Fusion):\n"
            "    router_hidden_dim=%d | valor_hidden_dim=%d | expert_hidden_dim=%d | lr=%s | weight_decay=%s",
            trial.number,
            args.n_trials,
            router_hidden_dim,
            valor_hidden_dim,
            expert_hidden_dim,
            lr,
            weight_decay,
        )

        seed_scores: List[float] = []
        seed_records: Dict[int, Dict[str, Any]] = {}

        for seed_idx, seed in enumerate(args.seeds):
            seed_start_time = time.time()
            seed_dir = os.path.join(trial_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)

            # Ensure complete reproducibility for this seed
            set_seed(seed)
            if hasattr(raw_train_loader, "generator") and raw_train_loader.generator is not None:
                raw_train_loader.generator.manual_seed(seed)
            elif hasattr(raw_train_loader, "sampler") and hasattr(raw_train_loader.sampler, "generator"):
                gen = torch.Generator()
                gen.manual_seed(seed)
                raw_train_loader.sampler.generator = gen
                raw_train_loader.generator = gen

            # Construct CPMDynamicFusion with sampled parameters + frozen parameters
            model = CPMDynamicFusion(
                num_features=FIXED_PARAMS["num_features"],
                num_bins=FIXED_PARAMS["num_bins"],
                embedding_dim=FIXED_PARAMS["embedding_dim"],
                treatment_dim=FIXED_PARAMS["treatment_dim"],
                refine_hidden_dim=FIXED_PARAMS["refine_hidden_dim"],
                refine_dim=FIXED_PARAMS["refine_dim"],
                num_experts=FIXED_PARAMS["num_experts"],
                expert_hidden_dim=expert_hidden_dim,
                expert_dim=FIXED_PARAMS["expert_dim"],
                tower_hidden_dim=FIXED_PARAMS["tower_hidden_dim"],
                activation=FIXED_PARAMS["expert_activation"],
                dropout_rate=FIXED_PARAMS["dropout_rate"],
                use_bn=FIXED_PARAMS["use_bn"],
                router_hidden_dim=router_hidden_dim,
                valor_hidden_dim=valor_hidden_dim,
            )

            trainer = CPMTrainer(
                model=model,
                lr=lr,
                weight_decay=weight_decay,
                lr_factor=FIXED_PARAMS["lr_factor"],
                lr_patience=FIXED_PARAMS["lr_patience"],
                min_lr=FIXED_PARAMS["min_lr"],
                device=device,
            )
            if FIXED_PARAMS.get("huber_delta", 1.0) != 1.0:
                trainer.criterion = nn.HuberLoss(delta=FIXED_PARAMS["huber_delta"])

            # Checkpoint selection strictly by monitored validation metric (val_loss or val_auuc)
            history = trainer.fit(
                train_loader=b_train_loader,
                val_loader=b_val_loader,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping,
                checkpoint_dir=seed_dir,
                model_name="cpm_dynamic_fusion",
                monitor=args.monitor,
                verbose=0,
            )

            # Load the best checkpoint and evaluate validation AUUC
            best_ckpt_path = os.path.join(seed_dir, "cpm_dynamic_fusion_best.pth")
            if os.path.exists(best_ckpt_path):
                trainer.load(best_ckpt_path)

            val_loss, val_auuc = trainer.validate(b_val_loader)
            seed_elapsed = time.time() - seed_start_time

            best_epoch = history.get(
                "best_epoch",
                history.get("best_loss_epoch" if args.monitor == "val_loss" else "best_auuc_epoch", 1)
            )

            seed_scores.append(val_auuc)
            seed_records[seed] = {
                "best_epoch": int(best_epoch),
                "val_loss": float(val_loss),
                "val_auuc": float(val_auuc),
                "training_time": round(seed_elapsed, 2),
            }

            logger.info(
                "  [Trial %03d | Seed %2d] Best Epoch: %02d (by %s) | Val Loss: %.5f | Val AUUC: %.5f (Time: %.1fs)",
                trial.number,
                seed,
                best_epoch,
                args.monitor,
                val_loss,
                val_auuc,
                seed_elapsed,
            )

            # ── Pruning Logic ─────────────────────────────────────────────────
            # Rule 1: Never prune after seed 10 (first seed).
            # Rule 2: Pruning check happens ONLY after seed 11 (2 completed seeds).
            if len(seed_scores) == 2:
                partial_mean = float(np.mean(seed_scores))
                # Report partial mean at step 1
                trial.report(partial_mean, step=1)

                if trial.should_prune():
                    logger.warning(
                        "  ✂️  [Trial %03d] PRUNED after seed 11 with partial mean AUUC: %.5f",
                        trial.number,
                        partial_mean,
                    )
                    recorder.record_trial(
                        trial_number=trial.number,
                        state="PRUNED",
                        params=params,
                        seed_records=seed_records,
                        mean_val_auuc=partial_mean,
                        std_val_auuc=float(np.std(seed_scores)),
                    )
                    raise optuna.TrialPruned()

        # ── Trial Completed (All 3 Seeds) ─────────────────────────────────────
        final_mean = float(np.mean(seed_scores))
        final_std = float(np.std(seed_scores))

        logger.info(
            "  🎉 [Trial %03d COMPLETE] Mean Val AUUC: %.5f ± %.5f",
            trial.number,
            final_mean,
            final_std,
        )

        recorder.record_trial(
            trial_number=trial.number,
            state="COMPLETE",
            params=params,
            seed_records=seed_records,
            mean_val_auuc=final_mean,
            std_val_auuc=final_std,
        )

        trial.set_user_attr("std_val_auuc", final_std)
        trial.set_user_attr("seed_records", seed_records)
        return final_mean

    # ── 5. Create In-Memory Optuna Study ──────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=1,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=args.n_trials)

    # ── 6. Export Results & Summaries ─────────────────────────────────────────
    export_study_results(study, args)
    return study


def export_study_results(study: optuna.Study, args: argparse.Namespace) -> None:
    """Save cdum_best_config.json, cdum_study_summary.json and report Top 10 trials."""
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    best_trial = study.best_trial if complete_trials else None

    # ── Save cdum_best_config.json ────────────────────────────────────────────
    if best_trial is not None:
        best_config = {
            "model_type": "cpm_dynamic_fusion",
            "monitor_metric": args.monitor,
            "best_trial_number": best_trial.number,
            "best_params": best_trial.params,
            "mean_val_auuc": round(float(best_trial.value), 6),
            "std_val_auuc": round(float(best_trial.user_attrs.get("std_val_auuc", 0.0)), 6),
            "per_seed_metrics": best_trial.user_attrs.get("seed_records", {}),
            "tuning_seeds": args.seeds,
            "fixed_hyperparameters": FIXED_PARAMS,
            "fixed_architecture_decisions": {
                "expert_dim": FIXED_PARAMS["expert_dim"],
                "refine_hidden_dim": FIXED_PARAMS["refine_hidden_dim"],
                "refine_dim": FIXED_PARAMS["refine_dim"],
                "tower_hidden_dim": FIXED_PARAMS["tower_hidden_dim"],
                "huber_delta": FIXED_PARAMS["huber_delta"],
                "expert_activation": "relu",
                "expert_output_activation": "relu",
                "gate_bias": False,
                "output_activation": "softplus",
            },
            "search_space": SEARCH_SPACE,
            "study_name": study.study_name,
        }
        best_config_path = os.path.join(args.output_dir, "cdum_best_config.json")
        with open(best_config_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)
        logger.info("Saved best configuration to: %s", best_config_path)

    # ── Save cdum_study_summary.json ──────────────────────────────────────────
    summary_data = {
        "model_type": "cpm_dynamic_fusion",
        "monitor_metric": args.monitor,
        "study_name": study.study_name,
        "sampler": "TPESampler",
        "sampler_seed": args.sampler_seed,
        "pruner": f"MedianPruner(n_startup_trials={args.startup_trials}, n_warmup_steps=1, interval_steps=1)",
        "number_of_requested_trials": args.n_trials,
        "number_of_complete_trials": len(complete_trials),
        "number_of_pruned_trials": len(pruned_trials),
        "number_of_fail_trials": len(failed_trials),
        "best_trial_number": best_trial.number if best_trial else None,
        "best_value": round(float(best_trial.value), 6) if best_trial else None,
        "best_params": best_trial.params if best_trial else None,
        "tuning_seeds": args.seeds,
        "search_space": SEARCH_SPACE,
    }

    # Add top 10 ranked trials
    sorted_complete = sorted(complete_trials, key=lambda t: t.value, reverse=True)
    top_10 = []
    for rank, t in enumerate(sorted_complete[:10], start=1):
        top_10.append({
            "rank": rank,
            "trial_number": t.number,
            "mean_val_auuc": round(float(t.value), 6),
            "std_val_auuc": round(float(t.user_attrs.get("std_val_auuc", 0.0)), 6),
            "params": t.params,
        })
    summary_data["top_trials"] = top_10

    summary_path = os.path.join(args.output_dir, "cdum_study_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved study summary to: %s", summary_path)

    # ── Print Top 10 Report Table ─────────────────────────────────────────────
    print("\n" + "=" * 115)
    print("  OPTUNA STUDY COMPLETION REPORT — CPM + VALOR + DYNAMIC FUSION (Criteo)")
    print("=" * 115)
    print(f"Model:                      CPMDynamicFusion")
    print(f"Study Name:                 {study.study_name}")
    print(f"Total Trials:               {len(study.trials)}")
    print(f"  - COMPLETE Trials:        {len(complete_trials)}")
    print(f"  - PRUNED Trials:          {len(pruned_trials)}")
    print(f"  - FAIL Trials:            {len(failed_trials)}")
    print(f"Tuning Seeds:               {args.seeds}")
    print("-" * 115)

    if top_10:
        print(f"{'Rank':<5} {'Trial':<7} {'Mean Val AUUC':<16} {'Std Val AUUC':<15} {'Router Hidden':<15} {'VALOR Hidden':<15} {'Expert Hidden':<15} {'LR':<10} {'Weight Decay':<12}")
        print("-" * 115)
        for item in top_10:
            p = item["params"]
            print(
                f"{item['rank']:<5} {item['trial_number']:<7} {item['mean_val_auuc']:<16.5f} {item['std_val_auuc']:<15.5f} "
                f"{str(p.get('router_hidden_dim', '-')):<15} {str(p.get('valor_hidden_dim', '-')):<15} {str(p.get('expert_hidden_dim', '-')):<15} {str(p.get('lr', '-')):<10} {str(p.get('weight_decay', '-')):<12}"
            )
        print("-" * 115)
        print("STATISTICAL CAUTION:")
        print("  - If difference between top trials is smaller than seed std, the performance difference is not statistically meaningful.")
        print("  - Tuning uses 3 seeds ([10, 11, 12]). Final test evaluation will strictly use separate seeds ([1, 2, 3, 4, 5]).")
    print("=" * 115 + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tune_cdum.py",
        description="Optuna Hyperparameter Tuning for CPM + VALOR + Dynamic Fusion on Criteo Uplift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=30,
                   help="Number of Optuna trials to run (default: 30).")
    p.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12],
                   help="Tuning seeds to use for model selection (must NOT contain 1, 2, 3, 4, 5).")
    p.add_argument("--data", type=str, default="/home/ducvu0904/Documents/dataset/Criteo",
                   help="Path to pre-split Criteo directory.")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="results/optuna",
                   help="Directory to store Optuna CSV, JSON, and trial checkpoints.")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4096,
                   help="Batch size (fixed: 4096).")
    p.add_argument("--epochs", type=int, default=30,
                   help="Maximum training epochs per seed.")
    p.add_argument("--early-stopping", "--early_stopping", dest="early_stopping", type=int, default=5,
                   help="Early stopping patience monitoring the selected monitor metric.")
    p.add_argument("--monitor", "--monitor-metric", dest="monitor", type=str, default="val_loss",
                   choices=["val_loss", "val_auuc"],
                   help="Validation metric to monitor for early stopping and best checkpoint selection ('val_loss' or 'val_auuc').")
    p.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=None,
                   help="Optional sample limit for quick smoke testing / profiling.")
    p.add_argument("--device", type=str, default=None,
                   help="Device ('cuda', 'cpu', or null for auto-detect).")
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2,
                   help="Number of DataLoader workers.")
    p.add_argument("--study-name", "--study_name", dest="study_name", type=str, default="cpm_dynamic_fusion_coarse_3seed",
                   help="Name of in-memory Optuna study.")
    p.add_argument("--sampler-seed", "--sampler_seed", dest="sampler_seed", type=int, default=42,
                   help="Random seed for Optuna TPESampler.")
    p.add_argument("--startup-trials", "--startup_trials", dest="startup_trials", type=int, default=8,
                   help="Number of initial startup trials before MedianPruner activates.")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_tuning(args)
