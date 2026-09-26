#!/usr/bin/env python3
"""
evaluate.py — Evaluate saved model checkpoints on the Criteo test set
====================================================================
Usage:
  # Evaluate a specific seed:
  python experiment/evaluate.py --seed 1

  # Evaluate all seeds (1-5) and print a summary table:
  python experiment/evaluate.py --all_seeds

  # Evaluate an explicit checkpoint file:
  python experiment/evaluate.py --checkpoint results/cdum/seed_1/best_auuc_checkpoint.pth
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocess.data_loader import get_dataloaders
from experiment.main import (
    build_parser,
    build_model,
    merge_config_into_args,
    prepare_loaders_for_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved model checkpoints on Criteo test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="experiment/config.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default=None, choices=["cdum", "cpm", "cpm_dynamic_fusion"],
                        help="Model to evaluate ('cdum', 'cpm', or 'cpm_dynamic_fusion'). Defaults to config model.name.")
    parser.add_argument("--router_hidden_dim", type=int, default=None,
                        help="Router hidden dimension for cpm_dynamic_fusion (default: from config or model default).")
    parser.add_argument("--valor_hidden_dim", type=int, default=None,
                        help="VALOR hidden dimension for cpm_dynamic_fusion (default: from config or model default).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit path to a .pth checkpoint file. If omitted, resolves from results_dir.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed to evaluate when neither --seeds nor --all_seeds is specified.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of seeds to evaluate (e.g. --seeds 1 2 3 4 5).")
    parser.add_argument("--all_seeds", action="store_true", default=False,
                        help="Evaluate all available seeds (1 to 5) and display summary.")
    parser.add_argument("--checkpoint_type", type=str, default="best_auuc",
                        choices=["best_auuc", "best", "best_loss", "last"],
                        help="Which checkpoint type to load if --checkpoint is not given.")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for evaluation dataloader.")
    parser.add_argument("--eval_k", type=float, default=0.3,
                        help="Top-k fraction for Uplift@k (default: 0.3 = 30%%).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu'). Auto-detect by default.")
    parser.add_argument("--save_csv", type=str, default="results/evaluation_summary.csv",
                        help="Path to save evaluation summary CSV.")
    parser.add_argument("--skip_missing", action="store_true", default=True,
                        help="Skip missing checkpoints instead of aborting execution.")
    return parser.parse_args()


def resolve_checkpoint_path(results_dir: str, model_name: str, seed: int, ckpt_type: str) -> str:
    seed_dir = os.path.join(results_dir, model_name.lower(), f"seed_{seed}")
    type_map = {
        "best_auuc": [
            "best_auuc_checkpoint.pth",
            "best_auuc_checkpoint.pt",
            f"{model_name.lower()}_best_auuc.pth",
            f"{model_name.lower()}_best_auuc.pt",
            "best_checkpoint.pth",
            "best_checkpoint.pt",
            f"{model_name.lower()}_best.pth",
            f"{model_name.lower()}_best.pt",
        ],
        "best": [
            "best_checkpoint.pth",
            "best_checkpoint.pt",
            f"{model_name.lower()}_best.pth",
            f"{model_name.lower()}_best.pt",
        ],
        "best_loss": [
            "best_loss_checkpoint.pth",
            "best_loss_checkpoint.pt",
            f"{model_name.lower()}_best_loss.pth",
            f"{model_name.lower()}_best_loss.pt",
        ],
        "last": [
            "last_checkpoint.pth",
            "last_checkpoint.pt",
            f"{model_name.lower()}_final.pth",
            f"{model_name.lower()}_final.pt",
        ],
    }
    candidates = type_map.get(
        ckpt_type,
        [f"{ckpt_type}_checkpoint.pth", f"{ckpt_type}.pth", f"{ckpt_type}_checkpoint.pt", f"{ckpt_type}.pt"]
    )
    for name in candidates:
        p = os.path.join(seed_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found for {model_name} seed {seed} in {seed_dir}. Checked: {candidates}"
    )


def evaluate_single_checkpoint(
    model_name: str,
    ckpt_path: str,
    seed_label: str,
    test_loader: Any,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Build model, load weights, and run evaluation on test_loader."""
    logger.info("=" * 70)
    logger.info(f"EVALUATING {model_name.upper()} [Seed: {seed_label}]")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info("=" * 70)

    # Instantiate model & trainer
    args.model = model_name.lower()
    trainer = build_model(args.model, args)
    trainer.load(ckpt_path)

    eval_kwargs: Dict[str, Any] = {"k": args.eval_k}
    if model_name.lower() in ("cdum", "cpm", "cpm_dynamic_fusion"):
        eval_kwargs["print_diagnostics"] = True

    try:
        metrics = trainer.evaluate(test_loader, **eval_kwargs)
    except TypeError:
        metrics = trainer.evaluate(test_loader, k=args.eval_k)

    k_pct = int(args.eval_k * 100)
    lift_val = metrics.get(f"lift@{k_pct}%", metrics.get("lift@30%", 0.0))
    logger.info(
        f"[{model_name.upper()} Seed {seed_label}] -> "
        f"Loss: {metrics.get('loss', 0):.5f} | "
        f"AUUC: {metrics.get('auuc', 0):.5f} | "
        f"Qini: {metrics.get('qini', 0):.5f} | "
        f"Lift@{k_pct}%: {lift_val:.5f}"
    )
    return metrics


def main():
    cli_args = parse_args()

    # Build full args from main parser and config.yaml
    base_parser = build_parser()
    args = base_parser.parse_args([])
    if cli_args.config and os.path.exists(cli_args.config):
        args = merge_config_into_args(args, cli_args.config)

    # CLI overrides
    if cli_args.batch_size:
        args.batch_size = cli_args.batch_size
    if cli_args.eval_k:
        args.eval_k = cli_args.eval_k
    if cli_args.device:
        args.device = cli_args.device
    elif args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cli_args.model is not None:
        args.model = cli_args.model
    if cli_args.router_hidden_dim is not None:
        args.router_hidden_dim = cli_args.router_hidden_dim
    if cli_args.valor_hidden_dim is not None:
        args.valor_hidden_dim = cli_args.valor_hidden_dim

    raw_model = (getattr(args, "model", None) or "cdum").lower()
    model_name = "cdum" if raw_model in ("cdum", "cpm") else raw_model
    args.model = model_name

    # Determine seeds to evaluate
    if cli_args.checkpoint:
        seeds_list = None
    elif cli_args.seeds is not None:
        seeds_list = cli_args.seeds
    elif cli_args.all_seeds:
        seeds_list = args.seeds if isinstance(args.seeds, (list, tuple)) else [1, 2, 3, 4, 5]
    else:
        seeds_list = [cli_args.seed]

    logger.info(f"Target Model:     {model_name.upper()}")
    logger.info(f"Target Seeds:     {seeds_list if seeds_list is not None else [cli_args.checkpoint]}")
    logger.info(f"Checkpoint Type:  {cli_args.checkpoint_type}")

    # Load DataLoaders (loaded once for efficiency)
    logger.info("Loading Criteo DataLoaders...")
    train_loader, val_loader, raw_test_loader = get_dataloaders(
        data_dir=args.data,
        train_path=getattr(args, "train_path", None),
        val_path=getattr(args, "val_path", None),
        test_path=getattr(args, "test_path", None),
        batch_size=args.batch_size,
        label_col=getattr(args, "label_col", "visit"),
        num_workers=getattr(args, "num_workers", 2),
    )

    logger.info(f"Preparing EquidistantBucketer for {model_name.upper()}...")
    _, _, eval_test_loader = prepare_loaders_for_model(
        model_name, args, train_loader, val_loader, raw_test_loader
    )

    csv_rows = []
    k_pct = int(args.eval_k * 100)

    if cli_args.checkpoint:
        seeds_to_eval = [("custom", cli_args.checkpoint)]
    else:
        seeds_to_eval = []
        for s in seeds_list:
            try:
                p = resolve_checkpoint_path(args.results_dir, model_name, s, cli_args.checkpoint_type)
                seeds_to_eval.append((str(s), p))
            except FileNotFoundError as e:
                if cli_args.skip_missing:
                    logger.warning(f"Skipping {model_name.upper()} seed {s}: checkpoint not found.")
                else:
                    raise e

    if not seeds_to_eval:
        logger.warning(f"No valid {model_name.upper()} checkpoints found.")
        return

    model_results: Dict[str, Dict[str, float]] = {}
    for seed_label, ckpt_path in seeds_to_eval:
        try:
            metrics = evaluate_single_checkpoint(
                model_name=model_name,
                ckpt_path=ckpt_path,
                seed_label=seed_label,
                test_loader=eval_test_loader,
                args=args,
            )
            model_results[seed_label] = metrics
            lift_val = metrics.get(f"lift@{k_pct}%", metrics.get("lift@30%", 0.0))
            csv_rows.append({
                "model": model_name,
                "seed": seed_label,
                "checkpoint_type": cli_args.checkpoint_type,
                "checkpoint_path": ckpt_path,
                "loss": metrics.get("loss", np.nan),
                "auuc": metrics.get("auuc", np.nan),
                "qini": metrics.get("qini", np.nan),
                f"lift@{k_pct}%": lift_val,
            })
        except Exception as e:
            logger.error(f"Error evaluating {model_name.upper()} [Seed {seed_label}]: {e}", exc_info=True)

    if len(model_results) > 1:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"MODEL SUMMARY: {model_name.upper()} ({len(model_results)} Seeds evaluated)")
        logger.info("=" * 80)
        logger.info(f"{'Seed':<8} {'Test Loss':<15} {'Test AUUC':<15} {'Test Qini':<15} {f'Test Lift@{k_pct}%':<15}")
        logger.info("-" * 80)
        for s_lbl, m in model_results.items():
            l_val = m.get(f"lift@{k_pct}%", m.get("lift@30%", 0.0))
            logger.info(
                f"{s_lbl:<8} {m.get('loss', 0):<15.5f} {m.get('auuc', 0):<15.5f} {m.get('qini', 0):<15.5f} {l_val:<15.5f}"
            )
        logger.info("-" * 80)
        for k in ["loss", "auuc", "qini", f"lift@{k_pct}%"]:
            vals = [m[k] for m in model_results.values() if k in m and not np.isnan(m[k])]
            if vals:
                mean, std = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                logger.info(f"Mean ± Std [{k:12s}]: {mean:.5f} ± {std:.5f}")
        logger.info("=" * 80)
        logger.info("")

    # Save to CSV
    if csv_rows and cli_args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(cli_args.save_csv)), exist_ok=True)
        df = pd.DataFrame(csv_rows)
        df.to_csv(cli_args.save_csv, index=False)
        logger.info(f"Evaluation results successfully saved to: {cli_args.save_csv}")


if __name__ == "__main__":
    main()
