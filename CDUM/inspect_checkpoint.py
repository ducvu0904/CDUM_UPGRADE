"""Inspect CDUM/CPM checkpoint representations and difference metrics without retraining."""

import argparse
import json
import os
import sys
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from CDUM.cpm import CPM


def load_model(config_path: str) -> CPM:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = CPM(
        num_features=cfg.get("cpm_num_features", 12),
        num_bins=cfg.get("cpm_num_bins", 101),
        embedding_dim=cfg.get("cpm_embedding_dim", 32),
        treatment_dim=cfg.get("cpm_treatment_dim", 128),
        refine_hidden_dim=cfg.get("cpm_refine_hidden_dim", 64),
        refine_dim=cfg.get("cpm_refine_dim", 32),
        num_experts=cfg.get("cpm_num_experts", 3),
        expert_hidden_dim=cfg.get("cpm_expert_hidden_dim", 128),
        expert_dim=cfg.get("cpm_expert_dim", 64),
        tower_hidden_dim=cfg.get("cpm_tower_hidden_dim", 32),
        activation=cfg.get("cpm_activation", "relu"),
        dropout_rate=cfg.get("cpm_dropout", 0.0),
        use_bn=cfg.get("cpm_batch_norm", False),
    )
    return model


def inspect_checkpoint(model: CPM, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        t0 = torch.tensor([0], dtype=torch.long)
        t1 = torch.tensor([1], dtype=torch.long)

        t_emb0 = model.encoder.encode_treatment(t0)
        t_emb1 = model.encoder.encode_treatment(t1)

        e_gui0, e_ind0 = model.treatment_refine(t_emb0)
        e_gui1, e_ind1 = model.treatment_refine(t_emb1)

        g0 = model.control_gate(e_gui0)
        g1 = model.treatment_gate(e_gui1)

        t_diff = (t_emb0 - t_emb1).abs().mean().item()
        gui_diff = (e_gui0 - e_gui1).abs().mean().item()
        ind_diff = (e_ind0 - e_ind1).abs().mean().item()
        gate_diff = (g0 - g1).abs().mean().item()

    return {
        "treatment_emb_diff": t_diff,
        "guidance_diff": gui_diff,
        "indicator_diff": ind_diff,
        "gate_diff": gate_diff,
        "g0": g0.squeeze().tolist(),
        "g1": g1.squeeze().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect CDUM checkpoint diagnostics without retraining.")
    parser.add_argument("--config", type=str, default="results/cdum/config.json", help="Path to config.json")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint path (.pth)")
    parser.add_argument("--results_dir", type=str, default="results/cdum", help="Directory containing seed checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seed list to evaluate")
    args = parser.parse_args()

    model = load_model(args.config)

    if args.ckpt is not None:
        print(f"\n--- Evaluating checkpoint: {args.ckpt} ---")
        metrics = inspect_checkpoint(model, args.ckpt)
        print(f"treatment emb diff: {metrics['treatment_emb_diff']:.6f}")
        print(f"guidance diff:      {metrics['guidance_diff']:.6f}")
        print(f"indicator diff:     {metrics['indicator_diff']:.6f}")
        print(f"gate diff:          {metrics['gate_diff']:.6f}")
        print(f"g0 (control gate):  {metrics['g0']}")
        print(f"g1 (treat gate):    {metrics['g1']}")
        return

    # Evaluate across all seeds
    print("\n" + "=" * 75)
    print(" CDUM CHECKPOINT DIAGNOSTICS (No Retraining Needed)")
    print("=" * 75)

    all_metrics = {"treatment_emb_diff": [], "guidance_diff": [], "indicator_diff": [], "gate_diff": []}

    for seed in args.seeds:
        ckpt_path = os.path.join(args.results_dir, f"seed_{seed}", "best_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            print(f"\n[Warning] Checkpoint not found: {ckpt_path}")
            continue

        res = inspect_checkpoint(model, ckpt_path)
        for k in all_metrics:
            all_metrics[k].append(res[k])

        print(f"\n[Seed {seed}] ({ckpt_path})")
        print(f"  treatment emb diff: {res['treatment_emb_diff']:.6f}")
        print(f"  guidance diff:      {res['guidance_diff']:.6f}")
        print(f"  indicator diff:     {res['indicator_diff']:.6f}")
        print(f"  gate diff:          {res['gate_diff']:.6f}")
        print(f"  g0 (control gate):  {[round(x, 4) for x in res['g0']]}")
        print(f"  g1 (treat gate):    {[round(x, 4) for x in res['g1']]}")

    print("\n" + "=" * 75)
    print(f" SUMMARY ({len(all_metrics['gate_diff'])} Seeds)")
    print("=" * 75)
    for k, vals in all_metrics.items():
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"  {k:20s}: {mean:.6f} ± {std:.6f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
