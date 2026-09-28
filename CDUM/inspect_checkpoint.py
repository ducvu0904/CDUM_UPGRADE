"""Inspect CPM and two-/three-branch fusion checkpoints without retraining."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from CDUM.cpm import CPM
from CDUM.variants import DRFU, TwoBranchDynamicFusion


def detect_and_load_model(config_path: Optional[str] = None, state_dict: Optional[dict] = None) -> torch.nn.Module:
    """Load model from config.json or deduce from state_dict keys."""
    cfg: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Check model architecture type
    is_variant = False
    model_name = str(cfg.get("model", cfg.get("model_name", ""))).lower()
    is_three_branch = model_name in ("drfu", "cpm_three_branch_dynamic_fusion")
    if state_dict is not None:
        is_three_branch = is_three_branch or any(k.startswith("prognostic_branch.") for k in state_dict)
    if is_three_branch or model_name in (
        "two_branch_dynamic_fusion", "cpm_dynamic_fusion", "dynamic_fusion"
    ):
        is_variant = True
    elif state_dict is not None:
        if any(
            k.startswith(("treatment_interaction.", "valor_branch.", "router."))
            for k in state_dict.keys()
        ):
            is_variant = True

    # Shared kwargs
    model_kwargs = {
        "num_features": cfg.get("cpm_num_features") or cfg.get("input_dim", 12),
        "num_bins": cfg.get("cpm_num_bins", 101),
        "embedding_dim": cfg.get("cpm_embedding_dim", 32),
        "treatment_dim": cfg.get("cpm_treatment_dim", 128),
        "refine_hidden_dim": cfg.get("cpm_refine_hidden_dim", 64),
        "refine_dim": cfg.get("cpm_refine_dim", 32),
        "num_experts": cfg.get("cpm_num_experts", 3),
        "expert_hidden_dim": cfg.get("cpm_expert_hidden_dim", 64),
        "expert_dim": cfg.get("cpm_expert_dim", 64),
        "tower_hidden_dim": cfg.get("cpm_tower_hidden_dim", 32),
        "activation": cfg.get("cpm_activation", "relu"),
        "dropout_rate": cfg.get("cpm_dropout", 0.0),
        "use_bn": cfg.get("cpm_batch_norm", False),
    }

    if is_variant:
        model_kwargs["router_hidden_dim"] = cfg.get("router_hidden_dim", None)
        model_kwargs["interaction_hidden_dim"] = cfg.get(
            "interaction_hidden_dim", cfg.get("valor_hidden_dim", None)
        )
        if is_three_branch:
            model_kwargs["prognostic_hidden_dim"] = cfg.get("prognostic_hidden_dim", None)
            # Without a config, tensor shapes reconstruct dimensions (activation
            # remains the default ReLU; use the saved config for other settings).
            if not cfg and state_dict is not None:
                model_kwargs.update({
                    "num_features": sum(k.startswith("encoder.feature_embeddings.") for k in state_dict),
                    "num_bins": state_dict["encoder.feature_embeddings.0.weight"].shape[0],
                    "embedding_dim": state_dict["encoder.feature_embeddings.0.weight"].shape[1],
                    "treatment_dim": state_dict["encoder.treatment_embeddings.weight"].shape[1],
                    "refine_hidden_dim": state_dict["treatment_refine.guidance_hidden.weight"].shape[0],
                    "refine_dim": state_dict["treatment_refine.guidance_output.weight"].shape[0],
                    "tower_hidden_dim": state_dict["control_tower.layer1.weight"].shape[0],
                    "num_experts": state_dict["control_gate.gate.weight"].shape[0],
                    "expert_hidden_dim": state_dict["user_experts.experts.0.hidden.weight"].shape[0],
                    "expert_dim": state_dict["user_experts.experts.0.output.weight"].shape[0],
                    "router_hidden_dim": state_dict["router.fc1.weight"].shape[0],
                    "interaction_hidden_dim": state_dict[
                        "treatment_interaction.mlp.hidden.weight"
                        if "treatment_interaction.mlp.hidden.weight" in state_dict
                        else "valor_branch.mlp.hidden.weight"
                    ].shape[0],
                    "prognostic_hidden_dim": state_dict["prognostic_branch.mlp.hidden.weight"].shape[0],
                })
            return DRFU(**model_kwargs)
        return TwoBranchDynamicFusion(**model_kwargs)
    return CPM(**model_kwargs)


def inspect_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    sample_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Inspect guidance, indicator, treatment-interaction, and router gates."""
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    is_variant = isinstance(model, TwoBranchDynamicFusion) or (
        hasattr(model, "treatment_interaction") and hasattr(model, "router")
    )

    with torch.no_grad():
        # ── 1. Static Treatment Representation & Guidance / Indicator Gates ──
        t0 = torch.tensor([0], dtype=torch.long)
        t1 = torch.tensor([1], dtype=torch.long)

        t_emb0 = model.encoder.encode_treatment(t0)
        t_emb1 = model.encoder.encode_treatment(t1)

        e_gui0, e_ind0 = model.treatment_refine(t_emb0)
        e_gui1, e_ind1 = model.treatment_refine(t_emb1)

        g0 = model.control_gate(e_gui0).squeeze(0)
        g1 = model.treatment_gate(e_gui1).squeeze(0)

        t_diff = (t_emb0 - t_emb1).abs().mean().item()
        gui_diff = (e_gui0 - e_gui1).abs().mean().item()
        ind_diff = (e_ind0 - e_ind1).abs().mean().item()
        gate_diff = (g0 - g1).abs().mean().item()

        metrics: Dict[str, Any] = {
            "model_type": type(model).__name__,
            "treatment_emb_diff": t_diff,
            "guidance_diff": gui_diff,
            "indicator_diff": ind_diff,
            "gate_diff": gate_diff,
            "g0": g0.tolist(),
            "g1": g1.tolist(),
            "ind0_mean": e_ind0.mean().item(),
            "ind1_mean": e_ind1.mean().item(),
        }

        # ── 2. Treatment-Interaction Gating Mask & Dynamic Router ────────────
        if is_variant:
            # 2a. Treatment-Interaction Gating Mask m_t = Sigmoid(Linear_t(e_t))
            m0 = torch.sigmoid(model.treatment_interaction.linear_t(t_emb0)).squeeze(0)
            m1 = torch.sigmoid(model.treatment_interaction.linear_t(t_emb1)).squeeze(0)
            m_abs_diff = (m0 - m1).abs()
            top_diff_indices = torch.topk(m_abs_diff, k=min(5, m_abs_diff.shape[0])).indices.tolist()

            metrics.update({
                "interaction_mask_diff_mean": m_abs_diff.mean().item(),
                "interaction_mask_diff_max": m_abs_diff.max().item(),
                "interaction_m0_mean": m0.mean().item(),
                "interaction_m1_mean": m1.mean().item(),
                "interaction_m0_min": m0.min().item(),
                "interaction_m0_max": m0.max().item(),
                "interaction_m1_min": m1.min().item(),
                "interaction_m1_max": m1.max().item(),
                "interaction_top_diff_dims": [
                    {"dim": int(idx), "m0": round(m0[idx].item(), 4), "m1": round(m1[idx].item(), 4), "diff": round(m_abs_diff[idx].item(), 4)}
                    for idx in top_diff_indices
                ],
            })

            # 2b. Router Static Intrinsic Bias / Prior
            branch_names = getattr(model.router, "branch_names", ("C", "I"))
            if hasattr(model.router, "fc2") and model.router.fc2.bias is not None:
                bias = model.router.fc2.bias
                prior = torch.softmax(bias, dim=-1)
                for i, name in enumerate(branch_names):
                    metrics[f"router_bias_{name}"] = bias[i].item()
                    metrics[f"router_prior_{name}"] = prior[i].item()
            elif hasattr(model.router, "fc2"):
                metrics["router_logit_bias_free"] = True
                zero_q = torch.zeros(1, model.router.fc1.in_features)
                zero_logits = model.router.fc2(model.router.relu(model.router.fc1(zero_q)))
                zero_weights = torch.softmax(zero_logits, dim=-1).squeeze(0)
                for i, name in enumerate(branch_names):
                    metrics[f"router_zero_input_{name}"] = zero_weights[i].item()

            # ── 3. Data-Dependent Dynamic Router & Expert Inspection ──────────
            if sample_batch is not None:
                x_ids, t, outcome = sample_batch[:3]
                diag_out = model(x_ids, t, return_diagnostics=True)
                diag0 = diag_out["diagnostics"][0]
                diag1 = diag_out["diagnostics"][1]

                pi0 = diag0["pi"]  # [B, number of representation branches]
                pi1 = diag1["pi"]

                entropy0 = -(pi0 * torch.log(pi0.clamp(min=1e-12))).sum(dim=-1).mean().item()
                entropy1 = -(pi1 * torch.log(pi1.clamp(min=1e-12))).sum(dim=-1).mean().item()

                metrics.update({
                    "data_evaluated": True,
                    "sample_size": x_ids.shape[0],
                    "router_ctrl_entropy": entropy0,
                    "router_treat_entropy": entropy1,
                })
                for i, name in enumerate(branch_names):
                    for label, diag in (("ctrl", diag0), ("treat", diag1)):
                        metrics[f"router_{label}_pi_{name}"] = diag["pi"][:, i].mean().item()
                        metrics[f"z_{name}_{label}_norm"] = diag[f"z_{name}"].norm(dim=-1).mean().item()
                if "z_P" in diag0:
                    metrics["prognostic_candidate_diff"] = (diag0["z_P"] - diag1["z_P"]).abs().max().item()

    return metrics


def print_metrics(res: Dict[str, Any], prefix: str = "  ") -> None:
    """Format and print gate metrics."""
    print(f"{prefix}[Guidance Gate & Refinement]")
    print(f"{prefix}  treatment emb diff : {res['treatment_emb_diff']:.6f}")
    print(f"{prefix}  guidance diff      : {res['guidance_diff']:.6f}")
    print(f"{prefix}  indicator diff     : {res['indicator_diff']:.6f} (ind0 mean: {res['ind0_mean']:.4f}, ind1 mean: {res['ind1_mean']:.4f})")
    print(f"{prefix}  gate diff          : {res['gate_diff']:.6f}")
    print(f"{prefix}  g0 (control gate)  : {[round(x, 4) for x in res['g0']]}")
    print(f"{prefix}  g1 (treat gate)    : {[round(x, 4) for x in res['g1']]}")

    if res.get("model_type") in ("TwoBranchDynamicFusion", "DRFU"):
        print(f"{prefix}[Treatment-Interaction Gating Mask (m_t = Sigmoid(Linear(e_t)))]")
        print(f"{prefix}  interaction mask diff    : mean={res['interaction_mask_diff_mean']:.6f}, max={res['interaction_mask_diff_max']:.6f}")
        print(f"{prefix}  mask m0 (control)  : mean={res['interaction_m0_mean']:.4f}, min={res['interaction_m0_min']:.4f}, max={res['interaction_m0_max']:.4f}")
        print(f"{prefix}  mask m1 (treat)    : mean={res['interaction_m1_mean']:.4f}, min={res['interaction_m1_min']:.4f}, max={res['interaction_m1_max']:.4f}")
        top_dims_str = ", ".join(f"d{item['dim']}: |{item['m0']:.2f}-{item['m1']:.2f}|={item['diff']:.2f}" for item in res.get("interaction_top_diff_dims", []))
        print(f"{prefix}  top diff channels  : {top_dims_str}")

        if res["model_type"] == "DRFU":
            names = ("P", "C", "I")
            print(f"{prefix}[Dynamic Fusion Router: P=prognostic, C=CPM, I=interaction]")
            if res.get("router_logit_bias_free"):
                baseline = ", ".join(f"pi_{n}={res[f'router_zero_input_{n}']:.4f}" for n in names)
                print(f"{prefix}  final logits       : fc2 bias=False")
                print(f"{prefix}  zero-input output   : {baseline}")
            if res.get("data_evaluated"):
                for label in ("ctrl", "treat"):
                    weights = ", ".join(f"pi_{n}={res[f'router_{label}_pi_{n}']:.4f}" for n in names)
                    norms = ", ".join(f"||z_{n}||={res[f'z_{n}_{label}_norm']:.2f}" for n in names)
                    print(f"{prefix}  {label:5s}: {weights}, H={res[f'router_{label}_entropy']:.4f} ({norms})")
                print(f"{prefix}  shared z_P max diff: {res['prognostic_candidate_diff']:.6f}")
            return

        print(f"{prefix}[Dynamic Fusion Router Gate (pi = Softmax(MLP([z_C, z_I])))]")
        if "router_prior_C" in res:
            suffix = " (bias-free)" if res.get("router_bias_free") else f" (fc2 bias: [{res['router_bias_C']:.4f}, {res['router_bias_I']:.4f}])"
            print(f"{prefix}  intrinsic prior    : pi_C (CPM)={res['router_prior_C']:.4f} | pi_I (Interaction)={res['router_prior_I']:.4f}{suffix}")

        if res.get("data_evaluated"):
            print(f"{prefix}  control branch (t=0): pi_C={res['router_ctrl_pi_C']:.4f}, pi_I={res['router_ctrl_pi_I']:.4f}, entropy={res['router_ctrl_entropy']:.4f} (norms: ||z_C||={res['z_C_ctrl_norm']:.2f}, ||z_I||={res['z_I_ctrl_norm']:.2f})")
            print(f"{prefix}  treat branch   (t=1): pi_C={res['router_treat_pi_C']:.4f}, pi_I={res['router_treat_pi_I']:.4f}, entropy={res['router_treat_entropy']:.4f} (norms: ||z_C||={res['z_C_treat_norm']:.2f}, ||z_I||={res['z_I_treat_norm']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Inspect CPM fusion checkpoint gating diagnostics without retraining.")
    parser.add_argument("--config", type=str, default="results/drfu/config.json", help="Path to config.json")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint path (.pth)")
    parser.add_argument("--results_dir", type=str, default="results/drfu", help="Directory containing seed checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seed list to evaluate")
    parser.add_argument("--checkpoint_type", type=str, default="best_auuc", choices=["best_auuc", "best", "best_loss", "last"],
                        help="Checkpoint filename type to resolve when scanning results_dir.")
    parser.add_argument("--data", type=str, default=None, help="Optional Criteo dataset directory to evaluate data-dependent router gates on real test samples.")
    parser.add_argument("--num_samples", type=int, default=4096, help="Batch size for sample data evaluation (default: 4096).")
    args = parser.parse_args()

    # Optional data loading for dynamic router inspection
    sample_batch = None
    if args.data and os.path.exists(args.data):
        from preprocess.data_loader import get_dataloaders
        from experiment.main import build_parser as build_main_parser, prepare_loaders_for_model
        print(f"[Data] Loading sample batch of {args.num_samples} samples from {args.data}...")
        train_l, val_l, raw_test_l = get_dataloaders(data_dir=args.data, batch_size=args.num_samples, num_workers=0)
        p = build_main_parser()
        main_args = p.parse_args([])
        if os.path.exists(args.config):
            with open(args.config, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            for k, v in cfg_data.items():
                setattr(main_args, k, v)
        model_name = getattr(main_args, "model", "drfu")
        _, _, eval_test_loader = prepare_loaders_for_model(model_name, main_args, train_l, val_l, raw_test_l)
        sample_batch = next(iter(eval_test_loader))
        print("  ✅ Sample batch loaded successfully.")

    # Checkpoint filename candidates
    type_filenames = {
        "best_auuc": ["best_auuc_checkpoint.pth", "cpm_dynamic_fusion_best_auuc.pth", "best_checkpoint.pth"],
        "best": ["best_checkpoint.pth", "cpm_dynamic_fusion_best.pth"],
        "best_loss": ["best_loss_checkpoint.pth", "cpm_dynamic_fusion_best_loss.pth"],
        "last": ["last_checkpoint.pth", "cpm_dynamic_fusion_final.pth"],
    }
    candidate_names = type_filenames.get(args.checkpoint_type, ["best_auuc_checkpoint.pth", "best_checkpoint.pth"])

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model = detect_and_load_model(args.config, state_dict=state_dict)
        print(f"\n===========================================================================")
        print(f" EVALUATING CHECKPOINT: {args.ckpt}")
        print(f" Model Type: {type(model).__name__}")
        print(f"===========================================================================")
        metrics = inspect_checkpoint(model, args.ckpt, sample_batch=sample_batch)
        print_metrics(metrics, prefix="  ")
        print("===========================================================================\n")
        return

    # Multi-seed evaluation
    print("\n" + "=" * 80)
    print(" CPM / CPM DYNAMIC FUSION CHECKPOINT GATING DIAGNOSTICS (No Retraining Needed)")
    print("=" * 80)

    all_metrics: Dict[str, List[float]] = {}
    seeds_evaluated = 0

    for seed in args.seeds:
        seed_dir = os.path.join(args.results_dir, f"seed_{seed}")
        ckpt_path = None
        for cand in candidate_names:
            p = os.path.join(seed_dir, cand)
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            print(f"\n[Warning] Checkpoint not found for Seed {seed} in {seed_dir} (Checked: {candidate_names})")
            continue

        state_dict = torch.load(ckpt_path, map_location="cpu")
        model = detect_and_load_model(args.config, state_dict=state_dict)
        res = inspect_checkpoint(model, ckpt_path, sample_batch=sample_batch)
        seeds_evaluated += 1

        for k, v in res.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                all_metrics.setdefault(k, []).append(float(v))

        print(f"\n[Seed {seed}] ({ckpt_path}) | Model: {res['model_type']}")
        print_metrics(res, prefix="  ")

    if seeds_evaluated > 0:
        print("\n" + "=" * 80)
        print(f" SUMMARY ({seeds_evaluated} Seeds Evaluated - Mean ± Std)")
        print("=" * 80)
        skip_keys = {"sample_size", "data_evaluated"}
        for k, vals in all_metrics.items():
            if k in skip_keys or not vals:
                continue
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            print(f"  {k:24s}: {mean:.6f} ± {std:.6f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
