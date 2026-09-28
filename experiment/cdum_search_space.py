"""
cdum_search_space.py — Hyperparameter search space definition for CPM + Treatment Interaction + Dynamic Fusion Optuna tuning.
"""

from typing import Dict, Any
import optuna

# ── Tunable Search Space for CPM Variants (Coarse Search) ────────────────────
# 3^7 = 2,187 total combinations for 3-branch; 3^6 = 729 for 2-branch.
SEARCH_SPACE = {
    "expert_dim": [32, 64, 128],
    "router_hidden_dim": [32, 64, 128],
    "interaction_hidden_dim": [64, 128, 256],
    "prognostic_hidden_dim": [64, 128, 256],
    "expert_hidden_dim": [64, 128, 256],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0, 1e-6, 1e-4],
}

# ── Frozen Hyperparameters & Architecture Decisions for Coarse Stage ─────────
FIXED_PARAMS = {
    # Optimization & Training
    "optimizer": "Adam",
    "lr_factor": 0.6,
    "lr_patience": 2,
    "min_lr": 1e-6,
    "batch_size": 4096,
    "huber_delta": 1.0,
    "xi": 0,
    "early_stopping_patience": 5,
    "early_stopping_monitor": "val_loss",

    # Architecture & Dimensions (Frozen for Coarse Stage)
    "embedding_dim": 32,
    "treatment_dim": 128,  # embedding_dim * 4
    "num_experts": 3,
    "expert_dim": 64,  # default fallback if not sampled
    "refine_hidden_dim": 64,  # current/default value
    "refine_dim": 32,  # matches tower_hidden_dim for Hadamard mask
    "tower_hidden_dim": 32,  # matches refine_dim for Hadamard mask
    "num_features": 12,
    "num_bins": 101,

    # Architectural Guarantees
    "expert_activation": "relu",
    "expert_output_activation": "relu",
    "gate_bias": False,
    "dropout_rate": 0.0,
    "use_bn": False,
    "output_activation": "softplus",
}


def sample_cdum_params(trial: optuna.Trial, model: str = "drfu") -> Dict[str, Any]:
    """Sample hyperparameters for CPM variants coarse search."""
    params = {
        "expert_dim": trial.suggest_categorical("expert_dim", SEARCH_SPACE["expert_dim"]),
        "router_hidden_dim": trial.suggest_categorical("router_hidden_dim", SEARCH_SPACE["router_hidden_dim"]),
        "interaction_hidden_dim": trial.suggest_categorical("interaction_hidden_dim", SEARCH_SPACE["interaction_hidden_dim"]),
        "expert_hidden_dim": trial.suggest_categorical("expert_hidden_dim", SEARCH_SPACE["expert_hidden_dim"]),
        "lr": trial.suggest_categorical("lr", SEARCH_SPACE["lr"]),
        "weight_decay": trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"]),
    }
    if model != "two_branch_dynamic_fusion" and "prognostic_hidden_dim" in SEARCH_SPACE:
        params["prognostic_hidden_dim"] = trial.suggest_categorical(
            "prognostic_hidden_dim", SEARCH_SPACE["prognostic_hidden_dim"]
        )
    return params
