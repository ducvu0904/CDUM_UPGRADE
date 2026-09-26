"""
cdum_search_space.py — Hyperparameter search space definition for CPM + VALOR + Dynamic Fusion Optuna tuning.
"""

from typing import Dict, Any
import optuna

# ── Tunable Search Space for CPM + VALOR + Dynamic Fusion (Coarse Search) ────
# 3 * 3 * 3 * 3 * 3 = 243 total categorical combinations.
SEARCH_SPACE = {
    "router_hidden_dim": [32, 64, 128],
    "valor_hidden_dim": [64, 128, 256],
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
    "expert_dim": 64,  # frozen for coarse search stage
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


def sample_cdum_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for CPM + VALOR + Dynamic Fusion coarse search."""
    return {
        "router_hidden_dim": trial.suggest_categorical("router_hidden_dim", SEARCH_SPACE["router_hidden_dim"]),
        "valor_hidden_dim": trial.suggest_categorical("valor_hidden_dim", SEARCH_SPACE["valor_hidden_dim"]),
        "expert_hidden_dim": trial.suggest_categorical("expert_hidden_dim", SEARCH_SPACE["expert_hidden_dim"]),
        "lr": trial.suggest_categorical("lr", SEARCH_SPACE["lr"]),
        "weight_decay": trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"]),
    }
