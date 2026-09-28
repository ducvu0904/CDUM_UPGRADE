"""Train and evaluate CDUM on pre-split Criteo uplift data.

Usage:
    python experiment/main.py --config experiment/config.yaml
    python experiment/main.py --data path/to/criteo --seed 1
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import json
import yaml
import time
import shutil
import random
import logging
import argparse
from datetime import datetime
from contextlib import nullcontext

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

import numpy as np
import pandas as pd
import torch

# ── Đảm bảo project root nằm trong sys.path ──────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocess.data_loader import get_dataloaders
from preprocess.cpm_processor import EquidistantBucketer
from CDUM import CPMTrainer
from CDUM.cpm import CPM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    "cpm": "cdum",
    "cpm_dynamic_fusion": "two_branch_dynamic_fusion",
    "cpm_three_branch_dynamic_fusion": "drfu",
}


def canonical_model_name(model_name: str) -> str:
    """Return the canonical model identifier while accepting legacy names."""
    normalized = model_name.lower()
    return MODEL_ALIASES.get(normalized, normalized)

# ══════════════════════════════════════════════════════════════════════════════
# Config YAML helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_yaml_config(config_path: str) -> dict:
    """Load nested YAML or an explicitly supplied saved flat config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = (json.load(f) if str(config_path).endswith(".json") else yaml.safe_load(f)) or {}

    if str(config_path).endswith(".json"):
        # run_model saves parser destinations verbatim. Do not infer configs
        # from checkpoint paths: callers explicitly select this file.
        if "interaction_hidden_dim" not in cfg and "valor_hidden_dim" in cfg:
            cfg["interaction_hidden_dim"] = cfg["valor_hidden_dim"]
        valid_keys = vars(build_parser().parse_args([]))
        return {key: value for key, value in cfg.items() if key in valid_keys}

    flat = {}
    cpm_keys = {
        "num_features": "cpm_num_features", "num_bins": "cpm_num_bins",
        "embedding_dim": "cpm_embedding_dim", "treatment_dim": "cpm_treatment_dim",
        "num_experts": "cpm_num_experts",
        "expert_hidden_dim": "cpm_expert_hidden_dim", "expert_dim": "cpm_expert_dim",
        "tower_hidden_dim": "cpm_tower_hidden_dim",
        "refine_hidden_dim": "cpm_refine_hidden_dim", "refine_dim": "cpm_refine_dim",
        "huber_delta": "cpm_huber_delta", "seq_len": "cpm_seq_len",
        "activation": "cpm_activation", "dropout_rate": "cpm_dropout",
        "batch_norm": "cpm_batch_norm", "l2_reg": "cpm_weight_decay",
        "router_hidden_dim": "router_hidden_dim",
        "interaction_hidden_dim": "interaction_hidden_dim",
        "valor_hidden_dim": "interaction_hidden_dim",
        "prognostic_hidden_dim": "prognostic_hidden_dim",
    }
    section_map = {
        "data":               {"path": "data", "train_path": "train_path", "val_path": "val_path",
                               "test_path": "test_path", "label_col": "label_col", "num_workers": "num_workers",
                               "test_size": "test_size", "val_ratio": "val_ratio"},
        "model":              {"name": "model", "input_dim": "input_dim", "router_hidden_dim": "router_hidden_dim", "interaction_hidden_dim": "interaction_hidden_dim", "valor_hidden_dim": "interaction_hidden_dim", "prognostic_hidden_dim": "prognostic_hidden_dim"},
        "training":           {"epochs": "epochs", "batch_size": "batch_size", "lr": "lr",
                               "lr_factor": "lr_factor", "lr_patience": "lr_patience", "min_lr": "min_lr",
                               "weight_decay": "weight_decay", "patience": "patience", "device": "device", "seeds": "seeds",
                               "seed": "seed", "monitor_metric": "monitor_metric"},
        "cpm":                cpm_keys,
        "cdum":               cpm_keys,
        "cpm_dynamic_fusion": cpm_keys,
        "cpm_three_branch_dynamic_fusion": cpm_keys,
        "two_branch_dynamic_fusion": cpm_keys,
        "drfu":               cpm_keys,
        "output":             {"checkpoint_dir": "checkpoint_dir", "results_dir": "results_dir",
                               "run_name": "run_name", "eval_k": "eval_k", "verbose": "verbose"},
    }
    for section, mapping in section_map.items():
        if section in cfg and cfg[section]:
            for yaml_key, arg_key in mapping.items():
                if yaml_key in cfg[section] and cfg[section][yaml_key] is not None:
                    flat[arg_key] = cfg[section][yaml_key]
    return flat


def merge_config_into_args(args: argparse.Namespace, config_path: str, raw_argv: list[str] | None = None) -> argparse.Namespace:
    """
    Merge config.yaml vào args.
    CLI args (đã được set tường minh) luôn có ưu tiên cao hơn config.
    """
    yaml_cfg = load_yaml_config(config_path)

    # Lấy tập các arg mà CLI đã cung cấp tường minh
    parser = build_parser()
    defaults = vars(parser.parse_args([]))  # parse với empty args -> full defaults
    cli_provided = {k for k, v in vars(args).items() if v != defaults.get(k)}

    # Bổ sung: Kiểm tra trực tiếp các option flags xuất hiện trong argv
    # (Tránh trường hợp CLI flag được truyền tường minh nhưng giá trị trùng với default của parser,
    # ví dụ: --monitor_metric val_auuc trùng default="val_auuc" dẫn tới bị config.yaml val_loss đè).
    argv_to_check = sys.argv[1:] if raw_argv is None else raw_argv
    option_to_dest = {}
    for action in parser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest

    for token in argv_to_check:
        opt_name = token.split("=")[0]
        if opt_name in option_to_dest:
            cli_provided.add(option_to_dest[opt_name])

    for key, value in yaml_cfg.items():
        if key not in cli_provided:          # chỉ ghi đè nếu CLI không cung cấp
            setattr(args, key, value)

    return args


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="CDUM experiments on the Criteo uplift dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config or saved config.json (e.g. experiment/config.yaml). "
                        "CLI args override config values.")

    # ── Data ──────────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument("--data", type=str, default="/home/ducvu0904/Documents/dataset/Criteo",
                      help="Path to directory containing pre-split Criteo datasets (or explicit split file).")
    data.add_argument("--train_path", type=str, default=None,
                      help="Optional explicit path to train split (.pt or .csv).")
    data.add_argument("--val_path", type=str, default=None,
                      help="Optional explicit path to val split (.pt or .csv).")
    data.add_argument("--test_path", type=str, default=None,
                      help="Optional explicit path to test split (.pt or .csv).")
    data.add_argument("--label_col", type=str, default="visit", choices=["visit", "conversion"],
                      help="Target outcome column ('visit' or 'conversion').")
    data.add_argument("--max_samples", type=int, default=None,
                      help="Optional cap on the number of samples loaded per split (for fast debugging).")
    data.add_argument("--seed", type=int, default=None,
                      help="Optional single seed. If set, overrides --seeds with [seed].")

    # ── Model ─────────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--model", type=str, default="cdum", choices=["cdum", "cpm", "two_branch_dynamic_fusion", "drfu", "cpm_dynamic_fusion", "cpm_three_branch_dynamic_fusion"],
                     help="CPM baseline or dynamic-fusion variant. 'cpm' is retained as an alias for CDUM.")
    mdl.add_argument("--input_dim", type=int, default=12,
                     help="Number of input features (12 for Criteo f0..f11).")

    # ── CDUM / CPM-specific ───────────────────────────────────────────────────
    cdm = p.add_argument_group("CDUM / CPM options")
    cdm.add_argument("--router_hidden_dim", "--cpm_router_hidden_dim", "--cdum_router_hidden_dim",
                     dest="router_hidden_dim", type=int, default=None,
                     help="Router hidden dimension for dynamic-fusion variants (defaults to expert_dim D).")
    cdm.add_argument(
        "--interaction_hidden_dim", "--treatment_interaction_hidden_dim",
        "--valor_hidden_dim", "--cpm_valor_hidden_dim", "--cdum_valor_hidden_dim",
        dest="interaction_hidden_dim", type=int, default=None,
        help="Treatment-interaction MLP hidden dimension (defaults to expert_hidden_dim).",
    )
    cdm.add_argument("--prognostic_hidden_dim", "--cpm_prognostic_hidden_dim", "--cdum_prognostic_hidden_dim",
                     dest="prognostic_hidden_dim", type=int, default=None,
                     help="Prognostic MLP hidden dimension for three-branch fusion (defaults to expert_hidden_dim).")
    cdm.add_argument("--cpm_num_features", "--cdum_num_features", dest="cpm_num_features", type=int, default=None,
                     help="Number of input features for CDUM (defaults to --input_dim).")
    cdm.add_argument("--cpm_num_bins", "--cdum_num_bins", dest="cpm_num_bins", type=int, default=101,
                     help="Number of bins for feature discretization in CDUM (default: 101, indices 0..100).")
    cdm.add_argument("--cpm_embedding_dim", "--cdum_embedding_dim", dest="cpm_embedding_dim", type=int, default=32,
                     help="Feature embedding dimension for CDUM.")
    cdm.add_argument("--cpm_treatment_dim", "--cdum_treatment_dim", dest="cpm_treatment_dim", type=int, default=128,
                     help="Treatment embedding dimension for CDUM (defaults to embedding_dim * 4 = 128).")
    cdm.add_argument("--cpm_seq_len", "--cdum_seq_len", dest="cpm_seq_len", type=int, default=10,
                     help="Sequence length for CDUM.")
    cdm.add_argument("--cpm_num_experts", "--cdum_num_experts", dest="cpm_num_experts", type=int, default=3,
                     help="Number of expert modules for CDUM.")
    cdm.add_argument("--cpm_expert_hidden_dim", "--cdum_expert_hidden_dim", dest="cpm_expert_hidden_dim", type=int, default=128,
                     help="Hidden dimension of user experts for CDUM (Expert hidden unit 1).")
    cdm.add_argument("--cpm_expert_dim", "--cdum_expert_dim", dest="cpm_expert_dim", type=int, default=64,
                     help="Output dimension of user experts for CDUM (Expert hidden unit 2).")
    cdm.add_argument("--cpm_tower_hidden_dim", "--cdum_tower_hidden_dim", dest="cpm_tower_hidden_dim", type=int, default=32,
                     help="Hidden dimension of treatment towers for CDUM (Tower hidden unit).")
    cdm.add_argument("--cpm_refine_hidden_dim", "--cdum_refine_hidden_dim", dest="cpm_refine_hidden_dim", type=int, default=64,
                     help="Hidden dimension for treatment refine in CDUM.")
    cdm.add_argument("--cpm_refine_dim", "--cdum_refine_dim", dest="cpm_refine_dim", type=int, default=32,
                     help="Output guidance/indicator dimension for CDUM.")
    cdm.add_argument("--cpm_activation", "--cdum_activation", dest="cpm_activation", type=str, default="relu",
                     help="Activation function for CDUM.")
    cdm.add_argument("--cpm_dropout", "--cdum_dropout", dest="cpm_dropout", type=float, default=0.0,
                     help="Dropout rate for CDUM.")
    cdm.add_argument("--cpm_batch_norm", "--cdum_batch_norm", dest="cpm_batch_norm", action="store_true", default=False,
                     help="Enable batch normalization for CDUM.")
    cdm.add_argument("--cpm_weight_decay", "--cdum_weight_decay", dest="cpm_weight_decay", type=float, default=None,
                     help="L2 regularization (weight decay) for CDUM (defaults to --weight_decay: 1e-5).")
    cdm.add_argument("--cpm_huber_delta", "--cdum_huber_delta", dest="cpm_huber_delta", type=float, default=1.0,
                     help="Huber loss delta for CDUM.")

    # ── Training ──────────────────────────────────────────────────────────────
    trn = p.add_argument_group("Training")
    trn.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    trn.add_argument("--batch_size", type=int, default=4096, help="Batch size.")
    trn.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    trn.add_argument("--lr_factor", type=float, default=0.5,
                     help="Factor by which the learning rate will be reduced (ReduceLROnPlateau).")
    trn.add_argument("--lr_patience", type=int, default=2,
                     help="Number of epochs with no improvement after which learning rate will be reduced.")
    trn.add_argument("--min_lr", type=float, default=1e-6,
                     help="Minimum learning rate for ReduceLROnPlateau.")
    trn.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay.")
    trn.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    trn.add_argument("--monitor_metric", type=str, default="val_auuc", choices=["val_auuc", "val_loss"],
                     help="Metric to monitor for best checkpoint selection and early stopping ('val_auuc' or 'val_loss', default: 'val_auuc').")
    trn.add_argument("--device", type=str, default=None,
                     help="Device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detect if not set.")
    trn.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                     help="List of random seeds to evaluate model stability (default: 1 2 3 4 5).")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                     help="Root directory to save best model checkpoints.")
    out.add_argument("--results_dir", type=str, default="results",
                     help="Root directory to save evaluation JSON results.")
    out.add_argument("--run_name", type=str, default=None,
                     help="Optional run name. Defaults to model name + timestamp.")
    out.add_argument("--eval_k", type=float, default=0.3,
                     help="Fraction k for Uplift@k evaluation (default 0.3 = 30%%).")
    out.add_argument("--verbose", type=int, default=1,
                     help="Verbosity level: 0=silent, 1=epoch log, 2=+checkpoint log.")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, args: argparse.Namespace):
    """Build the CDUM model and training wrapper."""
    model_name_lower = canonical_model_name(model_name)
    if model_name_lower in ("cdum", "two_branch_dynamic_fusion", "drfu"):
        num_features = getattr(args, "cpm_num_features", None) or args.input_dim
        num_bins = getattr(args, "cpm_num_bins", 101)
        embedding_dim = getattr(args, "cpm_embedding_dim", 32)
        treatment_dim = getattr(args, "cpm_treatment_dim", None) or (embedding_dim * 4)
        refine_hidden_dim = getattr(args, "cpm_refine_hidden_dim", 64)
        refine_dim = getattr(args, "cpm_refine_dim", 32)
        num_experts = getattr(args, "cpm_num_experts", 3)
        expert_hidden_dim = getattr(args, "cpm_expert_hidden_dim", 128)
        expert_dim = getattr(args, "cpm_expert_dim", 64)
        tower_hidden_dim = getattr(args, "cpm_tower_hidden_dim", 32)
        activation = getattr(args, "cpm_activation", "relu")
        dropout_rate = getattr(args, "cpm_dropout", 0.0)
        use_bn = getattr(args, "cpm_batch_norm", False)
        huber_delta = getattr(args, "cpm_huber_delta", 1.0)
        cpm_weight_decay = getattr(args, "cpm_weight_decay", None)
        weight_decay = cpm_weight_decay if cpm_weight_decay is not None else getattr(args, "weight_decay", 1e-5)

        model_kwargs = dict(
            num_features=num_features,
            num_bins=num_bins,
            embedding_dim=embedding_dim,
            treatment_dim=treatment_dim,
            refine_hidden_dim=refine_hidden_dim,
            refine_dim=refine_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            expert_dim=expert_dim,
            tower_hidden_dim=tower_hidden_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )

        if model_name_lower in ("two_branch_dynamic_fusion", "drfu"):
            from CDUM.variants import DRFU, TwoBranchDynamicFusion
            model_cls = TwoBranchDynamicFusion if model_name_lower == "two_branch_dynamic_fusion" else DRFU
            router_hidden_dim = getattr(args, "router_hidden_dim", None)
            if router_hidden_dim is not None:
                model_kwargs["router_hidden_dim"] = router_hidden_dim
            interaction_hidden_dim = getattr(args, "interaction_hidden_dim", None)
            if interaction_hidden_dim is not None:
                model_kwargs["interaction_hidden_dim"] = interaction_hidden_dim
            if model_name_lower == "drfu":
                model_kwargs["prognostic_hidden_dim"] = getattr(args, "prognostic_hidden_dim", None)
        else:
            model_cls = CPM

        model = model_cls(**model_kwargs)
        trainer = CPMTrainer(
            model=model,
            lr=args.lr,
            weight_decay=weight_decay,
            lr_factor=getattr(args, "lr_factor", 0.6),
            lr_patience=getattr(args, "lr_patience", 2),
            min_lr=getattr(args, "min_lr", 1e-6),
            device=args.device,
        )
        if huber_delta != 1.0:
            import torch.nn as nn
            trainer.criterion = nn.HuberLoss(delta=huber_delta)
        return trainer

    raise ValueError(f"Unsupported model '{model_name}'. Supported models: 'cdum', 'cpm', 'two_branch_dynamic_fusion', and 'drfu'.")


# ══════════════════════════════════════════════════════════════════════════════
# Seed & CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Thiết lập random seed toàn cục cho reproducibility tuyệt đối."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def update_summary_csv(summary_csv_path: str, model_name: str, summary_dict: dict):
    """
    Cập nhật hoặc thêm hàng kết quả summary (mean ± std) của mô hình vào results/summary.csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
    mean = summary_dict.get("mean", {})
    std = summary_dict.get("std", {})

    val_loss_m, val_loss_s = mean.get("val_loss", 0.0), std.get("val_loss", 0.0)
    val_auuc_m, val_auuc_s = mean.get("val_auuc", 0.0), std.get("val_auuc", 0.0)
    auuc_m, auuc_s = mean.get("test_auuc", 0.0), std.get("test_auuc", 0.0)
    qini_m, qini_s = mean.get("test_qini", 0.0), std.get("test_qini", 0.0)
    lift_m, lift_s = mean.get("test_lift@30", 0.0), std.get("test_lift@30", 0.0)
    best_ep_m, best_ep_s = mean.get("best_epoch", 0.0), std.get("best_epoch", 0.0)

    row_data = {
        "model": model_name.lower(),
        "best_epoch": f"{best_ep_m:.1f} ± {best_ep_s:.2f}",
        "val_loss": f"{val_loss_m:.5f} ± {val_loss_s:.5f}",
        "val_auuc": f"{val_auuc_m:.5f} ± {val_auuc_s:.5f}",
        "test_auuc": f"{auuc_m:.5f} ± {auuc_s:.5f}",
        "test_qini": f"{qini_m:.5f} ± {qini_s:.5f}",
        "test_lift@30": f"{lift_m:.5f} ± {lift_s:.5f}",
        "val_loss_mean": val_loss_m,
        "val_loss_std": val_loss_s,
        "val_auuc_mean": val_auuc_m,
        "val_auuc_std": val_auuc_s,
        "test_auuc_mean": auuc_m,
        "test_auuc_std": auuc_s,
        "test_qini_mean": qini_m,
        "test_qini_std": qini_s,
        "test_lift@30_mean": lift_m,
        "test_lift@30_std": lift_s,
    }

    lock_path = summary_csv_path + ".lock"
    lock_ctx = FileLock(lock_path, timeout=60) if FileLock is not None else nullcontext()

    with lock_ctx:
        if os.path.exists(summary_csv_path):
            try:
                df_summary = pd.read_csv(summary_csv_path)
                if "model" in df_summary.columns and model_name.lower() in df_summary["model"].values:
                    idx = df_summary[df_summary["model"] == model_name.lower()].index[0]
                    for k, v in row_data.items():
                        df_summary.loc[idx, k] = v
                else:
                    df_new = pd.DataFrame([row_data])
                    df_summary = pd.concat([df_summary, df_new], ignore_index=True)
            except Exception:
                df_summary = pd.DataFrame([row_data])
        else:
            df_summary = pd.DataFrame([row_data])

        df_summary.to_csv(summary_csv_path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader Wrapper for CDUM / CPM (Equidistant Bucketing)
# ══════════════════════════════════════════════════════════════════════════════

class BucketedDataLoader:
    """
    Wrapper quanh DataLoader để bucketize continuous features thành discrete bucket IDs
    cho CPM / CDUM on-the-fly.
    """
    def __init__(self, dataloader, bucketer: EquidistantBucketer):
        self.dataloader = dataloader
        self.bucketer = bucketer

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x_b, t_b, y_b = batch[0], batch[1], batch[2]
                if isinstance(x_b, torch.Tensor) and x_b.is_floating_point():
                    x_b = self.bucketer.transform(x_b)
                yield (x_b, t_b, y_b)
            else:
                yield batch

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return getattr(self.dataloader, "dataset", None)


def prepare_loaders_for_model(
    model_name: str,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
):
    """Nếu model là CDUM / CPM và features là continuous float, bọc DataLoaders bằng EquidistantBucketer."""
    if canonical_model_name(model_name) in ("cdum", "two_branch_dynamic_fusion", "drfu") and not isinstance(train_loader, BucketedDataLoader):
        num_bins = getattr(args, "cpm_num_bins", 101)
        bucketer = EquidistantBucketer(num_bins=num_bins)

        # Fit bucketer trên train features nếu là floating point
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "X"):
            x_mat = train_loader.dataset.X
            if not isinstance(x_mat, torch.Tensor):
                x_mat = torch.as_tensor(x_mat)
            if x_mat.is_floating_point():
                bucketer.fit(x_mat)

        if bucketer.denominators is None and hasattr(bucketer, "boundaries") and bucketer.boundaries is None:
            first_b = next(iter(train_loader))
            first_x = first_b[0] if isinstance(first_b, (list, tuple)) else first_b
            if not isinstance(first_x, torch.Tensor):
                first_x = torch.as_tensor(first_x)
            if first_x.is_floating_point():
                bucketer.fit(first_x)

        if bucketer.denominators is not None or getattr(bucketer, "boundaries", None) is not None:
            train_loader = BucketedDataLoader(train_loader, bucketer)
            if val_loader is not None and not isinstance(val_loader, BucketedDataLoader):
                val_loader = BucketedDataLoader(val_loader, bucketer)
            if test_loader is not None and not isinstance(test_loader, BucketedDataLoader):
                test_loader = BucketedDataLoader(test_loader, bucketer)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Run single seed
# ══════════════════════════════════════════════════════════════════════════════

def run_single_seed(
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
) -> dict:
    """Chạy huấn luyện và đánh giá mô hình cho 1 seed cụ thể."""
    set_seed(seed)

    # Đảm bảo generator của train_loader được re-seed chính xác theo seed hiện tại
    raw_loader = train_loader.dataloader if hasattr(train_loader, "dataloader") else train_loader
    if hasattr(raw_loader, "generator") and raw_loader.generator is not None:
        raw_loader.generator.manual_seed(seed)
    elif hasattr(raw_loader, "sampler") and hasattr(raw_loader.sampler, "generator"):
        gen = torch.Generator()
        gen.manual_seed(seed)
        raw_loader.sampler.generator = gen
        raw_loader.generator = gen

    train_loader, val_loader, test_loader = prepare_loaders_for_model(
        model_name, args, train_loader, val_loader, test_loader
    )
    model_lower = model_name.lower()

    # Thư mục checkpoint riêng cho seed: results/<model_name>/seed_<seed>
    seed_dir = os.path.join(args.results_dir, model_lower, f"seed_{seed}")
    tb_dir = os.path.join(args.results_dir, model_lower, f"seed_{seed}", "runs")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    logger.info(f"{'-'*60}")
    logger.info(f" Model: {model_name.upper()} | Seed: {seed} | Checkpoints: {seed_dir}")
    logger.info(f"{'-'*60}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_dir)
    except Exception as e:
        logger.warning(f"TensorBoard unavailable ({e}), continuing without TensorBoard logging.")

    trainer = build_model(model_name, args)
    t_start = time.time()

    # ── Fit ───────────────────────────────────────────────────────────────────
    fit_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping_patience=args.patience,
        checkpoint_dir=seed_dir,
        model_name=f"{model_lower}",
        writer=writer,
        verbose=args.verbose,
        monitor=getattr(args, "monitor_metric", "val_auuc"),
    )

    fit_kwargs["epochs"] = args.epochs

    history = trainer.fit(**fit_kwargs)
    elapsed = time.time() - t_start

    # ── Lưu best_checkpoint, best_auuc_checkpoint, best_loss_checkpoint, last_checkpoint ──
    files_to_organize = [
        (f"{model_lower}_best.pth", "best_checkpoint"),
        (f"{model_lower}_best_auuc.pth", "best_auuc_checkpoint"),
        (f"{model_lower}_best_loss.pth", "best_loss_checkpoint"),
        (f"{model_lower}_final.pth", "last_checkpoint"),
    ]
    for src_name, dst_base in files_to_organize:
        src_path = os.path.join(seed_dir, src_name)
        dst_pth = os.path.join(seed_dir, f"{dst_base}.pth")
        dst_pt = os.path.join(seed_dir, f"{dst_base}.pt")
        if os.path.exists(src_path):
            shutil.move(src_path, dst_pth)
            try:
                if os.path.exists(dst_pt):
                    os.remove(dst_pt)
                os.link(dst_pth, dst_pt)
            except Exception:
                shutil.copy2(dst_pth, dst_pt)

    # ── Đánh giá model chính (theo monitor_metric đã chọn) trên test set ───────
    monitor_metric = getattr(args, "monitor_metric", "val_auuc")
    logger.info(f"Evaluating {model_name.upper()} (Seed {seed}) [Primary: {monitor_metric}] on test set...")
    metrics = trainer.evaluate(test_loader, k=args.eval_k)

    if writer is not None:
        for metric_name, val in metrics.items():
            if isinstance(val, (int, float)):
                writer.add_scalar(f"Test/{metric_name}", val, 0)
        writer.close()

    # ── Thu thập metrics cho seed này ─────────────────────────────────────────
    val_loss_hist = history.get("val_loss", [])
    val_auuc_hist = history.get("val_auuc", [])

    best_loss_idx = int(np.nanargmin(val_loss_hist)) if (val_loss_hist and not np.all(np.isnan(val_loss_hist))) else 0
    best_loss_epoch = best_loss_idx + 1

    if val_auuc_hist and not np.all(np.isnan(val_auuc_hist)):
        best_auuc_idx = int(np.nanargmax(val_auuc_hist))
        best_auuc_epoch = best_auuc_idx + 1
    else:
        best_auuc_idx = best_loss_idx
        best_auuc_epoch = best_loss_epoch

    best_idx = best_auuc_idx if monitor_metric == "val_auuc" else best_loss_idx
    best_epoch = best_idx + 1
    best_val_loss = float(val_loss_hist[best_idx]) if val_loss_hist and best_idx < len(val_loss_hist) else float("nan")
    best_val_auuc = float(val_auuc_hist[best_idx]) if val_auuc_hist and best_idx < len(val_auuc_hist) else float("nan")

    auuc_val = float(metrics.get("auuc", 0.0))
    qini_val = float(metrics.get("qini", 0.0))
    lift_val = float(metrics.get("lift@30%", metrics.get(f"lift@{int(args.eval_k*100)}%", 0.0)))
    test_loss_val = float(metrics.get("loss", 0.0))

    # ── Đánh giá checkpoint đối chiếu (alternate checkpoint) cho so sánh ───────
    if best_auuc_epoch == best_loss_epoch:
        auuc_test_metrics = metrics
        loss_test_metrics = metrics
    elif monitor_metric == "val_auuc":
        auuc_test_metrics = metrics
        loss_ckpt = os.path.join(seed_dir, "best_loss_checkpoint.pth")
        if os.path.exists(loss_ckpt):
            try:
                trainer.load(loss_ckpt)
                loss_test_metrics = trainer.evaluate(test_loader, k=args.eval_k, print_diagnostics=False)
                trainer.load(os.path.join(seed_dir, "best_checkpoint.pth"))
            except Exception as e:
                logger.warning(f"Could not evaluate alternate loss checkpoint: {e}")
                loss_test_metrics = metrics
        else:
            loss_test_metrics = metrics
    else:
        loss_test_metrics = metrics
        auuc_ckpt = os.path.join(seed_dir, "best_auuc_checkpoint.pth")
        if os.path.exists(auuc_ckpt):
            try:
                trainer.load(auuc_ckpt)
                auuc_test_metrics = trainer.evaluate(test_loader, k=args.eval_k, print_diagnostics=False)
                trainer.load(os.path.join(seed_dir, "best_checkpoint.pth"))
            except Exception as e:
                logger.warning(f"Could not evaluate alternate auuc checkpoint: {e}")
                auuc_test_metrics = metrics
        else:
            auuc_test_metrics = metrics

    comparison_info = {
        "best_auuc": {
            "epoch": best_auuc_epoch,
            "val_loss": round(float(val_loss_hist[best_auuc_idx]), 6) if val_loss_hist and best_auuc_idx < len(val_loss_hist) else None,
            "val_auuc": round(float(val_auuc_hist[best_auuc_idx]), 6) if val_auuc_hist and best_auuc_idx < len(val_auuc_hist) else None,
            "test_auuc": round(float(auuc_test_metrics.get("auuc", 0.0)), 6),
            "test_qini": round(float(auuc_test_metrics.get("qini", 0.0)), 6),
            "test_lift@30": round(float(auuc_test_metrics.get("lift@30%", auuc_test_metrics.get(f"lift@{int(args.eval_k*100)}%", 0.0))), 6),
        },
        "best_loss": {
            "epoch": best_loss_epoch,
            "val_loss": round(float(val_loss_hist[best_loss_idx]), 6) if val_loss_hist and best_loss_idx < len(val_loss_hist) else None,
            "val_auuc": round(float(val_auuc_hist[best_loss_idx]), 6) if val_auuc_hist and best_loss_idx < len(val_auuc_hist) else None,
            "test_auuc": round(float(loss_test_metrics.get("auuc", 0.0)), 6),
            "test_qini": round(float(loss_test_metrics.get("qini", 0.0)), 6),
            "test_lift@30": round(float(loss_test_metrics.get("lift@30%", loss_test_metrics.get(f"lift@{int(args.eval_k*100)}%", 0.0))), 6),
        },
    }

    seed_result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_loss": round(best_val_loss, 6),
        "val_auuc": round(best_val_auuc, 6),
        "test_auuc": round(auuc_val, 6),
        "test_qini": round(qini_val, 6),
        "test_lift@30": round(lift_val, 6),
        "test_loss": round(test_loss_val, 6),
        "train_time_s": round(elapsed, 2),
        "comparison": comparison_info,
    }
    logger.info(
        f"Seed {seed} Done | Best Epoch: {best_epoch} | Val Loss: {best_val_loss:.5f} | Val AUUC: {best_val_auuc:.5f} "
        f"| Test AUUC: {auuc_val:.5f} | Test Qini: {qini_val:.5f} | Test Lift@30: {lift_val:.5f} "
        f"| Time: {elapsed:.1f}s"
    )
    if best_auuc_epoch != best_loss_epoch:
        logger.info(
            f"  [Checkpoint Comparison] "
            f"Best AUUC (Ep {best_auuc_epoch}): Test AUUC={comparison_info['best_auuc']['test_auuc']:.5f}, Qini={comparison_info['best_auuc']['test_qini']:.5f} | "
            f"Best Loss (Ep {best_loss_epoch}): Test AUUC={comparison_info['best_loss']['test_auuc']:.5f}, Qini={comparison_info['best_loss']['test_qini']:.5f}"
        )
    return seed_result


# ══════════════════════════════════════════════════════════════════════════════
# Run multi-seed experiment for a model
# ══════════════════════════════════════════════════════════════════════════════

def run_model(
    model_name: str,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
) -> dict:
    """
    Chạy thí nghiệm cho một mô hình qua danh sách seeds (mặc định: 1-5).
    Lưu trữ cấu trúc chuẩn:
      - results/<model>/seed_<seed>/best_checkpoint.pth & last_checkpoint.pth
      - results/<model>/config.json
      - results/<model>/metrics.json (mỗi seed + mean/std summary)
      - results/summary.csv
    """
    model_lower = model_name.lower()
    train_loader, val_loader, test_loader = prepare_loaders_for_model(
        model_name, args, train_loader, val_loader, test_loader
    )
    model_dir = os.path.join(args.results_dir, model_lower)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "run.log")

    file_handler = None
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s  %(levelname)-8s  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    root_logger.addHandler(file_handler)

    try:
        seeds = args.seeds
        if isinstance(seeds, int):
            seeds = [seeds]

        logger.info(f"\n{'='*70}")
        logger.info(f" MODEL: {model_name.upper()} | SEEDS: {seeds}")
        logger.info(f" Directory: {model_dir}")
        logger.info(f" Run Log:   {log_path}")
        logger.info(f"{'='*70}")

        # 1. Lưu config.json riêng biệt (hyperparameters & options)
        config_dict = vars(args).copy()
        config_dict["model_name"] = model_lower
        config_dict["run_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Hyperparameters saved to: {config_path}")

        # 2. Chạy từng seed
        raw_results = {}
        for i, s in enumerate(seeds, start=1):
            logger.info(f"\n>>> Running {model_name.upper()} — Seed {s} ({i}/{len(seeds)}) ...")
            res = run_single_seed(model_name, s, args, train_loader, val_loader, test_loader)
            raw_results[str(s)] = res

        # 3. Tính toán summary (mean và std)
        metric_keys = ["best_epoch", "val_loss", "val_auuc", "test_auuc", "test_qini", "test_lift@30"]
        summary_mean = {}
        summary_std = {}

        for k in metric_keys:
            vals = [r[k] for r in raw_results.values() if k in r and not np.isnan(r[k])]
            if vals:
                summary_mean[k] = round(float(np.mean(vals)), 6)
                summary_std[k] = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0
            else:
                summary_mean[k] = float("nan")
                summary_std[k] = float("nan")

        summary_section = {
            "mean": summary_mean,
            "std": summary_std,
        }

        # 4. Lưu metrics.json (chứa kết quả chi tiết từng seed và tổng hợp mean/std)
        metrics_record = {
            "model": model_lower,
            "seeds": raw_results,
            "summary": summary_section,
        }
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_record, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Metrics saved to: {metrics_path}")

        # 5. Cập nhật vào results/summary.csv
        summary_csv_path = os.path.join(args.results_dir, "summary.csv")
        update_summary_csv(summary_csv_path, model_lower, summary_section)
        logger.info(f"Summary CSV updated at: {summary_csv_path}")

        # Log summary cho model này
        logger.info("")
        logger.info(f"--- SUMMARY FOR {model_name.upper()} ({len(seeds)} Seeds) ---")
        for k in metric_keys:
            logger.info(f"  {k:15s}: {summary_mean[k]:.5f} ± {summary_std[k]:.5f}")

        return metrics_record
    finally:
        if file_handler is not None:
            file_handler.flush()
            file_handler.close()
            root_logger.removeHandler(file_handler)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Merge config.yaml (nếu có) vào args ──────────────────────────────────
    if args.config is not None:
        if not os.path.exists(args.config):
            parser.error(f"Config file not found: {args.config}")
        args = merge_config_into_args(args, args.config)
        logger.info(f"Loaded config from: {args.config}")

    # ── Xử lý danh sách seeds (mặc định 1..5) ─────────────────────────────────
    if args.seed is not None:
        args.seeds = [args.seed]
    elif isinstance(args.seeds, str):
        args.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    elif isinstance(args.seeds, int):
        args.seeds = [args.seeds]
    else:
        args.seeds = list(args.seeds)

    raw_model = getattr(args, "model", "cdum").lower()
    model_name = canonical_model_name(raw_model)
    args.model = model_name

    # ── Nạp dữ liệu pre-split và tạo DataLoaders ──────────────────────────────
    data_dir = args.data if (os.path.exists(args.data) and os.path.isdir(args.data)) else None
    train_path = getattr(args, "train_path", None)
    if train_path is None and os.path.exists(args.data) and not os.path.isdir(args.data):
        train_path = args.data

    init_seed = args.seeds[0] if getattr(args, "seeds", None) else 42
    set_seed(init_seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        train_path=train_path,
        val_path=getattr(args, "val_path", None),
        test_path=getattr(args, "test_path", None),
        batch_size=args.batch_size,
        label_col=getattr(args, "label_col", "visit"),
        num_workers=getattr(args, "num_workers", 2),
        max_samples=getattr(args, "max_samples", None),
        seed=init_seed,
    )

    logger.info(
        f"DataLoaders ready | train batches: {len(train_loader):,} "
        f"| val batches: {len(val_loader) if val_loader else 0:,} "
        f"| test batches: {len(test_loader) if test_loader else 0:,}"
    )

    return run_model(model_name, args, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
