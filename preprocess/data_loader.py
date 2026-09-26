"""
data_loader.py — Load pre-split Criteo dataset and convert into PyTorch model inputs
====================================================================================
Separation of Concerns:
  - Dataset splitting is handled upstream by dataset/Criteo/split_criteo.py
    (train/val/test splits, StandardScaler, .csv and .pt caches).
  - This module is SOLELY responsible for:
      1. Loading pre-split data from disk (.pt binary tensor caches or .csv files).
      2. Converting loaded samples into model-ready PyTorch Tensors (X, t, y).
      3. Constructing PyTorch DataLoaders yielding batches of (x_b, t_b, y_b)
         for CDUM training, validation, and evaluation.

Batch structure expected by CDUM:
  - x_b : torch.FloatTensor of shape (batch_size, 12)  [numeric features f0..f11]
  - t_b : torch.FloatTensor of shape (batch_size,)     [binary treatment 0/1]
  - y_b : torch.FloatTensor of shape (batch_size,)     [binary outcome: visit or conversion]
"""

import os
import sys
import logging
import warnings
import random
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Default locations for pre-split Criteo datasets
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
DEFAULT_CRITEO_DIR = "/home/ducvu0904/Documents/dataset/Criteo"

NON_FEATURE_COLS = {
    "y0", "y1", "true_tau", "treatment", "label",
    "spend", "conversion", "visit", "exposure",
}


def _to_tensor(
    data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert DataFrame / Series / ndarray to torch.Tensor with specified dtype."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return torch.from_numpy(data.to_numpy(dtype=np.float32)).to(dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32)).to(dtype=dtype)
    else:
        return torch.tensor(data, dtype=dtype)


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset for Criteo Uplift
# ══════════════════════════════════════════════════════════════════════════════

class CriteoDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-processed Criteo Uplift features & labels.
    Each item returns a tuple: (x, t, y) as float32 tensors, ready for model input.

    Attributes
    ----------
    X : torch.Tensor of shape (N, D) — feature matrix (e.g. f0..f11)
    t : torch.Tensor of shape (N,)   — treatment indicator (0 or 1)
    y : torch.Tensor of shape (N,)   — binary outcome (visit or conversion)
    """

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        t: Union[np.ndarray, pd.Series, torch.Tensor],
        y: Union[np.ndarray, pd.Series, torch.Tensor],
    ):
        self.X = _to_tensor(X, dtype=torch.float32)
        self.t = _to_tensor(t, dtype=torch.float32).view(-1)
        self.y = _to_tensor(y, dtype=torch.float32).view(-1)

        assert len(self.X) == len(self.t) == len(self.y), (
            f"Mismatched sample counts: X={len(self.X)}, t={len(self.t)}, y={len(self.y)}"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.t[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Split File Resolution & Loading
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_split_path(data_dir: str, split_name: str) -> str:
    """
    Locate the pre-split file for a given split ('train', 'val', or 'test') in data_dir.
    Prefers .pt binary tensor caches over .csv files for performance.
    """
    candidate_names = [
        f"{split_name}_criteo.pt",
        f"{split_name}.pt",
        f"{split_name}_criteo.csv",
        f"{split_name}.csv",
        f"{split_name}_criteo.csv.gz",
        f"{split_name}.csv.gz",
    ]
    for name in candidate_names:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not find pre-split file for '{split_name}' in directory '{data_dir}'. "
        f"Checked candidates: {candidate_names}. "
        f"Please run 'python /home/ducvu0904/Documents/dataset/Criteo/split_criteo.py' "
        f"to generate the pre-split files."
    )


def load_split(
    file_path: str,
    label_col: str = "visit",
    treat_col: str = "treatment",
    feature_cols: Optional[List[str]] = None,
    mmap: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a pre-split Criteo file (.pt binary cache or .csv) and return (X, t, y) tensors.

    Parameters
    ----------
    file_path    : Path to .pt or .csv file.
    label_col    : Target outcome column ('visit' or 'conversion').
    treat_col    : Treatment column name (default 'treatment').
    feature_cols : List of feature column names (for CSV reading).
    mmap         : Use memory-mapping for .pt tensor files.
    max_samples  : Optional cap on the number of samples loaded (for debugging).

    Returns
    -------
    (X, t, y) : Float32 tensors ready for CriteoDataset / DataLoader.
    """
    # If passed a CSV path, check if sibling .pt file exists for fast loading
    if file_path.endswith((".csv", ".csv.gz")):
        pt_sibling = file_path.replace(".csv.gz", ".pt").replace(".csv", ".pt")
        if os.path.exists(pt_sibling):
            logger.info(f"Found binary tensor cache {pt_sibling}, using it instead of CSV.")
            file_path = pt_sibling

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file not found: {file_path}")

    # Case 1: Binary PyTorch Tensor cache (.pt)
    if file_path.endswith(".pt"):
        logger.info(f"Loading binary tensor split: {file_path} (mmap={mmap})")
        data = None
        if mmap:
            try:
                data = torch.load(file_path, mmap=True, weights_only=False)
            except Exception as e:
                logger.debug(f"mmap load failed ({e}), falling back to regular torch.load")
                data = torch.load(file_path, weights_only=False)
        else:
            data = torch.load(file_path, weights_only=False)

        # Extract features X
        if "x_num" in data:
            X = data["x_num"].float()
        elif "X" in data:
            X = data["X"].float()
        else:
            raise KeyError(f"Expected 'x_num' or 'X' in {file_path}, found {list(data.keys())}")

        # Extract treatment t
        if treat_col in data:
            t = data[treat_col].float()
        elif "treatment" in data:
            t = data["treatment"].float()
        else:
            raise KeyError(f"Expected treatment key in {file_path}, found {list(data.keys())}")

        # Extract target y
        if label_col in data:
            y = data[label_col].float()
        elif label_col == "conversion" and "visit" in data:
            y = data["visit"].float()
        elif label_col == "visit" and "conversion" in data:
            y = data["conversion"].float()
        elif "label" in data:
            y = data["label"].float()
        else:
            raise KeyError(f"Target '{label_col}' not found in {file_path}. Available: {list(data.keys())}")

        if max_samples is not None and max_samples < len(X):
            X = X[:max_samples]
            t = t[:max_samples]
            y = y[:max_samples]

        return X, t, y

    # Case 2: CSV file
    logger.info(f"Loading CSV split: {file_path}")
    df = pd.read_csv(file_path, nrows=max_samples)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        if not feature_cols:
            feature_cols = [f"f{i}" for i in range(12)]

    if treat_col not in df.columns:
        raise KeyError(f"Treatment column '{treat_col}' not found in {file_path}")
    if label_col not in df.columns:
        # Fallback to alternate column if available
        alt = "conversion" if label_col == "visit" else "visit"
        if alt in df.columns:
            logger.warning(f"Label column '{label_col}' not found; falling back to '{alt}'")
            label_col = alt
        else:
            raise KeyError(f"Outcome column '{label_col}' not found in {file_path}")

    X = torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32))
    t = torch.from_numpy(df[treat_col].to_numpy(dtype=np.float32))
    y = torch.from_numpy(df[label_col].to_numpy(dtype=np.float32))

    return X, t, y


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader Builders
# ══════════════════════════════════════════════════════════════════════════════

def seed_worker(worker_id: int):
    """Ensure multi-processing DataLoader workers are deterministically seeded."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    t: Union[np.ndarray, pd.Series, torch.Tensor],
    y: Union[np.ndarray, pd.Series, torch.Tensor],
    batch_size: int = 2048,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Any] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Construct a PyTorch DataLoader yielding (x_batch, t_batch, y_batch) float32 tuples.
    """
    dataset = CriteoDataset(X, t, y)
    use_persistent = persistent_workers and (num_workers > 0)
    if generator is None and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    if worker_init_fn is None and num_workers > 0:
        worker_init_fn = seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=use_persistent,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def get_dataloaders(
    data_dir: Optional[str] = None,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 2048,
    label_col: str = "visit",
    treat_col: str = "treatment",
    feature_cols: Optional[List[str]] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_mmap: bool = True,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    # Backward compatibility with callers passing pre-extracted arrays/tensors:
    x_train: Optional[Any] = None,
    t_train: Optional[Any] = None,
    y_train: Optional[Any] = None,
    x_val: Optional[Any] = None,
    t_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    x_test: Optional[Any] = None,
    t_test: Optional[Any] = None,
    y_test: Optional[Any] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Load pre-split Criteo datasets and construct (train_loader, val_loader, test_loader).

    Modes of operation:
    1. Directory mode: Pass `data_dir` pointing to directory containing pre-split
       files (e.g. train_criteo.pt, val_criteo.pt, test_criteo.pt or .csv).
    2. Explicit path mode: Pass `train_path`, `val_path`, and `test_path`.
    3. In-memory mode (backward compatibility): Pass `x_train, t_train, y_train, ...`.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    # ── Mode 3: In-memory arrays / tensors already provided ────────────────────
    if x_train is not None and t_train is not None and y_train is not None:
        train_loader = get_dataloader(
            x_train, t_train, y_train,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            seed=seed,
        )
        val_loader = None
        if x_val is not None and t_val is not None and y_val is not None:
            val_loader = get_dataloader(
                x_val, t_val, y_val,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        test_loader = None
        if x_test is not None and t_test is not None and y_test is not None:
            test_loader = get_dataloader(
                x_test, t_test, y_test,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        return train_loader, val_loader, test_loader

    # ── Resolve file paths ───────────────────────────────────────────────────
    if data_dir is None and train_path is None:
        if os.path.exists(DEFAULT_CRITEO_DIR):
            data_dir = DEFAULT_CRITEO_DIR
            logger.info(f"Using default Criteo dataset directory: {data_dir}")
        else:
            raise ValueError(
                "Either `data_dir` or `train_path` must be provided to load Criteo splits."
            )

    if data_dir is not None:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        if train_path is None:
            train_path = _resolve_split_path(data_dir, "train")
        if val_path is None:
            val_path = _resolve_split_path(data_dir, "val")
        if test_path is None:
            test_path = _resolve_split_path(data_dir, "test")

    # ── Load splits into model-ready tensors ──────────────────────────────────
    logger.info(f"[DataLoader] Loading train split from {train_path}...")
    x_tr, t_tr, y_tr = load_split(
        train_path, label_col=label_col, treat_col=treat_col,
        feature_cols=feature_cols, mmap=use_mmap, max_samples=max_samples,
    )
    logger.info(f"  Train samples: {len(x_tr):,} | Treatment mean: {t_tr.mean().item():.4f} | Outcome mean: {y_tr.mean().item():.4f}")

    val_loader = None
    if val_path and os.path.exists(val_path):
        logger.info(f"[DataLoader] Loading val split from {val_path}...")
        x_va, t_va, y_va = load_split(
            val_path, label_col=label_col, treat_col=treat_col,
            feature_cols=feature_cols, mmap=use_mmap, max_samples=max_samples,
        )
        logger.info(f"  Val samples:   {len(x_va):,} | Treatment mean: {t_va.mean().item():.4f} | Outcome mean: {y_va.mean().item():.4f}")
        val_loader = get_dataloader(
            x_va, t_va, y_va,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    test_loader = None
    if test_path and os.path.exists(test_path):
        logger.info(f"[DataLoader] Loading test split from {test_path}...")
        x_te, t_te, y_te = load_split(
            test_path, label_col=label_col, treat_col=treat_col,
            feature_cols=feature_cols, mmap=use_mmap, max_samples=max_samples,
        )
        logger.info(f"  Test samples:  {len(x_te):,} | Treatment mean: {t_te.mean().item():.4f} | Outcome mean: {y_te.mean().item():.4f}")
        test_loader = get_dataloader(
            x_te, t_te, y_te,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    train_loader = get_dataloader(
        x_tr, t_tr, y_tr,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        seed=seed,
    )

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Helpers & Compatibility
# ══════════════════════════════════════════════════════════════════════════════

def compute_denominators(
    data: Union[pd.DataFrame, torch.Tensor, np.ndarray],
    feature_cols: Optional[List[str]] = None,
) -> List[float]:
    """Compute maximum value per feature column for normalization if needed."""
    if isinstance(data, pd.DataFrame):
        cols = feature_cols if feature_cols is not None else [c for c in data.columns if c not in NON_FEATURE_COLS]
        return [float(data[f].max()) for f in cols]
    elif isinstance(data, torch.Tensor):
        return data.max(dim=0).values.cpu().tolist()
    else:
        return np.asarray(data).max(axis=0).tolist()


def load_dataset(
    data_path: str,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "visit",
    treat_col: str = "treatment",
) -> pd.DataFrame:
    """
    Read a CSV dataset file and return a DataFrame.
    Note: For model training, prefer using get_dataloaders() on pre-split datasets.
    """
    warnings.warn(
        "load_dataset() is a legacy helper. For model training, use get_dataloaders() "
        "to load pre-split datasets created by split_criteo.py.",
        UserWarning,
        stacklevel=2,
    )
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(12)]

    logger.info(f"[load_dataset] Reading: {data_path}")
    df_all = pd.read_csv(data_path)
    logger.info(f"  Rows: {len(df_all):,}")
    return df_all


def split_dataset(*args, **kwargs):
    """
    Deprecated: On-the-fly dataset splitting.
    Please use dataset/Criteo/split_criteo.py to pre-split the dataset and
    preprocess/data_loader.py to load pre-split data.
    """
    warnings.warn(
        "split_dataset in preprocess/data_loader.py is deprecated! "
        "Dataset splitting is now handled by dataset/Criteo/split_criteo.py. "
        "Please pre-split your dataset and use get_dataloaders() to load it.",
        DeprecationWarning,
        stacklevel=2,
    )
    from sklearn.model_selection import train_test_split
    df_all = args[0] if len(args) > 0 else kwargs.get("df_all")
    feature_cols = kwargs.get("feature_cols", [f"f{i}" for i in range(12)])
    label_col = kwargs.get("label_col", "visit")
    treat_col = kwargs.get("treat_col", "treatment")
    test_size = kwargs.get("test_size", 0.1)
    val_ratio = kwargs.get("val_ratio", 1 / 9)
    random_state = kwargs.get("random_state", 42)

    df_train, df_tmp = train_test_split(df_all, test_size=test_size, random_state=random_state)
    df_val, df_test = train_test_split(df_tmp, test_size=val_ratio, random_state=random_state)

    x_train = df_train[feature_cols]; y_train = df_train[label_col]; t_train = df_train[treat_col]
    x_val   = df_val[feature_cols];   y_val   = df_val[label_col];   t_val   = df_val[treat_col]
    x_test  = df_test[feature_cols];  y_test  = df_test[label_col];  t_test  = df_test[treat_col]

    return x_train, y_train, t_train, x_val, y_val, t_val, x_test, y_test, t_test
