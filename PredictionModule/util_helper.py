"""
================================================================
Shared Infrastructure — Helpers for PredictionModule
================================================================

DDP setup, logging, device selection, metrics, checkpoint I/O,
pos-weight computation, and dynamic LR sweep glue.
================================================================
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from os.path import join
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: seed numpy + random per worker from torch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Seeded torch.Generator for DataLoader shuffling. None if seed is None."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ============================================================
# DDP Helpers
# ============================================================
def is_main_process() -> bool:
    """True when rank == 0 or DDP is inactive."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def auto_setup_ddp() -> bool:
    """
    Auto-detect and initialise DDP.

    Activated when LOCAL_RANK env is set, CUDA is available, and WORLD_SIZE > 1.
    Returns True if DDP was initialised.
    """
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return False
    if not torch.cuda.is_available():
        return False

    local_rank = int(local_rank_str)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return True


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# Logging
# ============================================================
def setup_logging(save_dir: str, log_name: str) -> str:
    """Configure logging to file + stdout.  Returns log file path."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = join(save_dir, f"{log_name}.log")

    root = logging.getLogger()
    root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Log file: {log_path}")
    return log_path


# ============================================================
# Device Selection
# ============================================================
def get_device(use_ddp: bool = False) -> torch.device:
    """Auto-select: CUDA (rank-aware for DDP) > MPS > CPU."""
    if use_ddp and dist.is_initialized():
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        if is_main_process():
            logging.info(f"DDP mode — {dist.get_world_size()} GPUs")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


# ============================================================
# Metrics
# ============================================================
def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                    threshold: float = 0.5) -> Dict:
    """
    Compute binary classification metrics.

    Returns dict with accuracy, precision, recall, f1, auroc,
    best_f1, best_threshold (threshold that maximises F1), and counts.
    """
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
    else:
        metrics["auroc"] = None
        logging.warning("  AUROC undefined — only one class present")

    # Sweep for best F1 threshold
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1_t = f1_score(y_true, (y_scores >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_t = t

    metrics["best_f1"] = float(best_f1)
    metrics["best_threshold"] = float(round(best_t, 2))
    return metrics


def classify_pair_types(
    pids_a: List[str], pids_b: List[str], train_proteins: Set[str]
) -> List[str]:
    """Classify pairs as seen-seen / seen-unseen / unseen-unseen."""
    pair_types = []
    for a, b in zip(pids_a, pids_b):
        a_in = a in train_proteins
        b_in = b in train_proteins
        if a_in and b_in:
            pair_types.append("seen-seen")
        elif a_in or b_in:
            pair_types.append("seen-unseen")
        else:
            pair_types.append("unseen-unseen")
    return pair_types


def compute_pos_weight(pairs, method: str = "auto") -> torch.Tensor:
    """
    Compute pos class weight from (pid_a, pid_b, target) list.

    *method*: 'auto' → neg/pos,  'sqrt' → sqrt(neg/pos).
    """
    targets = np.array([p[2] for p in pairs], dtype=np.float64)
    n_pos = targets.sum()
    n_neg = len(targets) - n_pos

    if n_pos == 0:
        logging.warning("  No positive samples — pos_weight = 1.0")
        return torch.tensor(1.0)

    ratio = n_neg / n_pos
    if method == "sqrt":
        ratio = np.sqrt(ratio)

    pw = torch.tensor(ratio, dtype=torch.float32)
    logging.info(f"  pos_weight ({method}): {pw.item():.4f}  "
                 f"(pos={int(n_pos)}, neg={int(n_neg)})")
    return pw


# ============================================================
# Checkpoint I/O
# ============================================================
def save_checkpoint(state_dict: dict, metadata: dict, save_path: str) -> None:
    """Save model checkpoint with metadata."""
    payload = {"model_state_dict": state_dict}
    payload.update(metadata)
    torch.save(payload, save_path)
    logging.info(f"  Checkpoint saved: {save_path}")


def load_checkpoint(path: str, device: str = "cpu"):
    """Load checkpoint dict."""
    return torch.load(path, map_location=device, weights_only=False)


# ============================================================
# Summary JSON
# ============================================================
def save_metrics_json(all_metrics: dict, config: dict, save_path: str) -> None:
    """Write a JSON summary of metrics + config."""
    serializable = {}
    for split, metrics in all_metrics.items():
        if isinstance(metrics, dict):
            serializable[split] = {
                k: (float(v) if isinstance(v, (np.floating, float)) and v is not None
                    else int(v) if isinstance(v, (np.integer, int)) and v is not None
                    else v)
                for k, v in metrics.items()
            }
        else:
            serializable[split] = metrics
    serializable["_config"] = {k: str(v) for k, v in config.items()}

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logging.info(f"  Metrics JSON saved: {save_path}")


# ============================================================
# Regression Metrics
# ============================================================
def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_predictors: int = 0
) -> Dict:
    """
    Compute regression metrics for protein fitness / binding-score prediction.

    Spearman ρ is the primary metric — the standard for DMS benchmarks,
    robust to score-scale differences across models.

    Args:
        n_predictors: Number of input features (embedding_dim). When > 0,
            adjusted R² is computed to penalise models with more parameters,
            enabling fairer cross-model comparison.

    Returns dict with: mse, mae, rmse, r2, adjusted_r2, pearson_r, pearson_p,
                       spearman_r, spearman_p, n_samples.
    """
    from scipy import stats

    n = int(len(y_true))
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float("nan") if ss_tot <= 0.0 else float(1.0 - (ss_res / ss_tot))

    # Adjusted R²: penalises higher-dimensional embeddings so that models
    # with different embedding_dim can be compared fairly.
    if n_predictors > 0 and n > n_predictors + 1 and ss_tot > 0.0:
        adjusted_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - n_predictors - 1))
    else:
        adjusted_r2 = float("nan")

    if len(np.unique(y_true)) < 2:
        logging.warning("  Spearman/Pearson undefined — only one unique target value")
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    else:
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_samples": n,
    }


# ============================================================
# DTA-Specific Regression Metrics
# ============================================================
def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Concordance Index (CI) for drug–target affinity prediction.

    CI measures the probability that for any two drug–target pairs with
    different true affinities, the model correctly ranks them.  It is the
    standard primary metric in DTA literature (DeepDTA, GraphDTA, etc.).

    CI = (# concordant pairs) / (# comparable pairs)

    A CI of 0.5 is random; 1.0 is perfect ranking.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    concordant = 0.0
    comparable = 0.0

    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1.0
            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
            else:
                if y_pred[i] < y_pred[j]:
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5

    if comparable == 0.0:
        return 0.0
    return float(concordant / comparable)


def rm2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute rm² (squared correlation coefficient with penalty).

    Used in DTA literature to penalise systematic over/under-prediction.
    rm² = r² × (1 - sqrt(|r² - r₀²|))
    where r₀² is the coefficient of determination forced through the origin.
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    r, _ = stats.pearsonr(y_true, y_pred)
    r2 = r ** 2

    # r₀²: determination coefficient for regression forced through origin
    #   y_pred = k * y_true  →  k = sum(y_true * y_pred) / sum(y_true²)
    ss_true = np.sum(y_true ** 2)
    if ss_true == 0.0:
        return 0.0
    k = np.sum(y_true * y_pred) / ss_true
    y_pred_origin = k * y_true
    ss_res = np.sum((y_pred - y_pred_origin) ** 2)
    ss_tot = np.sum((y_pred - np.mean(y_pred)) ** 2)
    r0_sq = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    rm2 = r2 * (1.0 - np.sqrt(abs(r2 - r0_sq)))
    return float(rm2)


def compute_dta_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_predictors: int = 0
) -> Dict:
    """
    Compute regression metrics for drug–target affinity prediction.

    Primary metric: Concordance Index (CI) — standard in DTA benchmarks.

    Also includes: MSE, RMSE, MAE, R², adjusted R², Pearson r, Spearman ρ,
    and rm² (squared correlation with penalty for systematic bias).

    Args:
        n_predictors: Number of input features (protein_dim + mol_dim).
            When > 0, adjusted R² is computed to penalise models with
            higher-dimensional embeddings.
    """
    # Get standard regression metrics
    base = compute_regression_metrics(y_true, y_pred, n_predictors=n_predictors)

    # Add DTA-specific metrics
    ci = concordance_index(y_true, y_pred)
    rm2 = rm2_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float("nan")

    base["ci"] = ci
    base["rm2"] = rm2
    return base


# ============================================================
# Dynamic LR Sweep — Glue Helpers
# ============================================================
def apply_sweep_lr(args) -> None:
    """
    If args.lr_from_sweep is set, override args.lr from the sweep JSON.

    No-op when the flag is empty. Logs the chosen LR for traceability.
    Works on every rank (DDP-safe — pure file read).
    """
    path = getattr(args, "lr_from_sweep", "") or ""
    if not path:
        return
    rule = getattr(args, "sweep_rule", "4a")
    from lr_selector import load_sweep_lr
    chosen = load_sweep_lr(path, rule)
    args.lr = float(chosen)
    if is_main_process():
        logging.info(f"  LR loaded from sweep ({rule}): {args.lr:.6e}")
        logging.info(f"    sweep file: {path}")


def run_lr_dynsweep(
    args,
    model_factory,
    train_loader,
    val_loader,
    forward_fn,
    criterion,
    device,
) -> float:
    """
    Run the dynamic LR sweep on rank 0, save results JSON+CSV, and return the chosen LR.
    Other ranks (if any) return 0.0. The main script must broadcast the returned LR.
    """
    from lr_selector import find_best_lrs, DEFAULT_LR_GRID

    if not is_main_process():
        return 0.0

    grid = getattr(args, "lr_dynsweep_grid", None) or DEFAULT_LR_GRID
    n_epochs = int(getattr(args, "lr_dynsweep_epochs", 2))
    alpha = float(getattr(args, "lr_dynsweep_alpha", 0.5))
    threshold = float(getattr(args, "lr_dynsweep_threshold", 0.9))
    sweep_rule = getattr(args, "sweep_rule", "4a")

    res = find_best_lrs(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        forward_fn=forward_fn,
        criterion=criterion,
        device=device,
        lr_grid=grid,
        num_epochs=n_epochs,
        alpha=alpha,
        rule3_threshold=threshold,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        log_name=f"{args.log_name}_dynsweep",
    )
    
    chosen_lr = res.get(f"rule_{sweep_rule}_lr")
    if chosen_lr is None:
        raise RuntimeError(
            f"Dynamic LR sweep did not produce a Rule {sweep_rule} winner. "
            f"Warnings: {res.get('warnings')}"
        )

    logging.info(f"  Dynamic sweep chosen LR (Rule {sweep_rule}): {chosen_lr:.6e}")
    return float(chosen_lr)
