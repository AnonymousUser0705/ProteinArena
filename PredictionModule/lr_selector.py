"""
================================================================
Learning Rate Selector — dynamic-sweep algorithm
================================================================

Project-agnostic module for selecting learning rates from a short
training sweep.

Algorithm summary (full spec: algo_lr_determination_spec.md):

  For each LR n in the sweep, train E epochs from a shared initial
  state and record T_{n,i}, V_{n,i} at i = 0..E.

  Per-LR scalars:
      ΔT_n = T_{n,0} − T_{n,E}
      ΔV_n = V_{n,0} − V_{n,E}
      |G_n| = |ΔV_n − ΔT_n|

  Per-pair slopes:
      s_{n,i} = V_{n,i+1} − V_{n,i}    (negative when val drops)

  Normalization (over Filter-1 survivors only):
      ΔV̆_n = ΔV_n / max_n ΔV_n
      ΔT̆_n = ΔT_n / max_n ΔT_n
      |G̃_n| = |G_n| / max_n |G_n|
      s̆_{n,i} = s_{n,i} / max_{n,i} |s_{n,i}|     (single global denom)

  Selection:
      1. Filter ΔV_n > 0
      2. Top-2 by ΔV̆_n
      3. Add LRs with ΔV̆_n / ΔV̆_best > rule3_threshold (default 0.9)
      4a. argmin |G̃_n|                                  (gap only)
      4b. argmin |G̃_n| + α · |1 − s_{n,0}/s_{n,1}|     (gap + stability)

Three layers:

  Layer 1: select_best_lrs(train_losses, val_losses, ...)
      Pure-numerical algorithm. No torch dependency.

  Layer 2: run_lr_sweep(model_factory, ..., lr_grid, num_epochs)
      Trains each LR for E epochs and returns loss trajectories.

  Layer 3: find_best_lrs(...)
      One-shot: sweep + select + (optionally) save artifacts.

Plus serialization helpers:

  save_sweep_results(result, save_dir, log_name) → path to JSON
  load_sweep_lr(json_path, rule)                → float
================================================================
"""

import json
import logging
import os
from os.path import join
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Defaults
# ============================================================
DEFAULT_LR_GRID: List[float] = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
DEFAULT_ALPHA: float = 0.5
DEFAULT_RULE3_THRESHOLD: float = 0.9
DEFAULT_NUM_EPOCHS: int = 2


# ============================================================
# Layer 1 — pure algorithm (numpy / pandas only)
# ============================================================
def select_best_lrs(
    train_losses_per_lr: Dict[float, List[float]],
    val_losses_per_lr: Dict[float, List[float]],
    alpha: float = DEFAULT_ALPHA,
    rule3_threshold: float = DEFAULT_RULE3_THRESHOLD,
) -> Dict:
    """
    Select Rule 4a and Rule 4b winning LRs from per-LR loss trajectories.

    Args:
        train_losses_per_lr: {lr: [T_0, T_1, ..., T_E]}
        val_losses_per_lr:   {lr: [V_0, V_1, ..., V_E]}
                             T_0 / V_0 are pre-training measurements.
        alpha:               Rule 4b stability weight (default 0.5).
        rule3_threshold:     ΔV̆_n / ΔV̆_best admission threshold (default 0.9).

    Returns dict with:
        rule_4a_lr        — gap-only winner (or None if sweep failed)
        rule_4b_lr        — gap + stability winner (or None)
        kept_set          — LRs surviving Rules 1–3
        survivors         — LRs surviving Filter 1
        table             — pandas DataFrame, one row per LR, with all values
        warnings          — list of strings
        alpha, rule3_threshold, num_epochs
    """
    if set(train_losses_per_lr.keys()) != set(val_losses_per_lr.keys()):
        raise ValueError("train and val LR dictionaries must have identical keys")

    # Sort LRs descending so the table reads top-to-bottom from largest to smallest
    lrs = sorted(train_losses_per_lr.keys(), reverse=True)
    if not lrs:
        raise ValueError("No LRs provided")

    n_meas = len(train_losses_per_lr[lrs[0]])
    E = n_meas - 1
    if E < 2:
        raise ValueError(
            f"Need ≥3 measurements per LR (E ≥ 2); got {n_meas}"
        )
    for lr in lrs:
        if len(train_losses_per_lr[lr]) != n_meas:
            raise ValueError(f"LR {lr}: train trajectory length mismatch")
        if len(val_losses_per_lr[lr]) != n_meas:
            raise ValueError(f"LR {lr}: val trajectory length mismatch")

    warnings: List[str] = []

    # --- Per-LR scalars ---
    rows = []
    for lr in lrs:
        T = train_losses_per_lr[lr]
        V = val_losses_per_lr[lr]
        delta_T = T[0] - T[E]
        delta_V = V[0] - V[E]
        G = abs(delta_V - delta_T)
        s_0 = V[1] - V[0]
        s_1 = V[2] - V[1]
        row = {
            "lr": float(lr),
            "T_0": T[0], "T_1": T[1], "T_2": T[2],
            "V_0": V[0], "V_1": V[1], "V_2": V[2],
            "delta_T": delta_T,
            "delta_V": delta_V,
            "G_raw": G,
            "s_0": s_0,
            "s_1": s_1,
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    # --- Filter 1: ΔV > 0 ---
    survivors_mask = df["delta_V"] > 0.0
    survivors = df.loc[survivors_mask, "lr"].tolist()

    if not survivors:
        warnings.append("No LRs passed Filter 1 (no ΔV > 0). Sweep failed.")
        # Fill normalization columns with NaN so the table is still well-formed
        for col in ("delta_T_norm", "delta_V_norm", "G_norm",
                    "s_0_norm", "s_1_norm", "s_ratio", "one_minus_ratio_abs"):
            df[col] = np.nan
        return {
            "rule_4a_lr": None,
            "rule_4b_lr": None,
            "kept_set": [],
            "survivors": [],
            "table": df,
            "warnings": warnings,
            "alpha": alpha,
            "rule3_threshold": rule3_threshold,
            "num_epochs": E,
        }

    # --- Normalization (denominators from survivors only) ---
    df_s = df.loc[survivors_mask]
    max_dT = float(df_s["delta_T"].max())
    max_dV = float(df_s["delta_V"].max())
    max_G = float(df_s["G_raw"].max())
    abs_s = pd.concat([df_s["s_0"], df_s["s_1"]]).abs()
    max_abs_s = float(abs_s.max())

    def _safe_div(numer: pd.Series, denom: float) -> pd.Series:
        return numer / denom if denom and denom > 0 else pd.Series(np.nan, index=numer.index)

    df["delta_T_norm"] = _safe_div(df["delta_T"], max_dT)
    df["delta_V_norm"] = _safe_div(df["delta_V"], max_dV)
    df["G_norm"] = _safe_div(df["G_raw"], max_G)
    df["s_0_norm"] = _safe_div(df["s_0"], max_abs_s)
    df["s_1_norm"] = _safe_div(df["s_1"], max_abs_s)

    # --- Slope ratio (= raw s_0/s_1 thanks to global denom) ---
    def _ratio(row) -> float:
        s1 = row["s_1"]
        if s1 == 0 or not np.isfinite(s1):
            return np.inf
        return row["s_0"] / s1

    df["s_ratio"] = df.apply(_ratio, axis=1)
    df["one_minus_ratio_abs"] = (1.0 - df["s_ratio"]).abs()

    # --- Rules 2 + 3 ---
    df_s = df.loc[survivors_mask].sort_values("delta_V_norm", ascending=False)
    top2 = df_s.head(2)["lr"].tolist()
    extras = df_s.loc[df_s["delta_V_norm"] > rule3_threshold, "lr"].tolist()
    kept_set: List[float] = []
    for lr in top2 + extras:
        if lr not in kept_set:
            kept_set.append(lr)

    df_kept = df[df["lr"].isin(kept_set)].copy()

    # --- Rule 4a: smallest |G̃| ---
    rule_4a_lr = float(df_kept.sort_values("G_norm", ascending=True).iloc[0]["lr"])

    # --- Rule 4b: smallest |G̃| + α · |1 − ratio| ---
    df_kept["rule_4b_score"] = df_kept["G_norm"] + alpha * df_kept["one_minus_ratio_abs"]
    rule_4b_lr = float(
        df_kept.sort_values(["rule_4b_score", "lr"], ascending=[True, True]).iloc[0]["lr"]
    )

    if rule_4a_lr == rule_4b_lr:
        warnings.append(
            f"Rule 4a and Rule 4b selected the same LR ({rule_4a_lr:.1e}); "
            f"ablation reduces to a single training run."
        )
    if len(kept_set) == 1:
        warnings.append(
            f"Kept set has size 1 ({kept_set[0]:.1e}); both rules trivially return it."
        )

    return {
        "rule_4a_lr": rule_4a_lr,
        "rule_4b_lr": rule_4b_lr,
        "kept_set": kept_set,
        "survivors": survivors,
        "table": df,
        "warnings": warnings,
        "alpha": alpha,
        "rule3_threshold": rule3_threshold,
        "num_epochs": E,
    }


# ============================================================
# Layer 2 — orchestrator (runs the actual sweep)
# ============================================================
def run_lr_sweep(
    model_factory: Callable,
    train_loader,
    val_loader,
    forward_fn: Callable,
    criterion,
    device,
    lr_grid: Optional[List[float]] = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    weight_decay: float = 1e-4,
) -> Tuple[Dict[float, List[float]], Dict[float, List[float]]]:
    """
    Train each LR in *lr_grid* for *num_epochs* and return loss trajectories.

    For each LR a fresh model is built via *model_factory*. Records E+1
    measurements per metric: index 0 is pre-training (initial weights),
    indices 1..E are after each training epoch.

    The same initial weights are reused across LRs as long as the caller
    has fixed the random seed before this call (the model_factory is
    invoked once per LR with the same seed-derived RNG state).

    Args:
        model_factory: Callable() -> nn.Module  (un-moved; sweep moves to device)
        train_loader, val_loader: DataLoaders
        forward_fn: Callable(model, batch, criterion, device) -> scalar loss
        criterion: pre-configured loss function
        device: target device
        lr_grid: list of LRs (default DEFAULT_LR_GRID)
        num_epochs: per-LR epochs (default 2)
        weight_decay: Adam weight decay

    Returns:
        (train_losses_per_lr, val_losses_per_lr)
        Each is {lr: [m_0, m_1, ..., m_E]}
    """
    import torch  # local import keeps Layer 1 torch-free
    import random

    if lr_grid is None:
        lr_grid = DEFAULT_LR_GRID

    train_losses_per_lr: Dict[float, List[float]] = {}
    val_losses_per_lr: Dict[float, List[float]] = {}

    logger.info("=" * 70)
    logger.info("LR DYNAMIC SWEEP")
    logger.info(f"  Grid       : {[f'{lr:.1e}' for lr in lr_grid]}")
    logger.info(f"  Epochs/LR  : {num_epochs}")
    logger.info(f"  Measurements per LR: {num_epochs + 1}  (i = 0..{num_epochs})")
    logger.info("=" * 70)

    # Snapshot RNG state to ensure identical initial weights & batch ordering per LR
    rng_state_torch = torch.get_rng_state()
    if torch.cuda.is_available():
        rng_state_cuda = torch.cuda.get_rng_state_all()
    else:
        rng_state_cuda = None
    rng_state_np = np.random.get_state()
    rng_state_py = random.getstate()

    for lr in lr_grid:
        logger.info(f"\n  --- Sweep LR = {lr:.1e} ---")

        # Restore RNG state
        torch.set_rng_state(rng_state_torch)
        if rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(rng_state_cuda)
        np.random.set_state(rng_state_np)
        random.setstate(rng_state_py)

        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Initial measurement (i=0, before any training)
        T_0 = _eval_loss(model, train_loader, forward_fn, criterion, device)
        V_0 = _eval_loss(model, val_loader, forward_fn, criterion, device)
        T_traj: List[float] = [T_0]
        V_traj: List[float] = [V_0]
        logger.info(f"    Init      | T_0={T_0:.6f} | V_0={V_0:.6f}")

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_sum = 0.0
            n_batches = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = forward_fn(model, batch, criterion, device)
                loss.backward()
                optimizer.step()
                train_sum += loss.item()
                n_batches += 1
            T_i = train_sum / max(n_batches, 1)
            V_i = _eval_loss(model, val_loader, forward_fn, criterion, device)
            T_traj.append(T_i)
            V_traj.append(V_i)
            logger.info(f"    Epoch {epoch}   | T_{epoch}={T_i:.6f} | V_{epoch}={V_i:.6f}")

        train_losses_per_lr[lr] = T_traj
        val_losses_per_lr[lr] = V_traj

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_losses_per_lr, val_losses_per_lr


def _eval_loss(model, loader, forward_fn, criterion, device) -> float:
    """Average loss on a loader without gradients."""
    import torch
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            loss = forward_fn(model, batch, criterion, device)
            total += loss.item()
            n += 1
    return total / max(n, 1)


# ============================================================
# Layer 3 — one-shot pipeline (sweep + select + save)
# ============================================================
def find_best_lrs(
    model_factory: Callable,
    train_loader,
    val_loader,
    forward_fn: Callable,
    criterion,
    device,
    lr_grid: Optional[List[float]] = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    alpha: float = DEFAULT_ALPHA,
    rule3_threshold: float = DEFAULT_RULE3_THRESHOLD,
    weight_decay: float = 1e-4,
    save_dir: Optional[str] = None,
    log_name: str = "lr_dynsweep",
) -> Dict:
    """
    Run sweep, select Rule 4a and Rule 4b winners, optionally save JSON+CSV.

    Returns the same dict as select_best_lrs() with two extra entries:
        train_losses_per_lr, val_losses_per_lr
    """
    train_losses, val_losses = run_lr_sweep(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        forward_fn=forward_fn,
        criterion=criterion,
        device=device,
        lr_grid=lr_grid,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
    )

    result = select_best_lrs(
        train_losses_per_lr=train_losses,
        val_losses_per_lr=val_losses,
        alpha=alpha,
        rule3_threshold=rule3_threshold,
    )
    result["train_losses_per_lr"] = train_losses
    result["val_losses_per_lr"] = val_losses

    logger.info("=" * 70)
    logger.info("LR SWEEP — SELECTION SUMMARY")
    logger.info(f"  Survivors (Filter 1) : {[f'{lr:.1e}' for lr in result['survivors']]}")
    logger.info(f"  Kept set (Rules 1–3) : {[f'{lr:.1e}' for lr in result['kept_set']]}")
    for w in result["warnings"]:
        logger.warning(f"  ⚠  {w}")
    logger.info("=" * 70)

    if save_dir is not None:
        save_sweep_results(result, save_dir, log_name)

    return result


# ============================================================
# Serialization
# ============================================================
def save_sweep_results(result: Dict, save_dir: str, log_name: str = "lr_dynsweep") -> str:
    """Save JSON summary + CSV table. Returns JSON path."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = join(save_dir, f"{log_name}_table.csv")
    json_path = join(save_dir, f"{log_name}_results.json")

    result["table"].to_csv(csv_path, index=False)

    payload = {
        "rule_4a_lr": result["rule_4a_lr"],
        "rule_4b_lr": result["rule_4b_lr"],
        "kept_set": result["kept_set"],
        "survivors": result["survivors"],
        "alpha": result["alpha"],
        "rule3_threshold": result["rule3_threshold"],
        "num_epochs": result.get("num_epochs"),
        "warnings": result["warnings"],
        "table": result["table"].to_dict(orient="records"),
        "train_losses_per_lr": result.get("train_losses_per_lr"),
        "val_losses_per_lr": result.get("val_losses_per_lr"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    logger.info(f"  Sweep results saved: {json_path} (+ {os.path.basename(csv_path)})")
    return json_path


def load_sweep_lr(json_path: str, rule: str) -> float:
    """
    Read Rule 4a or Rule 4b winning LR from a sweep results JSON.

    *rule* may be '4a', '4b', 'rule_4a', or 'rule_4b'.
    """
    rule_norm = rule.replace("rule_", "")
    if rule_norm not in ("4a", "4b"):
        raise ValueError(f"rule must be '4a' or '4b', got '{rule}'")
    with open(json_path) as f:
        data = json.load(f)
    key = f"rule_{rule_norm}_lr"
    if key not in data:
        raise KeyError(f"'{key}' not present in {json_path}")
    if data[key] is None:
        raise ValueError(
            f"Sweep failed: {key} is null in {json_path}. "
            f"Warnings: {data.get('warnings')}"
        )
    return float(data[key])


def _json_default(obj):
    """JSON serializer for numpy / torch scalar types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
