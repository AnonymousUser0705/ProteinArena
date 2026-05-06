#!/usr/bin/env python3
"""
================================================================
CAFA5 GO Term Prediction Pipeline
================================================================

Predicts GO term annotations (BPO / CCO / MFO) for proteins from
the CAFA5 challenge using protein embeddings.

Two operating modes for protein embeddings:
  1) Pre-extracted embeddings  – supply --emb_dirs (one or more) + --emb_suffix + --embedding_dim
  2) On-the-fly extraction     – supply --model_name + --model_path
     (optionally --save_embeddings_dir to persist extracted embeddings)

In both modes the script:
  • builds a GO vocabulary from all splits
  • builds a GOTermPredictor prediction head (single / three-head; original / typeN arch)
  • trains from scratch (or loads --checkpoint) with BCEWithLogitsLoss
  • evaluates on train/val/test with protein-centric Fmax (overall + per-namespace)
  • saves predictions NPY, labels NPY, protein-IDs NPY, metrics JSON, GO vocabulary, and checkpoint

Primary metric: protein-centric Fmax.
================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from util_model import MODEL_REGISTRY, extract_embeddings, load_model
from util_data import (
    CAFADataset,
    build_go_to_index,
    build_go_vocabulary,
    cafa_collate_single,
    cafa_collate_three,
    collect_cafa_protein_ids,
    collect_cafa_sequences,
    compute_cafa_pos_weight,
    load_cafa_splits,
    save_embeddings,
)
from util_helper import (
    apply_sweep_lr,
    auto_setup_ddp,
    cleanup_ddp,
    get_device,
    get_rank,
    get_world_size,
    is_main_process,
    make_loader_generator,
    run_lr_dynsweep,
    save_metrics_json,
    seed_worker,
    set_seed,
    setup_logging,
)


# ============================================================
# Argument Parsing
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="CAFA5 GO Term Prediction — pre-extracted or on-the-fly embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--splits_dir", type=str, required=True,
                    help="Directory with train.tsv, val.tsv, test.tsv")

    # ---- Mode 1: pre-extracted embeddings ----
    p.add_argument("--emb_dirs", type=str, nargs="+", default=[],
                    help="One or more directories to search for pre-extracted .pt embeddings (Mode 1). "
                         "Each protein is searched across all dirs in order.")
    p.add_argument("--emb_suffix", type=str, default="_per_tok.pt",
                    help="Filename suffix for embedding files")
    p.add_argument("--embedding_dim", type=int, default=0,
                    help="Embedding dimension (auto-detected in Mode 2)")
    p.add_argument("--foldvision_runs", type=int, default=0,
                   help="If > 0, loads multiple .npz runs for Test-Time Augmentation.")

    # ---- Mode 2: on-the-fly extraction ----
    p.add_argument("--model_name", type=str, default="",
                    choices=[""] + sorted(MODEL_REGISTRY.keys()),
                    help="PLM model name from registry (Mode 2)")
    p.add_argument("--model_path", type=str, default="",
                    help="Path / HuggingFace ID for PLM weights (Mode 2)")
    p.add_argument("--pdb_dir", type=str, default="",
                    help="Directory with .pdb files (required for structure models)")
    p.add_argument("--extraction_batch_size", type=int, default=8,
                    help="Batch size for on-the-fly embedding extraction")
    p.add_argument("--save_embeddings_dir", type=str, default="",
                    help="If set, save extracted embeddings here for later reuse")

    # ---- Prediction head ----
    p.add_argument("--head_mode", type=str, default="single",
                    choices=["single", "three"],
                    help="'single' = one flat output, 'three' = per-namespace heads")
    p.add_argument("--arch_type", type=str, default="original",
                    choices=["original", "type1", "type2", "type3"],
                    help="Prediction head architecture")
    p.add_argument("--hidden_dim", type=int, default=512,
                    help="Hidden dim (only used for 'original' arch_type)")
    p.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout rate (only used for 'original' arch_type)")
    p.add_argument("--checkpoint", type=str, default="",
                    help="Path to trained checkpoint (.pt). Empty → train from scratch")

    # ---- Training ----
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pos_weight", type=str, default="none",
                    choices=["none", "linear", "sqrt"],
                    help="Per-GO-term pos weighting for BCEWithLogitsLoss")
    p.add_argument("--pos_weight_cap", type=float, default=50.0,
                    help="Max cap for pos_weight values")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_dynsweep", action="store_true",
                   help="Run dynamic LR sweep (Rule 4a / 4b selector), save "
                        "sweep_results.json, and continue training with the selected LR. "
                        "Single-process; do NOT invoke under torchrun.")
    p.add_argument("--lr_dynsweep_epochs", type=int, default=2,
                    help="Epochs per LR for the dynamic sweep")
    p.add_argument("--lr_dynsweep_alpha", type=float, default=0.5,
                    help="Rule 4b stability weight α")
    p.add_argument("--lr_dynsweep_threshold", type=float, default=0.9,
                    help="Rule 3 admission threshold (ΔV̆/ΔV̆_best)")
    p.add_argument("--lr_dynsweep_grid", type=float, nargs="+", default=None,
                    help="Custom LR grid for dynamic sweep "
                         "(default: lr_selector.DEFAULT_LR_GRID)")
    p.add_argument("--lr_from_sweep", type=str, default="",
                    help="Path to sweep_results.json from a prior --lr_dynsweep "
                         "run. Overrides --lr at startup.")
    p.add_argument("--sweep_rule", type=str, default="4a", choices=["4a", "4b"],
                    help="Which sweep-selected LR to use (Rule 4a = gap-only, "
                         "Rule 4b = gap + α·stability)")
    p.add_argument("--eval_splits", type=str, nargs="+",
                    default=["train", "val", "test"],
                    choices=["train", "val", "test"])

    # ---- Output ----
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--log_name", type=str, default="cafa_pred")

    # ---- Misc ----
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--early_stopping_patience", type=int, default=0,
                    help="0 = disabled")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    p.add_argument("--log_every_n_steps", type=int, default=50,
                    help="Step-level logging frequency (0 = disabled)")

    return p.parse_args()


# ============================================================
# Architecture configs
# ============================================================
ARCH_CONFIGS = {
    "original": None,
    "type1": [{"out": 1024, "dropout": 0.2}],
    "type2": [{"out": 1024, "dropout": 0.3}, {"out": 512, "dropout": 0.2}],
    "type3": [{"out": 1024, "dropout": 0.3}, {"out": 768, "dropout": 0.25}, {"out": 512, "dropout": 0.2}],
}


def _build_backbone(embedding_dim, arch_type, hidden_dim=512, dropout=0.1):
    """Build shared backbone. Returns (nn.Sequential, final_dim)."""
    if arch_type == "original":
        backbone = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        return backbone, hidden_dim

    layers_cfg = ARCH_CONFIGS[arch_type]
    layers = []
    in_dim = embedding_dim
    for cfg in layers_cfg:
        out_dim = cfg["out"]
        drop = cfg["dropout"]
        layers.extend([
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(drop),
        ])
        in_dim = out_dim
    return nn.Sequential(*layers), in_dim


# ============================================================
# GOTermPredictor
# ============================================================
class GOTermPredictor(nn.Module):
    """
    Prediction head for GO term multi-label classification.

    Modes:
      'single': one flat output head (BPO+CCO+MFO concatenated)
      'three':  separate output heads per namespace

    No sigmoid in forward() — BCEWithLogitsLoss is used for training.
    """

    def __init__(self, embedding_dim, go_vocab, hidden_dim=512, dropout=0.1,
                 mode="single", arch_type="original"):
        super().__init__()
        self.mode = mode
        self.arch_type = arch_type
        self.ns_sizes = {ns: len(terms) for ns, terms in go_vocab.items()}

        self.shared, final_dim = _build_backbone(
            embedding_dim, arch_type, hidden_dim=hidden_dim, dropout=dropout,
        )

        if mode == "single":
            total_output = sum(self.ns_sizes.values())
            self.output_head = nn.Linear(final_dim, total_output)
        elif mode == "three":
            self.bpo_head = nn.Linear(final_dim, self.ns_sizes["BPO"])
            self.cco_head = nn.Linear(final_dim, self.ns_sizes["CCO"])
            self.mfo_head = nn.Linear(final_dim, self.ns_sizes["MFO"])
        else:
            raise ValueError(f"mode must be 'single' or 'three', got '{mode}'")

    def forward(self, x):
        h = self.shared(x)
        if self.mode == "single":
            return self.output_head(h)
        else:
            return {
                "BPO": self.bpo_head(h),
                "CCO": self.cco_head(h),
                "MFO": self.mfo_head(h),
            }


# ============================================================
# Fmax Computation
# ============================================================
def compute_fmax(y_true, y_scores, thresholds=np.linspace(0, 1, 101)):
    """
    Protein-centric Fmax (vectorized).

    Sweeps thresholds, computes per-protein precision/recall,
    averages across proteins, then computes F1.
    Returns (fmax, best_threshold).
    """
    y_true = y_true.astype(np.float32)
    y_scores = y_scores.astype(np.float32)

    # Precompute per-protein positive counts: shape (N_proteins,)
    n_pos = y_true.sum(axis=1)  # fn_if_predict_nothing = n_pos

    fmax = 0.0
    best_threshold = 0.0

    for tau in thresholds:
        y_pred = (y_scores >= tau).astype(np.float32)

        # Vectorized per-protein tp/fp/fn — all shape (N_proteins,)
        tp = (y_true * y_pred).sum(axis=1)
        fp = ((1.0 - y_true) * y_pred).sum(axis=1)
        fn = n_pos - tp

        denom_p = tp + fp
        denom_r = tp + fn

        precision = np.where(denom_p > 0, tp / denom_p, 0.0)
        recall = np.where(denom_r > 0, tp / denom_r, 0.0)

        avg_prec = precision.mean()
        avg_rec = recall.mean()

        if avg_prec + avg_rec > 0:
            f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
            if f1 > fmax:
                fmax = f1
                best_threshold = tau

    return fmax, best_threshold


# ============================================================
# Training
# ============================================================
def train_model(model, train_dataset, val_dataset, args, device,
                go_vocab, collate_fn=None, use_ddp=False, pos_weight=None):
    """Train GOTermPredictor from scratch; return (best_model, train_losses, val_losses)."""
    log_steps = args.log_every_n_steps
    es_patience = args.early_stopping_patience
    nw = args.num_workers
    use_pin = device.type == "cuda"

    if is_main_process():
        logging.info("=" * 70)
        logging.info("TRAINING FROM SCRATCH")
        eff_bs = args.batch_size * get_world_size()
        logging.info(f"  Epochs: {args.num_epochs}  BS/GPU: {args.batch_size}  "
                     f"Eff BS: {eff_bs}  LR: {args.lr}")
        logging.info(f"  Head: {args.head_mode}  Arch: {args.arch_type}  "
                     f"Pos-weight: {args.pos_weight}  "
                     f"Early-stop: {'off' if es_patience == 0 else es_patience}")
        logging.info("=" * 70)

    model = model.to(device)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate_fn, num_workers=nw, pin_memory=use_pin,
        persistent_workers=(nw > 0), drop_last=use_ddp,
        worker_init_fn=seed_worker, generator=make_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=collate_fn,
        num_workers=nw, pin_memory=use_pin,
        persistent_workers=(nw > 0), drop_last=False,
        worker_init_fn=seed_worker,
    )

    total_steps = len(train_loader)

    # --- Loss function ---
    if args.head_mode == "single":
        pw_tensor = pos_weight.to(device) if pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        criteria = None
    else:
        criterion = None
        if pos_weight is not None:
            criteria = {
                ns: nn.BCEWithLogitsLoss(pos_weight=pos_weight[ns].to(device))
                for ns in ("BPO", "CCO", "MFO")
            }
        else:
            criteria = {ns: nn.BCEWithLogitsLoss() for ns in ("BPO", "CCO", "MFO")}

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- train ----
        model.train()
        epoch_loss = 0.0
        running = 0.0
        t0 = time.time()

        for step, (batch_emb, batch_labels, _) in enumerate(train_loader, 1):
            batch_emb = batch_emb.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_emb)

            if args.head_mode == "single":
                batch_labels = batch_labels.to(device, non_blocking=True)
                loss = criterion(output, batch_labels)
            else:
                loss = sum(
                    criteria[ns](output[ns], batch_labels[ns].to(device, non_blocking=True))
                    for ns in ("BPO", "CCO", "MFO")
                ) / 3.0

            loss.backward()
            optimizer.step()

            sl = loss.item()
            epoch_loss += sl
            running += sl

            if is_main_process() and log_steps > 0 and step % log_steps == 0:
                pct = 100.0 * step / total_steps
                logging.info(f"  Epoch {epoch+1} | Step {step}/{total_steps} "
                             f"({pct:.0f}%) | Loss {running / log_steps:.6f}")
                running = 0.0

        avg_train = epoch_loss / total_steps
        train_losses.append(avg_train)

        # ---- val ----
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for batch_emb, batch_labels, _ in val_loader:
                batch_emb = batch_emb.to(device, non_blocking=True)
                
                if batch_emb.ndim == 3:
                    B, N, D = batch_emb.shape
                    batch_emb = batch_emb.view(B * N, D)
                    output = model(batch_emb)
                    if args.head_mode == "single":
                        batch_labels = batch_labels.to(device, non_blocking=True)
                        probs = torch.sigmoid(output).view(B, N, -1).mean(dim=1)
                        clamped_probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
                        avg_logits = torch.log(clamped_probs / (1.0 - clamped_probs))
                        loss = criterion(avg_logits, batch_labels)
                    else:
                        loss = 0.0
                        for ns in ("BPO", "CCO", "MFO"):
                            ns_probs = torch.sigmoid(output[ns]).view(B, N, -1).mean(dim=1)
                            ns_clamped = torch.clamp(ns_probs, 1e-7, 1 - 1e-7)
                            ns_logits = torch.log(ns_clamped / (1.0 - ns_clamped))
                            loss += criteria[ns](ns_logits, batch_labels[ns].to(device, non_blocking=True))
                        loss /= 3.0
                else:
                    output = model(batch_emb)
                    if args.head_mode == "single":
                        batch_labels = batch_labels.to(device, non_blocking=True)
                        loss = criterion(output, batch_labels)
                    else:
                        loss = sum(
                            criteria[ns](output[ns], batch_labels[ns].to(device, non_blocking=True))
                            for ns in ("BPO", "CCO", "MFO")
                        ) / 3.0
                epoch_val += loss.item()

        avg_val = epoch_val / len(val_loader)
        if use_ddp:
            t_val = torch.tensor([avg_val], device=device)
            dist.all_reduce(t_val, op=dist.ReduceOp.AVG)
            avg_val = t_val.item()
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            raw = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
        else:
            no_improve += 1

        if is_main_process():
            elapsed = time.time() - t0
            logging.info(
                f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
                f"Train {avg_train:.6f} | Val {avg_val:.6f} | "
                f"Best {best_val_loss:.6f} | {elapsed:.1f}s"
            )

        if es_patience > 0 and no_improve >= es_patience:
            if is_main_process():
                logging.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Save checkpoint
    if is_main_process():
        ckpt_path = join(args.save_dir, f"{args.log_name}_best_checkpoint.pt")
        torch.save({
            "model_state_dict": best_state,
            "head_mode": args.head_mode,
            "arch_type": args.arch_type,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "best_val_loss": best_val_loss,
            "stopped_epoch": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, ckpt_path)
        logging.info(f"  Checkpoint: {ckpt_path}")

    if use_ddp:
        dist.barrier()

    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(best_state)
    return raw.to(device), train_losses, val_losses


# ============================================================
# Evaluation
# ============================================================
def evaluate_split(model, dataset, split_name, go_vocab, args, device, collate_fn=None):
    """Evaluate on one split. Returns (preds, labels, pids, metrics_dict)."""
    logging.info(f"--- Evaluating {split_name} ({len(dataset)} proteins) ---")
    raw = model.module if hasattr(model, "module") else model
    raw = raw.to(device).eval()

    # num_workers=0 for eval: data is in-memory, avoids pickling large dicts
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    all_preds, all_labels, all_pids = [], [], []
    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (batch_emb, batch_labels, pids) in enumerate(loader, 1):
            if batch_idx % 200 == 0 or batch_idx == n_batches:
                logging.info(f"  Inference: batch {batch_idx}/{n_batches}")
            batch_emb = batch_emb.to(device, non_blocking=True)
            
            # FoldVision Test-Time Augmentation
            if batch_emb.ndim == 3:
                B, N, D = batch_emb.shape
                batch_emb = batch_emb.view(B * N, D)
                output = raw(batch_emb)
                
                if args.head_mode == "single":
                    probs = torch.sigmoid(output).view(B, N, -1).mean(dim=1).cpu().numpy()
                    labels_np = batch_labels.numpy()
                else:
                    # Output is dict
                    probs_list = []
                    for ns in ("BPO", "CCO", "MFO"):
                        # Average the probabilities for each namespace
                        ns_probs = torch.sigmoid(output[ns]).view(B, N, -1).mean(dim=1)
                        probs_list.append(ns_probs)
                    probs = torch.cat(probs_list, dim=1).cpu().numpy()
                    labels_np = torch.cat(
                        [batch_labels[ns] for ns in ("BPO", "CCO", "MFO")], dim=1
                    ).numpy()
            else:
                output = raw(batch_emb)

                if args.head_mode == "single":
                    probs = torch.sigmoid(output).cpu().numpy()
                    labels_np = batch_labels.numpy()
                else:
                    probs = torch.cat(
                        [torch.sigmoid(output[ns]) for ns in ("BPO", "CCO", "MFO")], dim=1
                    ).cpu().numpy()
                    labels_np = torch.cat(
                        [batch_labels[ns] for ns in ("BPO", "CCO", "MFO")], dim=1
                    ).numpy()

            all_preds.append(probs)
            all_labels.append(labels_np)
            all_pids.extend(pids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- Metrics ---
    metrics = {}

    # Overall Fmax
    fmax_overall, best_t = compute_fmax(all_labels, all_preds)
    metrics["overall_fmax"] = fmax_overall
    metrics["overall_threshold"] = best_t
    logging.info(f"  Overall Fmax: {fmax_overall:.4f} (threshold: {best_t:.2f})")

    # Per-namespace Fmax
    ns_sizes = {ns: len(go_vocab[ns]) for ns in ("BPO", "CCO", "MFO")}
    offset = 0
    for ns in ("BPO", "CCO", "MFO"):
        ns_labels = all_labels[:, offset:offset + ns_sizes[ns]]
        ns_preds = all_preds[:, offset:offset + ns_sizes[ns]]
        has_annotation = ns_labels.sum(axis=1) > 0

        if has_annotation.sum() > 0:
            ns_fmax, ns_t = compute_fmax(ns_labels[has_annotation], ns_preds[has_annotation])
            metrics[f"{ns}_fmax"] = ns_fmax
            metrics[f"{ns}_threshold"] = ns_t
            metrics[f"{ns}_proteins"] = int(has_annotation.sum())
            logging.info(f"  {ns} Fmax: {ns_fmax:.4f} (t={ns_t:.2f}, n={has_annotation.sum()})")
        else:
            metrics[f"{ns}_fmax"] = None
            metrics[f"{ns}_threshold"] = None
            metrics[f"{ns}_proteins"] = 0
            logging.info(f"  {ns} Fmax: N/A (no proteins)")
        offset += ns_sizes[ns]

    # Additional stats
    metrics["num_proteins"] = len(all_pids)
    metrics["label_density"] = float(all_labels.sum() / max(all_labels.size, 1))
    metrics["avg_active_terms"] = float(all_labels.sum(axis=1).mean())
    logging.info(f"  Label density: {metrics['label_density']:.6f}")
    logging.info(f"  Avg active GO terms/protein: {metrics['avg_active_terms']:.1f}")

    # Save npy artifacts
    np.save(join(args.save_dir, f"{args.log_name}_{split_name}_predictions.npy"), all_preds)
    np.save(join(args.save_dir, f"{args.log_name}_{split_name}_labels.npy"), all_labels)
    np.save(join(args.save_dir, f"{args.log_name}_{split_name}_protein_ids.npy"),
            np.array(all_pids))
    logging.info(f"  Saved predictions ({all_preds.shape}), labels, protein_ids")

    return all_preds, all_labels, all_pids, metrics


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    if args.lr_dynsweep and args.lr_from_sweep:
        raise ValueError("--lr_dynsweep and --lr_from_sweep are mutually exclusive")
    if args.seed is not None:
        set_seed(args.seed)
    main_t0 = time.time()
    use_ddp = auto_setup_ddp()
    os.makedirs(args.save_dir, exist_ok=True)
    apply_sweep_lr(args)

    if is_main_process():
        setup_logging(args.save_dir, args.log_name)
    else:
        logging.basicConfig(level=logging.WARNING)

    # ---- Determine mode ----
    mode_preextracted = bool(args.emb_dirs)
    mode_onthefly = bool(args.model_name)

    if not mode_preextracted and not mode_onthefly:
        logging.error("Provide either --emb_dirs (Mode 1) or --model_name (Mode 2)")
        cleanup_ddp()
        sys.exit(1)
    if mode_preextracted and mode_onthefly:
        logging.error("Cannot use both emb_dirs and --model_name. Pick one mode.")
        cleanup_ddp()
        sys.exit(1)

    if is_main_process():
        logging.info("=" * 70)
        logging.info("CAFA5 GO Term Prediction Pipeline")
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Mode: {'pre-extracted' if mode_preextracted else 'on-the-fly'}")
        logging.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            if k == "lr":
                continue
            logging.info(f"  {k}: {v}")

    device = get_device(use_ddp)

    # ---- Load splits ----
    splits = load_cafa_splits(args.splits_dir)
    all_protein_ids = collect_cafa_protein_ids(splits)
    if is_main_process():
        logging.info(f"  Unique proteins across splits: {len(all_protein_ids)}")

    # ---- Build GO vocabulary ----
    if is_main_process():
        logging.info("Building GO vocabulary...")
    go_vocab = build_go_vocabulary(args.splits_dir)
    ns_to_idx, flat_to_idx = build_go_to_index(go_vocab)

    total_terms = sum(len(v) for v in go_vocab.values())
    if is_main_process():
        logging.info(f"  BPO: {len(go_vocab['BPO'])}  CCO: {len(go_vocab['CCO'])}  "
                     f"MFO: {len(go_vocab['MFO'])}  Total: {total_terms}")
        # Save vocab for reproducibility
        vocab_path = join(args.save_dir, f"{args.log_name}_go_vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(go_vocab, f, indent=2)
        logging.info(f"  GO vocab saved: {vocab_path}")

    # ============================================================
    # Obtain embeddings
    # ============================================================
    if mode_preextracted:
        # --- Mode 1: load from disk (scan all provided dirs) ---
        if args.embedding_dim <= 0:
            logging.error("--embedding_dim is required in pre-extracted mode")
            cleanup_ddp()
            sys.exit(1)

        embedding_dim = args.embedding_dim
        emb_dirs = args.emb_dirs  # list of one or more directories

        if is_main_process():
            logging.info(f"  Embedding search dirs ({len(emb_dirs)}): {emb_dirs}")

        # For each protein, scan directories in order until found
        embeddings: dict = {}
        missing: list = []

        if args.foldvision_runs > 0:
            from util_data import preload_foldvision_embeddings
            if is_main_process():
                logging.info(f"  Loading FoldVision embeddings from {emb_dirs[0]}...")
            embeddings, missing = preload_foldvision_embeddings(
                all_protein_ids, emb_dirs[0], args.foldvision_runs
            )
        else:
            for pid in all_protein_ids:
                found = False
                for emb_dir in emb_dirs:
                    emb_path = join(emb_dir, f"{pid}{args.emb_suffix}")
                    if os.path.exists(emb_path):
                        emb = torch.load(emb_path, map_location="cpu", weights_only=True)
                        if emb.dim() == 2:
                            emb = emb.mean(dim=0)
                        elif emb.dim() != 1:
                            logging.warning(f"Unexpected shape {emb.shape} for {pid} — skipping")
                            missing.append(pid)
                            found = True
                            break
                        if emb.shape[0] != embedding_dim:
                            logging.warning(
                                f"Dim mismatch for {pid}: expected {embedding_dim}, "
                                f"got {emb.shape[0]} — skipping"
                            )
                            missing.append(pid)
                            found = True
                            break
                        embeddings[pid] = emb
                        found = True
                        break
                if not found:
                    missing.append(pid)

    else:
        # --- Mode 2: on-the-fly extraction ---
        spec = MODEL_REGISTRY[args.model_name]
        embedding_dim = spec.embedding_dim
        args.embedding_dim = embedding_dim

        sequences = collect_cafa_sequences(splits)
        if is_main_process():
            logging.info(f"  Sequences collected: {len(sequences)}")

        model_ctx = load_model(args.model_name, args.model_path, device)
        runtime_embedding_dim = int(model_ctx.extras.get("embedding_dim", spec.embedding_dim))
        if runtime_embedding_dim != embedding_dim and is_main_process():
            logging.info(
                f"  Embedding dim override from model config: {embedding_dim} -> {runtime_embedding_dim}"
            )
        embedding_dim = runtime_embedding_dim
        args.embedding_dim = embedding_dim

        if is_main_process():
            logging.info("  Extracting embeddings on-the-fly...")
        t0 = time.time()
        embeddings = extract_embeddings(
            model_ctx,
            sorted(all_protein_ids),
            sequences=sequences if spec.input_type == "sequence" else None,
            pdb_dir=args.pdb_dir or None,
            batch_size=args.extraction_batch_size,
        )
        if is_main_process():
            logging.info(f"  Extraction: {len(embeddings)} embeddings in {time.time()-t0:.1f}s")

        if args.save_embeddings_dir:
            save_embeddings(embeddings, args.save_embeddings_dir, args.model_name)

        del model_ctx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        missing = [p for p in all_protein_ids if p not in embeddings]

    if is_main_process():
        logging.info(f"  Embeddings available: {len(embeddings)}/{len(all_protein_ids)}")

    if missing and is_main_process():
        logging.warning(f"  Missing embeddings for {len(missing)}/{len(all_protein_ids)} protein(s). "
                        f"These will be skipped.")
        for pid in missing[:20]:
            logging.warning(f"    - {pid}")
        if len(missing) > 20:
            logging.warning(f"    ... and {len(missing) - 20} more")

    # ============================================================
    # Build datasets
    # ============================================================
    collate_fn = cafa_collate_three if args.head_mode == "three" else cafa_collate_single

    datasets = {}
    for name in ("train", "val", "test"):
        if name in splits:
            tsv_path = join(args.splits_dir, f"{name}.tsv")
            datasets[name] = CAFADataset(
                split_tsv=tsv_path,
                embeddings=embeddings,
                go_vocab=go_vocab,
                ns_to_idx=ns_to_idx,
                flat_to_idx=flat_to_idx,
                label_mode=args.head_mode,
                split_name=name,
            )

    # ============================================================
    # Prediction head model
    # ============================================================
    model = GOTermPredictor(
        embedding_dim=embedding_dim,
        go_vocab=go_vocab,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mode=args.head_mode,
        arch_type=args.arch_type,
    )
    total_p = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logging.info(f"  GOTermPredictor: {args.head_mode}/{args.arch_type}, params={total_p:,}")
        if args.arch_type == "original":
            logging.info(f"  Architecture: {embedding_dim} → {args.hidden_dim} → output")
        else:
            cfg = ARCH_CONFIGS[args.arch_type]
            dims = [str(embedding_dim)] + [str(c["out"]) for c in cfg] + ["output"]
            logging.info(f"  Architecture: {' → '.join(dims)}")

    # ============================================================
    # Train or load checkpoint
    # ============================================================
    train_losses, val_losses = None, None

    if args.checkpoint and os.path.exists(args.checkpoint):
        if is_main_process():
            logging.info(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        train_losses = ckpt.get("train_losses")
        val_losses = ckpt.get("val_losses")
        if is_main_process():
            for key in ("head_mode", "arch_type", "embedding_dim", "best_val_loss", "stopped_epoch"):
                if key in ckpt:
                    logging.info(f"    {key}: {ckpt[key]}")
    else:
        if "train" not in datasets or "val" not in datasets:
            logging.error("Need train + val splits to train from scratch")
            cleanup_ddp()
            sys.exit(1)

        pw = None
        if args.pos_weight != "none":
            pw = compute_cafa_pos_weight(
                datasets["train"], go_vocab,
                mode=args.head_mode,
                method=args.pos_weight,
                cap=args.pos_weight_cap,
            )

        # ---- Dynamic LR Sweep (Rule 4a / 4b selector — sweep-only mode) ----
        if args.lr_dynsweep:
            sweep_train_loader = DataLoader(
                datasets["train"], batch_size=args.batch_size,
                shuffle=True, collate_fn=collate_fn,
                num_workers=0, pin_memory=False, drop_last=False,
                generator=make_loader_generator(args.seed),
            )
            sweep_val_loader = DataLoader(
                datasets["val"], batch_size=args.batch_size,
                shuffle=False, collate_fn=collate_fn,
                num_workers=0, pin_memory=False, drop_last=False,
            )

            if args.head_mode == "single":
                pw_sweep = pw.to(device) if pw is not None else None
                sweep_criterion = nn.BCEWithLogitsLoss(pos_weight=pw_sweep)
            else:
                if pw is not None:
                    _sweep_criteria = {
                        ns: nn.BCEWithLogitsLoss(pos_weight=pw[ns].to(device))
                        for ns in ("BPO", "CCO", "MFO")
                    }
                else:
                    _sweep_criteria = {ns: nn.BCEWithLogitsLoss()
                                       for ns in ("BPO", "CCO", "MFO")}

                class _SweepCriterion(nn.Module):
                    def forward(self, output, targets):
                        return sum(
                            _sweep_criteria[ns](
                                output[ns], targets[ns].to(output[ns].device)
                            )
                            for ns in ("BPO", "CCO", "MFO")
                        ) / 3.0
                sweep_criterion = _SweepCriterion()

            _go_vocab_ref = go_vocab
            _head_mode_ref = args.head_mode
            _arch_type_ref = args.arch_type
            _hidden_dim_ref = args.hidden_dim
            _dropout_ref = args.dropout

            def _sweep_model_factory():
                return GOTermPredictor(
                    embedding_dim=embedding_dim,
                    go_vocab=_go_vocab_ref,
                    hidden_dim=_hidden_dim_ref,
                    dropout=_dropout_ref,
                    mode=_head_mode_ref,
                    arch_type=_arch_type_ref,
                )

            def _sweep_forward(model, batch, criterion, dev):
                emb, labels, _ = batch
                emb = emb.to(dev, non_blocking=True)
                if emb.ndim == 3:
                    B, N, D = emb.shape
                    emb = emb.view(B * N, D)
                    output = model(emb)
                    if _head_mode_ref == "single":
                        labels = labels.to(dev, non_blocking=True)
                        preds = torch.sigmoid(output).view(B, N, -1).mean(dim=1)
                        preds = torch.clamp(preds, 1e-7, 1 - 1e-7)
                        preds = torch.log(preds / (1.0 - preds))
                        return criterion(preds, labels)
                    else:
                        labels_dev = {
                            ns: labels[ns].to(dev, non_blocking=True)
                            for ns in ("BPO", "CCO", "MFO")
                        }
                        return_preds = {}
                        for ns in ("BPO", "CCO", "MFO"):
                            ns_preds = torch.sigmoid(output[ns]).view(B, N, -1).mean(dim=1)
                            ns_preds = torch.clamp(ns_preds, 1e-7, 1 - 1e-7)
                            return_preds[ns] = torch.log(ns_preds / (1.0 - ns_preds))
                        return criterion(return_preds, labels_dev)
                else:
                    output = model(emb)
                    if _head_mode_ref == "single":
                        labels = labels.to(dev, non_blocking=True)
                        return criterion(output, labels)
                    else:
                        labels_dev = {
                            ns: labels[ns].to(dev, non_blocking=True)
                            for ns in ("BPO", "CCO", "MFO")
                        }
                        return criterion(output, labels_dev)

            best_lr = run_lr_dynsweep(
                args, _sweep_model_factory,
                sweep_train_loader, sweep_val_loader,
                _sweep_forward, sweep_criterion, device,
            )
            if use_ddp:
                lr_tensor = torch.tensor([best_lr], device=device)
                dist.broadcast(lr_tensor, src=0)
                best_lr = lr_tensor.item()
            args.lr = best_lr

        if use_ddp:
            model = model.to(device)
            model = DDP(model, device_ids=[device.index], output_device=device.index)

        model, train_losses, val_losses = train_model(
            model, datasets["train"], datasets["val"], args, device,
            go_vocab=go_vocab, collate_fn=collate_fn,
            use_ddp=use_ddp, pos_weight=pw,
        )

    # Save training curves
    if train_losses is not None and is_main_process():
        np.savez(
            join(args.save_dir, f"{args.log_name}_curves.npz"),
            train_losses=train_losses, val_losses=val_losses,
        )

    # ============================================================
    # Evaluation
    # ============================================================
    all_metrics = {}
    if is_main_process():
        logging.info("")
        logging.info("=" * 70)
        logging.info("EVALUATION")
        logging.info("=" * 70)

        raw = model.module if hasattr(model, "module") else model
        raw = raw.to(device)

        for sn in args.eval_splits:
            if sn not in datasets:
                logging.warning(f"  Skipping {sn} — not available")
                continue
            logging.info("")
            _, _, _, metrics = evaluate_split(
                raw, datasets[sn], sn, go_vocab, args, device,
                collate_fn=collate_fn,
            )
            all_metrics[sn] = metrics

        # Summary JSON
        if val_losses is not None:
            all_metrics["best_epoch"] = int(np.argmin(val_losses)) + 1
            all_metrics["total_epochs"] = len(val_losses)

        save_metrics_json(
            all_metrics, vars(args),
            join(args.save_dir, f"{args.log_name}_metrics.json"),
        )

        # Summary table
        logging.info("")
        logging.info(f"{'Split':<8} {'Overall':>8} {'BPO':>8} {'CCO':>8} {'MFO':>8}")
        logging.info("-" * 44)
        for sn, m in all_metrics.items():
            if not isinstance(m, dict):
                continue
            overall = f"{m['overall_fmax']:.4f}" if m.get("overall_fmax") is not None else "N/A"
            bpo = f"{m['BPO_fmax']:.4f}" if m.get("BPO_fmax") is not None else "N/A"
            cco = f"{m['CCO_fmax']:.4f}" if m.get("CCO_fmax") is not None else "N/A"
            mfo = f"{m['MFO_fmax']:.4f}" if m.get("MFO_fmax") is not None else "N/A"
            logging.info(f"{sn:<8} {overall:>8} {bpo:>8} {cco:>8} {mfo:>8}")

        logging.info("=" * 70)
        logging.info(f"All outputs in: {args.save_dir}")
        logging.info("Done.")

    if is_main_process():
        total_elapsed = time.time() - main_t0
        logging.info(f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    cleanup_ddp()


if __name__ == "__main__":
    main()
