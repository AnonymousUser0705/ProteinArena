#!/usr/bin/env python3
"""
================================================================
SPOT Protein–Small Molecule Interaction Prediction Pipeline
================================================================

Predicts binary protein–small-molecule interactions using protein
embeddings and pre-computed MolFormer-XL molecule embeddings.

Two operating modes for protein embeddings:
  1) Pre-extracted embeddings  – supply --emb_dir + --emb_suffix + --embedding_dim
  2) On-the-fly extraction     – supply --model_name + --model_path
     (optionally --save_embeddings_dir to persist extracted embeddings)

In both modes the script:
  • builds a lightweight SPOTPredictor prediction head (concat / bilinear)
  • trains from scratch (or loads --checkpoint) with BCEWithLogitsLoss
  • evaluates on train/val/test with seen/unseen pair-type breakdown
  • saves predictions TSV, metrics JSON, ROC curves, and checkpoint

Primary metrics: AUROC, F1, MCC.
================================================================
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import matthews_corrcoef, roc_curve

from util_model import MODEL_REGISTRY, extract_embeddings, load_model
from util_data import (
    SPOTPairDataset,
    SPOTPairDatasetV2,
    collect_spot_mids,
    collect_spot_protein_ids,
    collect_spot_protein_ids_v2,
    collect_spot_sequences_v2,
    collect_spot_smiles,
    load_mol_embeddings,
    load_spot_sequences,
    load_spot_splits,
    load_spot_splits_v2,
    preload_embeddings,
    preload_foldvision_embeddings,
    save_embeddings,
    spot_collate,
    spot_collate_v2,
)
from util_helper import (
    apply_sweep_lr,
    auto_setup_ddp,
    cleanup_ddp,
    compute_metrics,
    compute_pos_weight,
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
        description="SPOT Protein–Small Molecule Interaction Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--data_format", type=str, default="tsv",
                    choices=["tsv", "npy"],
                    help="'tsv' = uniform DAVIS-style TSVs (default); "
                         "'npy' = legacy npy-based pipeline (DEPRECATED)")
    p.add_argument("--splits_dir", type=str, required=True,
                    help="Directory with split files. TSV mode: train.tsv/val.tsv/test.tsv; "
                         "npy mode: labels_dict.npy + *_names.npy + UID_to_Seq.npy")

    # ---- Mode 1: pre-extracted protein embeddings ----
    p.add_argument("--emb_dir", type=str, default="",
                    help="Directory of pre-extracted .pt embeddings (Mode 1)")
    p.add_argument("--emb_suffix", type=str, default="_per_tok.pt",
                    help="Filename suffix for embedding files")
    p.add_argument("--embedding_dim", type=int, default=0,
                    help="Protein embedding dimension (auto-detected in Mode 2)")
    p.add_argument("--emb_type", type=str, default="per_token",
                    choices=["per_token", "mean"],
                    help="Type of stored embeddings")
    p.add_argument("--foldvision_runs", type=int, default=0,
                   help="If > 0, loads multiple .npz runs for Test-Time Augmentation (TSV mode only).")

    # ---- Mode 2: on-the-fly protein extraction ----
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
                    help="If set, save extracted protein embeddings here")

    # ---- Molecule features ----
    p.add_argument("--mol_emb_path", type=str, required=True,
                    help="Path to pre-computed molecule embeddings (.npy/.pt). "
                         "TSV mode: keyed by SMILES string; npy mode: keyed by MID")
    p.add_argument("--mol_feat_dim", type=int, default=0,
                    help="Molecule feature dimension (auto-detected)")

    # ---- Prediction head ----
    p.add_argument("--arch_type", type=str, default="concat",
                    choices=["concat", "bilinear"],
                    help="Prediction head architecture")
    p.add_argument("--hidden_dim", type=int, default=512,
                    help="Hidden dimension in prediction head")
    p.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout rate")
    p.add_argument("--checkpoint", type=str, default="",
                    help="Path to trained checkpoint (.pt). Empty = train from scratch")

    # ---- Training ----
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pos_weight", type=str, default="none",
                    choices=["none", "auto", "sqrt"],
                    help="Positive-class weighting for BCEWithLogitsLoss")
    p.add_argument("--batch_size", type=int, default=64)
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
    p.add_argument("--log_name", type=str, default="spot_pred")

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
# Prediction Head
# ============================================================
class SPOTPredictor(nn.Module):
    """
    Lightweight prediction head for protein–molecule interaction.

    'concat':   [prot_emb ; mol_feat]                    → MLP → 1
    'bilinear': bilinear(prot, mol) + [prot ; mol]       → MLP → 1
    """

    def __init__(self, protein_dim, mol_dim, hidden_dim=512,
                 dropout=0.1, arch_type="concat"):
        super().__init__()
        self.arch_type = arch_type

        if arch_type == "concat":
            input_dim = protein_dim + mol_dim
        elif arch_type == "bilinear":
            self.bilinear = nn.Bilinear(protein_dim, mol_dim, hidden_dim)
            input_dim = hidden_dim + protein_dim + mol_dim
        else:
            raise ValueError(f"arch_type must be 'concat' or 'bilinear', got '{arch_type}'")

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prot_emb, mol_feat):
        if self.arch_type == "concat":
            x = torch.cat([prot_emb, mol_feat], dim=-1)
        else:  # bilinear
            bi = self.bilinear(prot_emb, mol_feat)
            x = torch.cat([bi, prot_emb, mol_feat], dim=-1)
        return self.head(x).squeeze(-1)


# ============================================================
# Seen / Unseen Classification
# ============================================================
def classify_spot_pair_types(pids, mids_list, train_proteins, train_mids):
    """
    Classify pairs into four categories based on training set membership.

    Returns list of: 'both_seen', 'prot_unseen', 'mol_unseen', 'both_unseen'
    """
    pair_types = []
    for pid, mid in zip(pids, mids_list):
        p_in = pid in train_proteins
        m_in = mid in train_mids
        if p_in and m_in:
            pair_types.append("both_seen")
        elif not p_in and m_in:
            pair_types.append("prot_unseen")
        elif p_in and not m_in:
            pair_types.append("mol_unseen")
        else:
            pair_types.append("both_unseen")
    return pair_types


# ============================================================
# Training
# ============================================================
def train_model(model, train_dataset, val_dataset, args, device,
                use_ddp=False, pos_weight=None, collate_fn=spot_collate):
    """Train SPOTPredictor from scratch; return (best_model, train_losses, val_losses)."""
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
        logging.info(f"  Arch: {args.arch_type}  Pos-weight: {args.pos_weight}  "
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

    pw_tensor = pos_weight.to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

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

        for step, (prot, mol, tgt, _, _) in enumerate(train_loader, 1):
            prot = prot.to(device, non_blocking=True)
            mol = mol.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(model(prot, mol), tgt)
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
        vp, vt = [], []
        with torch.no_grad():
            for prot, mol, tgt, _, _ in val_loader:
                prot = prot.to(device, non_blocking=True)
                mol = mol.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                
                if prot.ndim == 3:
                    B, N, D = prot.shape
                    prot = prot.view(B * N, D)
                    if mol is not None:
                        mol = mol.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
                    logits = model(prot, mol)
                    # Average probabilities across N runs
                    probs = torch.sigmoid(logits).view(B, N).mean(dim=1)
                    # For criterion (which expects logits)
                    clamped_probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
                    avg_logits = torch.log(clamped_probs / (1.0 - clamped_probs))
                    epoch_val += criterion(avg_logits, tgt).item()
                    vp.append(probs.cpu().numpy())
                else:
                    logits = model(prot, mol)
                    epoch_val += criterion(logits, tgt).item()
                    vp.append(torch.sigmoid(logits).cpu().numpy())
                    
                vt.append(tgt.cpu().numpy())

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
            vp_np = np.concatenate(vp)
            vt_np = np.concatenate(vt)
            from sklearn.metrics import f1_score as f1s, roc_auc_score as aucs
            e_auc = aucs(vt_np, vp_np) if len(np.unique(vt_np)) > 1 else 0.0
            e_f1 = f1s(vt_np, (vp_np >= 0.5).astype(int), zero_division=0)
            e_mcc = matthews_corrcoef(vt_np, (vp_np >= 0.5).astype(int))
            elapsed = time.time() - t0
            logging.info(
                f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
                f"Train {avg_train:.6f} | Val {avg_val:.6f} | "
                f"AUROC {e_auc:.4f} | F1 {e_f1:.4f} | MCC {e_mcc:.4f} | "
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
            "arch_type": args.arch_type,
            "protein_dim": args.embedding_dim,
            "mol_dim": args.mol_feat_dim,
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
def evaluate_split(model, dataset, split_name, args, device,
                   train_proteins=None, train_mols=None,
                   collate_fn=spot_collate):
    """Evaluate on one split.  Returns (DataFrame, metrics_dict)."""
    import pandas as pd

    logging.info(f"--- Evaluating {split_name} ({len(dataset)} pairs) ---")
    raw = model.module if hasattr(model, "module") else model
    raw = raw.to(device).eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    all_preds, all_labels = [], []
    all_pids, all_mids = [], []

    with torch.no_grad():
        for prot, mol, tgt, pids, mids in loader:
            prot = prot.to(device, non_blocking=True)
            mol = mol.to(device, non_blocking=True)
            
            # FoldVision Test-Time Augmentation
            if prot.ndim == 3:
                B, N, D = prot.shape
                prot = prot.view(B * N, D)
                if mol is not None:
                    mol = mol.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
                
                preds = raw(prot, mol)
                probs = torch.sigmoid(preds).view(B, N).mean(dim=1).cpu().numpy()
            else:
                probs = torch.sigmoid(raw(prot, mol)).cpu().numpy()
                
            all_preds.append(probs)
            all_labels.append(tgt.numpy())
            all_pids.extend(pids)
            all_mids.extend(mids)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Standard binary metrics
    metrics = compute_metrics(all_labels, all_preds)

    # MCC (additional metric used in SPOT literature)
    pred_binary = (all_preds >= 0.5).astype(int)
    metrics["mcc"] = float(matthews_corrcoef(all_labels, pred_binary))
    best_pred = (all_preds >= metrics["best_threshold"]).astype(int)
    metrics["best_mcc"] = float(matthews_corrcoef(all_labels, best_pred))

    logging.info(f"  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
                 f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
                 f"MCC={metrics['mcc']:.4f}")
    logging.info(f"  Best F1={metrics['best_f1']:.4f}(t={metrics['best_threshold']:.2f})  "
                 f"Best MCC={metrics['best_mcc']:.4f}")
    if metrics["auroc"] is not None:
        logging.info(f"  AUROC={metrics['auroc']:.4f}")
    logging.info(f"  Samples={metrics['n_samples']} "
                 f"(pos={metrics['n_positive']}, neg={metrics['n_negative']})")

    # Seen/unseen breakdown
    pair_types = None
    if train_proteins is not None and train_mols is not None:
        pair_types = classify_spot_pair_types(
            all_pids, all_mids, train_proteins, train_mols
        )
        for pt in ("both_seen", "prot_unseen", "mol_unseen", "both_unseen"):
            mask = np.array([t == pt for t in pair_types])
            cnt = mask.sum()
            if cnt > 0:
                pm = compute_metrics(all_labels[mask], all_preds[mask])
                pm_mcc = float(matthews_corrcoef(
                    all_labels[mask], (all_preds[mask] >= 0.5).astype(int)
                ))
                metrics[f"{pt}_accuracy"] = pm["accuracy"]
                metrics[f"{pt}_f1"] = pm["f1"]
                metrics[f"{pt}_auroc"] = pm["auroc"]
                metrics[f"{pt}_mcc"] = pm_mcc
                metrics[f"{pt}_n"] = pm["n_samples"]
                auc_s = f"{pm['auroc']:.4f}" if pm["auroc"] is not None else "N/A"
                logging.info(f"  {pt:16s}  Acc={pm['accuracy']:.4f}  "
                             f"F1={pm['f1']:.4f}  AUROC={auc_s}  "
                             f"MCC={pm_mcc:.4f}  N={cnt}")
            else:
                metrics[f"{pt}_accuracy"] = None
                metrics[f"{pt}_f1"] = None
                metrics[f"{pt}_auroc"] = None
                metrics[f"{pt}_mcc"] = None
                metrics[f"{pt}_n"] = 0

    # ROC curve
    if metrics["auroc"] is not None:
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_path = join(args.save_dir, f"{args.log_name}_{split_name}_roc.npz")
        np.savez(roc_path, fpr=fpr, tpr=tpr, thresholds=thresholds)

    # Predictions TSV
    df = pd.DataFrame({
        "protein_id": all_pids,
        "molecule_id": all_mids,
        "target": all_labels.astype(int),
        "prediction_score": np.round(all_preds, 6),
        "prediction": pred_binary,
    })
    if pair_types is not None:
        df["pair_type"] = pair_types
    pred_path = join(args.save_dir, f"{args.log_name}_{split_name}_predictions.tsv")
    df.to_csv(pred_path, sep="\t", index=False)
    logging.info(f"  Predictions saved: {pred_path}")

    return df, metrics


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

    # ---- Determine protein embedding mode ----
    mode_preextracted = bool(args.emb_dir)
    mode_onthefly = bool(args.model_name)

    if args.data_format == "npy":
        if args.foldvision_runs > 0:
            logging.error("FoldVision TTA (--foldvision_runs) is only supported in 'tsv' mode.")
            cleanup_ddp()
            sys.exit(1)
        if is_main_process():
            logging.warning("----------------------------------------------------------------------")
            logging.warning("DEPRECATION NOTICE: The 'npy' data format mode for SPOT is being ")
            logging.warning("discontinued. Please migrate to the uniform 'tsv' data format. ")
            logging.warning("Advanced features like Test-Time Augmentation will only be available ")
            logging.warning("in the 'tsv' mode going forward.")
            logging.warning("----------------------------------------------------------------------")

    if not mode_preextracted and not mode_onthefly:
        logging.error("Provide either --emb_dir (Mode 1) or --model_name (Mode 2)")
        cleanup_ddp()
        sys.exit(1)
    if mode_preextracted and mode_onthefly:
        logging.error("Cannot use both --emb_dir and --model_name. Pick one mode.")
        cleanup_ddp()
        sys.exit(1)

    if is_main_process():
        logging.info("=" * 70)
        logging.info("SPOT Protein–Small Molecule Interaction Pipeline")
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Protein mode: {'pre-extracted' if mode_preextracted else 'on-the-fly'}")
        logging.info(f"Molecule embeddings: {args.mol_emb_path}")
        logging.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            if k == "lr":
                continue
            logging.info(f"  {k}: {v}")

    device = get_device(use_ddp)

    # ---- Load splits ----
    use_tsv = args.data_format == "tsv"
    if use_tsv:
        # ── TSV format (uniform with DAVIS) ──
        splits_df = load_spot_splits_v2(args.splits_dir)
        all_protein_ids = collect_spot_protein_ids_v2(splits_df)
        all_smiles = collect_spot_smiles(splits_df)
        if is_main_process():
            logging.info(f"  [tsv] Unique proteins across splits: {len(all_protein_ids)}")
            logging.info(f"  [tsv] Unique SMILES across splits: {len(all_smiles)}")
    else:
        # ── Legacy npy format ──
        if is_main_process():
            logging.info("Loading labels dict...")
        labels_dict_path = join(args.splits_dir, "labels_dict.npy")
        labels_dict = np.load(labels_dict_path, allow_pickle=True).item()
        labels_dict = {str(k): int(v) for k, v in labels_dict.items()}
        if is_main_process():
            logging.info(f"  Labels dict: {len(labels_dict)} entries")

        splits = load_spot_splits(args.splits_dir, labels_dict)
        all_protein_ids = collect_spot_protein_ids(splits)
        all_mids = collect_spot_mids(splits)
        if is_main_process():
            logging.info(f"  [npy] Unique proteins across splits: {len(all_protein_ids)}")
            logging.info(f"  [npy] Unique molecules across splits: {len(all_mids)}")

    # ============================================================
    # Obtain protein embeddings
    # ============================================================
    if mode_preextracted:
        if args.embedding_dim <= 0:
            logging.error("--embedding_dim is required in pre-extracted mode")
            cleanup_ddp()
            sys.exit(1)

        if is_main_process():
            logging.info(f"  Loading protein embeddings from: {args.emb_dir}")
        if args.foldvision_runs > 0:
            prot_embeddings, missing = preload_foldvision_embeddings(
                all_protein_ids, args.emb_dir, args.foldvision_runs
            )
        else:
            prot_embeddings, missing = preload_embeddings(
                all_protein_ids, args.emb_dir, args.emb_suffix,
                args.embedding_dim, args.emb_type,
            )
        embedding_dim = args.embedding_dim

    else:
        spec = MODEL_REGISTRY[args.model_name]
        embedding_dim = spec.embedding_dim
        args.embedding_dim = embedding_dim

        sequences = collect_spot_sequences_v2(splits_df) if use_tsv else load_spot_sequences(args.splits_dir)
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
            logging.info("  Extracting protein embeddings on-the-fly...")
        t0 = time.time()
        prot_embeddings = extract_embeddings(
            model_ctx,
            sorted(all_protein_ids),
            sequences=sequences if spec.input_type == "sequence" else None,
            pdb_dir=args.pdb_dir or None,
            batch_size=args.extraction_batch_size,
        )
        if is_main_process():
            logging.info(f"  Extraction: {len(prot_embeddings)} embeddings "
                         f"in {time.time()-t0:.1f}s")

        if args.save_embeddings_dir:
            save_embeddings(prot_embeddings, args.save_embeddings_dir, args.model_name)

        del model_ctx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        missing = [p for p in all_protein_ids if p not in prot_embeddings]

    if is_main_process():
        logging.info(f"  Protein embeddings available: "
                     f"{len(prot_embeddings)}/{len(all_protein_ids)}")
        if missing:
            logging.warning(f"  Missing protein embeddings: {len(missing)}")

    # ============================================================
    # Obtain molecule features
    # ============================================================
    if is_main_process():
        logging.info(f"  Loading molecule embeddings from: {args.mol_emb_path}")
    mol_features = load_mol_embeddings(args.mol_emb_path)
    first_vec = next(iter(mol_features.values()))
    mol_dim = first_vec.shape[0]

    args.mol_feat_dim = mol_dim
    if is_main_process():
        if use_tsv:
            mol_available = sum(1 for s in all_smiles if s in mol_features)
            logging.info(f"  Molecule features available: "
                         f"{mol_available}/{len(all_smiles)} (dim={mol_dim})")
        else:
            mol_available = sum(1 for m in all_mids if m in mol_features)
            logging.info(f"  Molecule features available: "
                         f"{mol_available}/{len(all_mids)} (dim={mol_dim})")

    # ============================================================
    # Build datasets
    # ============================================================
    datasets = {}
    collate_fn = spot_collate_v2 if use_tsv else spot_collate

    if use_tsv:
        for name in ("train", "val", "test"):
            if name in splits_df:
                tsv_path = join(args.splits_dir, f"{name}.tsv")
                datasets[name] = SPOTPairDatasetV2(
                    tsv_path, prot_embeddings, mol_features, split_name=name,
                )
    else:
        for name in ("train", "val", "test"):
            if name in splits:
                datasets[name] = SPOTPairDataset(
                    splits[name], prot_embeddings, mol_features, split_name=name,
                )

    # Guard: abort early if datasets are empty (e.g. key mismatch)
    if "train" in datasets and len(datasets["train"]) == 0:
        logging.error(
            "Train dataset has 0 pairs after filtering. Check that molecule "
            "embedding keys match the split keys (SMILES for tsv, MIDs for npy)."
        )
        cleanup_ddp()
        sys.exit(1)

    # Identify training proteins / molecules for seen/unseen analysis
    train_proteins, train_mols_set = None, None
    if "train" in datasets:
        if use_tsv:
            pids, smiles_list, _ = datasets["train"].get_ids()
            train_proteins = set(pids)
            train_mols_set = set(smiles_list)
        else:
            pids, mids, _ = datasets["train"].get_ids()
            train_proteins = set(pids)
            train_mols_set = set(mids)
        if is_main_process():
            logging.info(f"  Training proteins (seen ref): {len(train_proteins)}")
            logging.info(f"  Training molecules (seen ref): {len(train_mols_set)}")

    # ============================================================
    # Prediction head model
    # ============================================================
    model = SPOTPredictor(
        protein_dim=embedding_dim,
        mol_dim=mol_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        arch_type=args.arch_type,
    )
    total_p = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logging.info(f"  SPOTPredictor: {args.arch_type}, "
                     f"prot_dim={embedding_dim}, mol_dim={mol_dim}, "
                     f"params={total_p:,}")

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
    else:
        if "train" not in datasets or "val" not in datasets:
            logging.error("Need train + val splits to train from scratch")
            cleanup_ddp()
            sys.exit(1)

        pw = None
        if args.pos_weight != "none":
            pw = compute_pos_weight(datasets["train"].pairs, method=args.pos_weight)

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
            pw_tensor = pw.to(device) if pw is not None else None
            sweep_criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

            _emb_dim_ref = embedding_dim
            _mol_dim_ref = mol_dim

            def _sweep_model_factory():
                return SPOTPredictor(
                    protein_dim=_emb_dim_ref,
                    mol_dim=_mol_dim_ref,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    arch_type=args.arch_type,
                )

            def _sweep_forward(model, batch, criterion, dev):
                prot, mol, tgt, _, _ = batch
                prot = prot.to(dev, non_blocking=True)
                mol = mol.to(dev, non_blocking=True)
                tgt = tgt.to(dev, non_blocking=True)
                if prot.ndim == 3:
                    B, N, D = prot.shape
                    prot = prot.view(B * N, D)
                    if mol is not None:
                        mol = mol.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
                    preds = model(prot, mol)
                    # test-time pooling logic for SPOT
                    preds = torch.sigmoid(preds).view(B, N).mean(dim=1)
                    # convert mean probabilities back to logits for BCEWithLogitsLoss
                    # stable logit: clamp to avoid log(0)
                    preds = torch.clamp(preds, 1e-7, 1 - 1e-7)
                    preds = torch.log(preds / (1.0 - preds))
                    return criterion(preds, tgt)
                else:
                    return criterion(model(prot, mol), tgt)

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
            use_ddp=use_ddp, pos_weight=pw, collate_fn=collate_fn,
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
            _, metrics = evaluate_split(
                raw, datasets[sn], sn, args, device,
                train_proteins=train_proteins,
                train_mols=train_mols_set,
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
        logging.info(f"{'Split':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
                     f"{'F1':>8} {'AUROC':>8} {'MCC':>8} {'BestF1':>8}")
        logging.info("-" * 66)
        for sn, m in all_metrics.items():
            if not isinstance(m, dict):
                continue
            auc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "N/A"
            logging.info(f"{sn:<8} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                         f"{m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8} "
                         f"{m['mcc']:>8.4f} {m['best_f1']:>8.4f}")

        logging.info("=" * 70)
        logging.info(f"All outputs in: {args.save_dir}")
        logging.info("Done.")

    if is_main_process():
        total_elapsed = time.time() - main_t0
        logging.info(f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    cleanup_ddp()


if __name__ == "__main__":
    main()
