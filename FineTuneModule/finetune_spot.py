#!/usr/bin/env python3
"""
================================================================
SPOT Protein–Small Molecule Fine-Tuning Pipeline
================================================================

End-to-end fine-tuning of a Protein Language Model + SPOTPredictor
head for binary protein–small-molecule interaction prediction.
The molecule side uses pre-computed (frozen) MolFormer embeddings;
only the PLM backbone receives gradients.

Only the uniform TSV-format pipeline (DAVIS-style) is supported
here — the legacy 'npy' format used by predict_spot.py is being
deprecated and is not carried over to fine-tuning.

Mirrors finetune_pring.py / finetune_davis.py:
  - Differential learning rates (backbone vs head)
  - Optional dynamic LR sweep (--lr_dynsweep) using lr_selector
    (single-LR sweep, shared between backbone and head)
  - Layer freezing, AMP, gradient accumulation, DDP support
  - Seen/unseen pair-type breakdown at evaluation time

Primary metrics: AUROC, F1, MCC.
================================================================
"""

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import matthews_corrcoef, roc_curve

# ---- Imports from PredictionModule ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PredictionModule"))

from util_model import MODEL_REGISTRY, load_model
from util_data import (
    collect_spot_protein_ids_v2,
    collect_spot_sequences_v2,
    collect_spot_smiles,
    load_mol_embeddings,
    load_spot_splits_v2,
)
from util_helper import (
    auto_setup_ddp,
    cleanup_ddp,
    compute_metrics,
    compute_pos_weight,
    get_device,
    get_world_size,
    is_main_process,
    make_loader_generator,
    save_metrics_json,
    seed_worker,
    set_seed,
    setup_logging,
)
from predict_spot import SPOTPredictor, classify_spot_pair_types

# ---- Imports from FineTuneModule ----
from util_finetune import (
    SPOTSequenceDataset,
    build_param_groups,
    freeze_backbone_layers,
    get_amp_context,
    run_lr_dynsweep_finetune,
    spot_sequence_collate,
)
from util_finetune_model import finetune_forward, get_supported_ft_families


# ============================================================
# Argument Parsing
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="SPOT Fine-Tuning — end-to-end PLM + interaction head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory with train.tsv, val.tsv, test.tsv "
                        "(columns: protein_id, sequence, SMILES, output)")

    # ---- Model ----
    p.add_argument("--model_name", type=str, required=True,
                   choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--model_path", type=str, required=True)

    # ---- Molecule features ----
    p.add_argument("--mol_emb_path", type=str, required=True,
                   help="Path to pre-computed MolFormer embeddings (.npy/.pt) "
                        "keyed by SMILES string")
    p.add_argument("--mol_feat_dim", type=int, default=0)

    # ---- Prediction head ----
    p.add_argument("--arch_type", type=str, default="concat",
                   choices=["concat", "bilinear"])
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to fine-tuned checkpoint (.pt). Required for --eval_only")

    # ---- Fine-tuning ----
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_layers", type=int, default=0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["plateau", "cosine_warmup", "none"])
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    # ---- Dynamic LR sweep ----
    p.add_argument("--lr_dynsweep", action="store_true",
                   help="Run dynamic LR sweep (Rule 4a / 4b selector). "
                        "Single-process; do NOT invoke under torchrun.")
    p.add_argument("--lr_dynsweep_epochs", type=int, default=2)
    p.add_argument("--lr_dynsweep_alpha", type=float, default=0.5)
    p.add_argument("--lr_dynsweep_threshold", type=float, default=0.9)
    p.add_argument("--lr_dynsweep_grid", type=float, nargs="+", default=None)
    p.add_argument("--lr_from_sweep", type=str, default="")
    p.add_argument("--sweep_rule", type=str, default="4a", choices=["4a", "4b"])

    # ---- Training ----
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--pos_weight", type=str, default="none",
                   choices=["none", "auto", "sqrt"])
    p.add_argument("--eval_splits", type=str, nargs="+",
                   default=["train", "val", "test"],
                   choices=["train", "val", "test"])

    p.add_argument("--eval_only", action="store_true")

    # ---- Output ----
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--log_name", type=str, default="ft_spot")

    # ---- Misc ----
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=50)

    return p.parse_args()


# ============================================================
# End-to-End Model
# ============================================================
class EndToEndSPOTModel(nn.Module):
    """PLM backbone + SPOTPredictor head."""

    def __init__(self, model_ctx, head: SPOTPredictor):
        super().__init__()
        self.model_ctx = model_ctx
        self.backbone = model_ctx.model
        self.head = head

    def forward(self, seqs, mol_embs):
        prot_emb = finetune_forward(self.model_ctx, seqs)
        return self.head(prot_emb, mol_embs)


# ============================================================
# Sweep-LR application
# ============================================================
def _apply_sweep_lr_to_finetune(args) -> None:
    path = getattr(args, "lr_from_sweep", "") or ""
    if not path:
        return
    rule = getattr(args, "sweep_rule", "4a")
    from lr_selector import load_sweep_lr
    chosen = float(load_sweep_lr(path, rule))
    args.backbone_lr = chosen
    args.head_lr = chosen
    if is_main_process():
        logging.info(f"  LR loaded from sweep ({rule}): {chosen:.6e} "
                     f"(applied to both backbone and head)")
        logging.info(f"    sweep file: {path}")


# ============================================================
# Training
# ============================================================
def train_model(
    e2e_model: EndToEndSPOTModel,
    train_dataset: SPOTSequenceDataset,
    val_dataset: SPOTSequenceDataset,
    args,
    device: torch.device,
    use_ddp: bool = False,
    pos_weight=None,
):
    log_steps = args.log_every_n_steps
    es_patience = args.early_stopping_patience
    nw = args.num_workers
    use_pin = device.type == "cuda"
    accum_steps = args.grad_accum_steps

    if is_main_process():
        logging.info("=" * 70)
        logging.info("FINE-TUNING")
        eff_bs = args.batch_size * accum_steps * get_world_size()
        logging.info(
            f"  Epochs: {args.num_epochs}  BS/GPU: {args.batch_size}  "
            f"Accum: {accum_steps}  Eff BS: {eff_bs}"
        )
        logging.info(
            f"  Backbone LR: {args.backbone_lr}  Head LR: {args.head_lr}  "
            f"Freeze: {args.freeze_layers} layers"
        )
        logging.info(
            f"  Arch: {args.arch_type}  Pos-weight: {args.pos_weight}  "
            f"AMP: {args.use_amp}  Early-stop: "
            f"{'off' if es_patience == 0 else es_patience}"
        )
        logging.info("=" * 70)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=spot_sequence_collate, num_workers=nw,
        pin_memory=use_pin, drop_last=use_ddp,
        worker_init_fn=seed_worker, generator=make_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=spot_sequence_collate,
        num_workers=nw, pin_memory=use_pin, drop_last=False,
        worker_init_fn=seed_worker,
    )

    total_steps = len(train_loader)

    pw_tensor = pos_weight.to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    base = e2e_model.module if use_ddp else e2e_model
    param_groups = build_param_groups(
        base.backbone, base.head,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    total_training_steps = total_steps * args.num_epochs
    scheduler_is_per_step = False
    scheduler = None
    if args.scheduler == "cosine_warmup":
        warmup_steps = int(total_training_steps * args.warmup_ratio)
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def _cw(current_step):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(
                1, total_training_steps - warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=_cw)
        scheduler_is_per_step = True
        if is_main_process():
            logging.info(
                f"  Scheduler: cosine_warmup | warmup={warmup_steps} steps"
            )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
        if is_main_process():
            logging.info("  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    else:
        if is_main_process():
            logging.info("  Scheduler: none (constant LR)")

    if args.use_amp:
        autocast_ctx, scaler = get_amp_context(device)
    else:
        autocast_ctx, scaler = nullcontext(), None

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        e2e_model.train()
        epoch_loss = 0.0
        running = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (seqs, mol_embs, tgt, _, _) in enumerate(train_loader, 1):
            mol_embs = mol_embs.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with autocast_ctx:
                logits = e2e_model(seqs, mol_embs)
                loss = criterion(logits, tgt) / accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % accum_steps == 0 or step == total_steps:
                if args.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        e2e_model.parameters(), args.max_grad_norm
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler_is_per_step:
                    scheduler.step()
                optimizer.zero_grad()

            sl = loss.item() * accum_steps
            epoch_loss += sl
            running += sl

            if is_main_process() and log_steps > 0 and step % log_steps == 0:
                pct = 100.0 * step / total_steps
                logging.info(
                    f"  Epoch {epoch+1} | Step {step}/{total_steps} "
                    f"({pct:.0f}%) | Loss {running / log_steps:.6f}"
                )
                running = 0.0

        avg_train = epoch_loss / total_steps
        train_losses.append(avg_train)

        e2e_model.eval()
        epoch_val = 0.0
        vp, vt = [], []
        with torch.no_grad():
            for seqs, mol_embs, tgt, _, _ in val_loader:
                mol_embs = mol_embs.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                logits = e2e_model(seqs, mol_embs)
                epoch_val += criterion(logits, tgt).item()
                vp.append(torch.sigmoid(logits).cpu().numpy())
                vt.append(tgt.cpu().numpy())

        avg_val = epoch_val / max(len(val_loader), 1)
        if use_ddp:
            t_val = torch.tensor([avg_val], device=device)
            dist.all_reduce(t_val, op=dist.ReduceOp.AVG)
            avg_val = t_val.item()
        val_losses.append(avg_val)

        if not scheduler_is_per_step and scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(avg_val)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr and is_main_process():
                logging.info(f"  LR reduced: {old_lr:.6f} → {new_lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            raw = e2e_model.module if hasattr(e2e_model, "module") else e2e_model
            best_state = {
                "backbone_state_dict": {
                    k: v.cpu().clone() for k, v in raw.backbone.state_dict().items()
                },
                "head_state_dict": {
                    k: v.cpu().clone() for k, v in raw.head.state_dict().items()
                },
            }
        else:
            no_improve += 1

        if is_main_process():
            from sklearn.metrics import f1_score as f1s, roc_auc_score as aucs
            vp_np = np.concatenate(vp)
            vt_np = np.concatenate(vt)
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

    if is_main_process() and best_state is not None:
        ckpt_path = join(args.save_dir, f"{args.log_name}_best_checkpoint.pt")
        torch.save({
            "backbone_state_dict": best_state["backbone_state_dict"],
            "head_state_dict": best_state["head_state_dict"],
            "model_name": args.model_name,
            "arch_type": args.arch_type,
            "protein_dim": MODEL_REGISTRY[args.model_name].embedding_dim,
            "mol_dim": args.mol_feat_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "best_val_loss": best_val_loss,
            "stopped_epoch": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
        }, ckpt_path)
        logging.info(f"  Checkpoint: {ckpt_path}")

    if use_ddp:
        dist.barrier()

    if best_state is not None:
        raw = e2e_model.module if hasattr(e2e_model, "module") else e2e_model
        raw.backbone.load_state_dict(best_state["backbone_state_dict"])
        raw.head.load_state_dict(best_state["head_state_dict"])

    return e2e_model, train_losses, val_losses


# ============================================================
# Evaluation
# ============================================================
def evaluate_split(
    e2e_model: EndToEndSPOTModel,
    dataset: SPOTSequenceDataset,
    split_name: str,
    args,
    device: torch.device,
    train_proteins=None,
    train_mols=None,
):
    import pandas as pd

    logging.info(f"--- Evaluating {split_name} ({len(dataset)} pairs) ---")
    raw = e2e_model.module if hasattr(e2e_model, "module") else e2e_model
    raw.eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=spot_sequence_collate, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    all_preds, all_labels = [], []
    all_pids, all_mids = [], []
    with torch.no_grad():
        for seqs, mol_embs, tgt, pids, mids in loader:
            mol_embs = mol_embs.to(device, non_blocking=True)
            probs = torch.sigmoid(raw(seqs, mol_embs)).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(tgt.numpy())
            all_pids.extend(pids)
            all_mids.extend(mids)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = compute_metrics(all_labels, all_preds)
    pred_binary = (all_preds >= 0.5).astype(int)
    metrics["mcc"] = float(matthews_corrcoef(all_labels, pred_binary))
    best_pred = (all_preds >= metrics["best_threshold"]).astype(int)
    metrics["best_mcc"] = float(matthews_corrcoef(all_labels, best_pred))

    logging.info(
        f"  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
        f"MCC={metrics['mcc']:.4f}"
    )
    logging.info(
        f"  Best F1={metrics['best_f1']:.4f}(t={metrics['best_threshold']:.2f})  "
        f"Best MCC={metrics['best_mcc']:.4f}"
    )
    if metrics["auroc"] is not None:
        logging.info(f"  AUROC={metrics['auroc']:.4f}")
    logging.info(
        f"  Samples={metrics['n_samples']} "
        f"(pos={metrics['n_positive']}, neg={metrics['n_negative']})"
    )

    pair_types = None
    if train_proteins is not None and train_mols is not None:
        pair_types = classify_spot_pair_types(
            all_pids, all_mids, train_proteins, train_mols,
        )
        for pt in ("both_seen", "prot_unseen", "mol_unseen", "both_unseen"):
            mask = np.array([t == pt for t in pair_types])
            cnt = mask.sum()
            if cnt > 0:
                pm = compute_metrics(all_labels[mask], all_preds[mask])
                pm_mcc = float(matthews_corrcoef(
                    all_labels[mask], (all_preds[mask] >= 0.5).astype(int),
                ))
                metrics[f"{pt}_accuracy"] = pm["accuracy"]
                metrics[f"{pt}_f1"] = pm["f1"]
                metrics[f"{pt}_auroc"] = pm["auroc"]
                metrics[f"{pt}_mcc"] = pm_mcc
                metrics[f"{pt}_n"] = pm["n_samples"]
                auc_s = f"{pm['auroc']:.4f}" if pm["auroc"] is not None else "N/A"
                logging.info(
                    f"  {pt:16s}  Acc={pm['accuracy']:.4f}  F1={pm['f1']:.4f}  "
                    f"AUROC={auc_s}  MCC={pm_mcc:.4f}  N={cnt}"
                )
            else:
                metrics[f"{pt}_accuracy"] = None
                metrics[f"{pt}_f1"] = None
                metrics[f"{pt}_auroc"] = None
                metrics[f"{pt}_mcc"] = None
                metrics[f"{pt}_n"] = 0

    if metrics["auroc"] is not None:
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_path = join(args.save_dir, f"{args.log_name}_{split_name}_roc.npz")
        np.savez(roc_path, fpr=fpr, tpr=tpr, thresholds=thresholds)

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
    _apply_sweep_lr_to_finetune(args)

    if is_main_process():
        setup_logging(args.save_dir, args.log_name)
    else:
        logging.basicConfig(level=logging.WARNING)

    device = get_device(use_ddp)

    spec = MODEL_REGISTRY[args.model_name]
    supported = get_supported_ft_families()
    if spec.family not in supported:
        logging.error(
            f"Model '{args.model_name}' (family={spec.family}) is not supported "
            f"for fine-tuning. Supported families: {supported}"
        )
        cleanup_ddp()
        sys.exit(1)

    if is_main_process():
        logging.info("=" * 70)
        logging.info("SPOT Fine-Tuning Pipeline")
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Model: {args.model_name} (family={spec.family})")
        logging.info(f"Molecule embeddings: {args.mol_emb_path}")
        logging.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            logging.info(f"  {k}: {v}")

    splits_df = load_spot_splits_v2(args.splits_dir)
    if not splits_df:
        logging.error(f"No split files found in: {args.splits_dir}")
        cleanup_ddp()
        sys.exit(1)
    all_protein_ids = collect_spot_protein_ids_v2(splits_df)
    all_smiles = collect_spot_smiles(splits_df)
    sequences = collect_spot_sequences_v2(splits_df)
    if is_main_process():
        logging.info(f"  Unique proteins: {len(all_protein_ids)}  "
                     f"Unique SMILES: {len(all_smiles)}")

    if is_main_process():
        logging.info(f"  Loading PLM backbone: {args.model_name}")
    model_ctx = load_model(args.model_name, args.model_path, device)
    embedding_dim = int(model_ctx.extras.get("embedding_dim", spec.embedding_dim))
    args.embedding_dim = embedding_dim
    if is_main_process():
        logging.info(f"  Backbone embedding dim: {embedding_dim}")

    if args.freeze_layers > 0:
        freeze_backbone_layers(model_ctx, args.freeze_layers)

    if is_main_process():
        logging.info(f"  Loading molecule embeddings from: {args.mol_emb_path}")
    mol_features = load_mol_embeddings(args.mol_emb_path)
    first_vec = next(iter(mol_features.values()))
    mol_dim = first_vec.shape[0]
    args.mol_feat_dim = mol_dim
    if is_main_process():
        mol_available = sum(1 for s in all_smiles if s in mol_features)
        logging.info(f"  Molecule features available: "
                     f"{mol_available}/{len(all_smiles)} (dim={mol_dim})")

    head = SPOTPredictor(
        protein_dim=embedding_dim,
        mol_dim=mol_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        arch_type=args.arch_type,
    )
    e2e_model = EndToEndSPOTModel(model_ctx, head).to(device)

    total_params = sum(p.numel() for p in e2e_model.parameters())
    trainable_params = sum(p.numel() for p in e2e_model.parameters() if p.requires_grad)
    if is_main_process():
        logging.info(
            f"  Total params: {total_params:,}  Trainable: {trainable_params:,}"
        )

    datasets = {}
    for name in ("train", "val", "test"):
        if name in splits_df:
            tsv_path = join(args.splits_dir, f"{name}.tsv")
            datasets[name] = SPOTSequenceDataset(
                tsv_path, sequences, mol_features, split_name=name,
            )

    if "train" in datasets and len(datasets["train"]) == 0:
        logging.error(
            "Train dataset has 0 pairs after filtering. Check that molecule "
            "embedding keys (SMILES) match the SMILES column in the TSV files."
        )
        cleanup_ddp()
        sys.exit(1)

    train_proteins, train_mols_set = None, None
    if "train" in datasets:
        pids, smiles, _ = datasets["train"].get_ids()
        train_proteins = set(pids)
        train_mols_set = set(smiles)
        if is_main_process():
            logging.info(f"  Training proteins (seen ref): {len(train_proteins)}")
            logging.info(f"  Training molecules (seen ref): {len(train_mols_set)}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        if is_main_process():
            logging.info(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "backbone_state_dict" in ckpt:
            e2e_model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        if "head_state_dict" in ckpt:
            e2e_model.head.load_state_dict(ckpt["head_state_dict"])

    if args.eval_only:
        if not args.checkpoint:
            logging.error("--eval_only requires --checkpoint")
            cleanup_ddp()
            sys.exit(1)
        all_metrics = {}
        if is_main_process():
            logging.info("")
            logging.info("=" * 70)
            logging.info("EVALUATION (fine-tuned model)")
            logging.info("=" * 70)
            for sn in args.eval_splits:
                if sn not in datasets:
                    logging.warning(f"  Skipping {sn} — not available")
                    continue
                _, metrics = evaluate_split(
                    e2e_model, datasets[sn], sn, args, device,
                    train_proteins=train_proteins,
                    train_mols=train_mols_set,
                )
                all_metrics[sn] = metrics
            save_metrics_json(
                all_metrics, vars(args),
                join(args.save_dir, f"{args.log_name}_metrics.json"),
            )
        cleanup_ddp()
        return

    if "train" not in datasets or "val" not in datasets:
        logging.error("Need train + val splits to train")
        cleanup_ddp()
        sys.exit(1)

    pw = None
    if args.pos_weight != "none":
        pw = compute_pos_weight(datasets["train"].pairs, method=args.pos_weight)

    if args.lr_dynsweep:
        sweep_train_loader = DataLoader(
            datasets["train"], batch_size=args.batch_size,
            shuffle=True, collate_fn=spot_sequence_collate,
            num_workers=0, pin_memory=False, drop_last=False,
            generator=make_loader_generator(args.seed),
        )
        sweep_val_loader = DataLoader(
            datasets["val"], batch_size=args.batch_size,
            shuffle=False, collate_fn=spot_sequence_collate,
            num_workers=0, pin_memory=False, drop_last=False,
        )
        pw_tensor = pw.to(device) if pw is not None else None
        sweep_criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        def _sweep_forward(model, batch, criterion, dev):
            seqs, mol_embs, tgt, _, _ = batch
            mol_embs = mol_embs.to(dev, non_blocking=True)
            tgt = tgt.to(dev, non_blocking=True)
            return criterion(model(seqs, mol_embs), tgt)

        best_lr = run_lr_dynsweep_finetune(
            args, e2e_model,
            sweep_train_loader, sweep_val_loader,
            _sweep_forward, sweep_criterion, device,
        )
        if use_ddp:
            lr_tensor = torch.tensor([best_lr], device=device)
            dist.broadcast(lr_tensor, src=0)
            best_lr = lr_tensor.item()
        args.backbone_lr = best_lr
        args.head_lr = best_lr

    if use_ddp:
        e2e_model = DDP(
            e2e_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    e2e_model, train_losses, val_losses = train_model(
        e2e_model, datasets["train"], datasets["val"], args, device,
        use_ddp=use_ddp, pos_weight=pw,
    )

    if is_main_process() and train_losses:
        np.savez(
            join(args.save_dir, f"{args.log_name}_curves.npz"),
            train_losses=train_losses, val_losses=val_losses,
        )

    all_metrics = {}
    if is_main_process():
        logging.info("")
        logging.info("=" * 70)
        logging.info("EVALUATION (fine-tuned model)")
        logging.info("=" * 70)

        raw = e2e_model.module if hasattr(e2e_model, "module") else e2e_model
        for sn in args.eval_splits:
            if sn not in datasets:
                logging.warning(f"  Skipping {sn} — not available")
                continue
            _, metrics = evaluate_split(
                raw, datasets[sn], sn, args, device,
                train_proteins=train_proteins,
                train_mols=train_mols_set,
            )
            all_metrics[sn] = metrics

        if val_losses:
            all_metrics["best_epoch"] = int(np.argmin(val_losses)) + 1
            all_metrics["total_epochs"] = len(val_losses)

        save_metrics_json(
            all_metrics, vars(args),
            join(args.save_dir, f"{args.log_name}_metrics.json"),
        )

        logging.info("")
        logging.info(
            f"{'Split':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
            f"{'F1':>8} {'AUROC':>8} {'MCC':>8} {'BestF1':>8}"
        )
        logging.info("-" * 66)
        for sn, m in all_metrics.items():
            if not isinstance(m, dict):
                continue
            auc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "N/A"
            logging.info(
                f"{sn:<8} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8} "
                f"{m['mcc']:>8.4f} {m['best_f1']:>8.4f}"
            )

        logging.info("=" * 70)
        logging.info(f"All outputs in: {args.save_dir}")
        logging.info("Done.")

    if is_main_process():
        total_elapsed = time.time() - main_t0
        logging.info(
            f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)"
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
