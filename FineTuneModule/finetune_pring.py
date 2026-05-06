#!/usr/bin/env python3
"""
================================================================
PRING PPI Fine-Tuning Pipeline
================================================================

End-to-end fine-tuning of Protein Language Models for PPI
prediction. Unlike predict_pring.py (frozen embeddings →
lightweight head), this script opens the PLM backbone for
gradient updates alongside the prediction head.

Features:
  - Differential learning rates (backbone vs head)
  - Optional dynamic LR sweep (--lr_dynsweep) using lr_selector
    (single-LR sweep, shared between backbone and head)
  - Layer freezing (--freeze_layers N)
  - Mixed precision (AMP) on CUDA, graceful fallback on MPS/CPU
  - Gradient accumulation (--grad_accum_steps)
  - DDP multi-GPU support (via torchrun)
  - Evaluation with seen/unseen pair-type breakdown

Examples:

  # Full fine-tuning with dynamic LR sweep
  python finetune_pring.py \\
      --splits_dir /path/to/PRING/Splits \\
      --model_name esm2_650M \\
      --model_path /path/to/esm2 \\
      --save_dir /path/to/results \\
      --lr_dynsweep

  # Train with explicit differential LRs
  python finetune_pring.py \\
      --splits_dir /path/to/PRING/Splits \\
      --model_name prott5 \\
      --model_path /path/to/prott5 \\
      --backbone_lr 1e-5 --head_lr 1e-3 \\
      --save_dir /path/to/results

  # Evaluate from checkpoint
  python finetune_pring.py \\
      --splits_dir /path/to/PRING/Splits \\
      --model_name esm2_650M \\
      --model_path /path/to/esm2 \\
      --save_dir /path/to/results \\
      --eval_only --checkpoint /path/to/checkpoint.pt

  # Multi-GPU (DDP)
  torchrun --nproc_per_node=4 finetune_pring.py \\
      --splits_dir /path/to/PRING/Splits \\
      --model_name esm2_650M \\
      --model_path /path/to/esm2 \\
      --save_dir /path/to/results
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
from sklearn.metrics import roc_curve

# ---- Imports from PredictionModule ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PredictionModule"))

from util_model import MODEL_REGISTRY, load_model
from util_data import collect_unique_proteins, load_pring_splits
from util_helper import (
    apply_sweep_lr,
    auto_setup_ddp,
    classify_pair_types,
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
from predict_pring import PPIPredictor

# ---- Imports from FineTuneModule ----
from util_finetune import (
    PPISequenceDataset,
    build_param_groups,
    freeze_backbone_layers,
    get_amp_context,
    ppi_sequence_collate,
    run_lr_dynsweep_finetune,
)
from util_finetune_model import finetune_forward, get_supported_ft_families


# ============================================================
# Argument Parsing
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="PRING PPI Fine-Tuning — end-to-end PLM + prediction head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory with train.tsv, val.tsv, test.tsv "
                        "(columns: protein_a, protein_b, ..., output)")

    # ---- Model ----
    p.add_argument("--model_name", type=str, required=True,
                   choices=sorted(MODEL_REGISTRY.keys()),
                   help="PLM model name from registry")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path / HuggingFace ID for PLM weights")

    # ---- Prediction head ----
    p.add_argument("--arch_type", type=str, default="symmetric",
                   choices=["concat", "symmetric"],
                   help="Prediction head architecture")
    p.add_argument("--hidden_dim", type=int, default=512,
                   help="Hidden dimension in prediction head")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to fine-tuned checkpoint (.pt). Required for --eval_only")

    # ---- Fine-tuning ----
    p.add_argument("--backbone_lr", type=float, default=1e-5,
                   help="Learning rate for PLM backbone")
    p.add_argument("--head_lr", type=float, default=1e-3,
                   help="Learning rate for prediction head")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_layers", type=int, default=0,
                   help="Number of bottom backbone layers to freeze (0 = none)")
    p.add_argument("--use_amp", action="store_true",
                   help="Enable mixed-precision training (CUDA only)")
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["plateau", "cosine_warmup", "none"],
                   help="LR scheduler")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Fraction of total steps for linear warmup (cosine_warmup only)")

    # ---- Dynamic LR sweep (single LR shared between backbone and head) ----
    p.add_argument("--lr_dynsweep", action="store_true",
                   help="Run dynamic LR sweep (Rule 4a / 4b selector) over the "
                        "end-to-end model. The chosen LR is applied to both "
                        "backbone and head. Single-process; do NOT invoke under torchrun.")
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
                        "run. Overrides --backbone_lr and --head_lr at startup.")
    p.add_argument("--sweep_rule", type=str, default="4a", choices=["4a", "4b"],
                   help="Which sweep-selected LR to use")

    # ---- Training ----
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Per-GPU batch size (small — backbone is in graph)")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps for larger effective batch")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max gradient norm for clipping (0 = disabled)")
    p.add_argument("--pos_weight", type=str, default="none",
                   choices=["none", "auto", "sqrt"],
                   help="Positive-class weighting for BCEWithLogitsLoss")
    p.add_argument("--eval_splits", type=str, nargs="+",
                   default=["train", "val", "test"],
                   choices=["train", "val", "test"])

    # ---- Modes ----
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, only evaluate from --checkpoint")

    # ---- Output ----
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--log_name", type=str, default="ft_pring")

    # ---- Misc ----
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 recommended for sequence batches)")
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help="0 = disabled")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--log_every_n_steps", type=int, default=50,
                   help="Step-level logging frequency (0 = disabled)")

    return p.parse_args()


# ============================================================
# End-to-End Model (PLM backbone + PPI head)
# ============================================================
class EndToEndPRINGModel(nn.Module):
    """
    Wraps a PLM backbone (via model_ctx) and a PPIPredictor head into
    a single nn.Module for end-to-end fine-tuning.

    Forward accepts raw sequence strings, runs them through the PLM
    backbone with gradients enabled, then through the PPI head.
    """

    def __init__(self, model_ctx, head: PPIPredictor):
        super().__init__()
        self.model_ctx = model_ctx
        self.backbone = model_ctx.model
        self.head = head

    def forward(self, seqs_a, seqs_b):
        emb_a = finetune_forward(self.model_ctx, seqs_a)
        emb_b = finetune_forward(self.model_ctx, seqs_b)
        return self.head(emb_a, emb_b)


# ============================================================
# Apply sweep LR to args (overwrites both backbone and head LR)
# ============================================================
def _apply_sweep_lr_to_finetune(args) -> None:
    """If --lr_from_sweep is set, override args.backbone_lr / args.head_lr."""
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
    e2e_model: EndToEndPRINGModel,
    train_dataset: PPISequenceDataset,
    val_dataset: PPISequenceDataset,
    args,
    device: torch.device,
    use_ddp: bool = False,
    pos_weight=None,
):
    """End-to-end fine-tuning loop. Returns (model, train_losses, val_losses)."""
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
        collate_fn=ppi_sequence_collate, num_workers=nw,
        pin_memory=use_pin, drop_last=use_ddp,
        worker_init_fn=seed_worker, generator=make_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=ppi_sequence_collate,
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

    # ---- Scheduler ----
    total_training_steps = total_steps * args.num_epochs
    scheduler_is_per_step = False
    scheduler = None
    if args.scheduler == "cosine_warmup":
        warmup_steps = int(total_training_steps * args.warmup_ratio)
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def _cosine_warmup_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(
                1, total_training_steps - warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=_cosine_warmup_lambda)
        scheduler_is_per_step = True
        if is_main_process():
            logging.info(
                f"  Scheduler: cosine_warmup | "
                f"warmup={warmup_steps} steps ({args.warmup_ratio:.0%}), "
                f"total={total_training_steps} steps"
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

        # ---- train ----
        e2e_model.train()
        epoch_loss = 0.0
        running = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (seqs_a, seqs_b, targets, _, _) in enumerate(train_loader, 1):
            targets = targets.to(device, non_blocking=True)

            with autocast_ctx:
                logits = e2e_model(seqs_a, seqs_b)
                loss = criterion(logits, targets) / accum_steps

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

        # ---- val ----
        e2e_model.eval()
        epoch_val = 0.0
        vp, vt = [], []
        with torch.no_grad():
            for seqs_a, seqs_b, targets, _, _ in val_loader:
                targets = targets.to(device, non_blocking=True)
                logits = e2e_model(seqs_a, seqs_b)
                epoch_val += criterion(logits, targets).item()
                vp.append(torch.sigmoid(logits).cpu().numpy())
                vt.append(targets.cpu().numpy())

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
            vp_np = np.concatenate(vp)
            vt_np = np.concatenate(vt)
            from sklearn.metrics import f1_score as f1s, roc_auc_score as aucs
            e_auc = aucs(vt_np, vp_np) if len(np.unique(vt_np)) > 1 else 0.0
            e_f1 = f1s(vt_np, (vp_np >= 0.5).astype(int), zero_division=0)
            elapsed = time.time() - t0
            logging.info(
                f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
                f"Train {avg_train:.6f} | Val {avg_val:.6f} | "
                f"AUROC {e_auc:.4f} | F1 {e_f1:.4f} | "
                f"Best {best_val_loss:.6f} | {elapsed:.1f}s"
            )

        if es_patience > 0 and no_improve >= es_patience:
            if is_main_process():
                logging.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Save best checkpoint
    if is_main_process() and best_state is not None:
        ckpt_path = join(args.save_dir, f"{args.log_name}_best_checkpoint.pt")
        torch.save({
            "backbone_state_dict": best_state["backbone_state_dict"],
            "head_state_dict": best_state["head_state_dict"],
            "model_name": args.model_name,
            "arch_type": args.arch_type,
            "embedding_dim": MODEL_REGISTRY[args.model_name].embedding_dim,
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
    e2e_model: EndToEndPRINGModel,
    dataset: PPISequenceDataset,
    split_name: str,
    args,
    device: torch.device,
    train_proteins=None,
):
    """Evaluate on one split. Returns (DataFrame, metrics_dict)."""
    import pandas as pd

    logging.info(f"--- Evaluating {split_name} ({len(dataset)} pairs) ---")
    raw = e2e_model.module if hasattr(e2e_model, "module") else e2e_model
    raw.eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=ppi_sequence_collate, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    all_preds, all_labels = [], []
    all_a, all_b = [], []

    with torch.no_grad():
        for seqs_a, seqs_b, targets, pids_a, pids_b in loader:
            logits = raw(seqs_a, seqs_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(targets.numpy())
            all_a.extend(pids_a)
            all_b.extend(pids_b)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = compute_metrics(all_labels, all_preds)
    logging.info(
        f"  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
        f"Best F1={metrics['best_f1']:.4f}(t={metrics['best_threshold']:.2f})"
    )
    if metrics["auroc"] is not None:
        logging.info(f"  AUROC={metrics['auroc']:.4f}")
    logging.info(
        f"  Samples={metrics['n_samples']} "
        f"(pos={metrics['n_positive']}, neg={metrics['n_negative']})"
    )

    pair_types = None
    if train_proteins is not None:
        pair_types = classify_pair_types(all_a, all_b, train_proteins)
        for pt in ("seen-seen", "seen-unseen", "unseen-unseen"):
            mask = np.array([t == pt for t in pair_types])
            cnt = mask.sum()
            if cnt > 0:
                pm = compute_metrics(all_labels[mask], all_preds[mask])
                metrics[f"{pt}_accuracy"] = pm["accuracy"]
                metrics[f"{pt}_f1"] = pm["f1"]
                metrics[f"{pt}_auroc"] = pm["auroc"]
                metrics[f"{pt}_n"] = pm["n_samples"]
                auc_s = f"{pm['auroc']:.4f}" if pm["auroc"] is not None else "N/A"
                logging.info(
                    f"  {pt:16s}  Acc={pm['accuracy']:.4f}  "
                    f"F1={pm['f1']:.4f}  AUROC={auc_s}  N={cnt}"
                )
            else:
                metrics[f"{pt}_accuracy"] = None
                metrics[f"{pt}_f1"] = None
                metrics[f"{pt}_auroc"] = None
                metrics[f"{pt}_n"] = 0

    if metrics["auroc"] is not None:
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_path = join(args.save_dir, f"{args.log_name}_{split_name}_roc.npz")
        np.savez(roc_path, fpr=fpr, tpr=tpr, thresholds=thresholds)

    df = pd.DataFrame({
        "protein_a": all_a,
        "protein_b": all_b,
        "output": all_labels.astype(int),
        "prediction_score": np.round(all_preds, 6),
        "prediction": (all_preds >= 0.5).astype(int),
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
        logging.info("PRING PPI Fine-Tuning Pipeline")
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Model: {args.model_name} (family={spec.family})")
        logging.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            logging.info(f"  {k}: {v}")

    # ---- Load splits ----
    splits = load_pring_splits(args.splits_dir)
    sequences = collect_unique_proteins(splits)
    if is_main_process():
        logging.info(f"  Unique proteins: {len(sequences)}")

    # ---- Load PLM backbone ----
    if is_main_process():
        logging.info(f"  Loading PLM backbone: {args.model_name}")
    model_ctx = load_model(args.model_name, args.model_path, device)

    embedding_dim = int(model_ctx.extras.get("embedding_dim", spec.embedding_dim))
    if is_main_process():
        logging.info(f"  Backbone embedding dim: {embedding_dim}")

    if args.freeze_layers > 0:
        freeze_backbone_layers(model_ctx, args.freeze_layers)

    head = PPIPredictor(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        arch_type=args.arch_type,
    )

    e2e_model = EndToEndPRINGModel(model_ctx, head).to(device)

    total_params = sum(p.numel() for p in e2e_model.parameters())
    trainable_params = sum(p.numel() for p in e2e_model.parameters() if p.requires_grad)
    if is_main_process():
        logging.info(
            f"  Total params: {total_params:,}  Trainable: {trainable_params:,}"
        )

    # ---- Build datasets ----
    datasets = {}
    for name in ("train", "val", "test"):
        if name in splits:
            tsv_path = join(args.splits_dir, f"{name}.tsv")
            datasets[name] = PPISequenceDataset(tsv_path, sequences, split_name=name)

    train_proteins = None
    if "train" in datasets:
        pa, pb, _ = datasets["train"].get_protein_ids()
        train_proteins = set(pa) | set(pb)
        if is_main_process():
            logging.info(f"  Training proteins (seen/unseen ref): {len(train_proteins)}")

    # ---- Load checkpoint (eval-only or warm start) ----
    if args.checkpoint and os.path.exists(args.checkpoint):
        if is_main_process():
            logging.info(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "backbone_state_dict" in ckpt:
            e2e_model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        if "head_state_dict" in ckpt:
            e2e_model.head.load_state_dict(ckpt["head_state_dict"])

    # ---- Eval-only mode ----
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
                )
                all_metrics[sn] = metrics

            save_metrics_json(
                all_metrics, vars(args),
                join(args.save_dir, f"{args.log_name}_metrics.json"),
            )

        cleanup_ddp()
        return

    # ---- Training prep ----
    if "train" not in datasets or "val" not in datasets:
        logging.error("Need train + val splits to train")
        cleanup_ddp()
        sys.exit(1)

    pw = None
    if args.pos_weight != "none":
        pw = compute_pos_weight(datasets["train"].pairs, method=args.pos_weight)

    # ---- Dynamic LR sweep (rank 0; broadcast result) ----
    if args.lr_dynsweep:
        sweep_train_loader = DataLoader(
            datasets["train"], batch_size=args.batch_size,
            shuffle=True, collate_fn=ppi_sequence_collate,
            num_workers=0, pin_memory=False, drop_last=False,
            generator=make_loader_generator(args.seed),
        )
        sweep_val_loader = DataLoader(
            datasets["val"], batch_size=args.batch_size,
            shuffle=False, collate_fn=ppi_sequence_collate,
            num_workers=0, pin_memory=False, drop_last=False,
        )
        pw_tensor = pw.to(device) if pw is not None else None
        sweep_criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        def _sweep_forward(model, batch, criterion, dev):
            seqs_a, seqs_b, tgt, _, _ = batch
            tgt = tgt.to(dev, non_blocking=True)
            logits = model(seqs_a, seqs_b)
            return criterion(logits, tgt)

        best_lr = run_lr_dynsweep_finetune(
            args, e2e_model,
            sweep_train_loader, sweep_val_loader,
            _sweep_forward, sweep_criterion, device,
        )
        if use_ddp:
            lr_tensor = torch.tensor([best_lr], device=device)
            dist.broadcast(lr_tensor, src=0)
            best_lr = lr_tensor.item()
        # Single-LR sweep: apply to both backbone and head
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

    # ---- Evaluation ----
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
            f"{'F1':>8} {'AUROC':>8} {'BestF1':>8}"
        )
        logging.info("-" * 58)
        for sn, m in all_metrics.items():
            if not isinstance(m, dict):
                continue
            auc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "N/A"
            logging.info(
                f"{sn:<8} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8} "
                f"{m['best_f1']:>8.4f}"
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
