#!/usr/bin/env python3
"""
================================================================
CAFA5 GO Term Fine-Tuning Pipeline
================================================================

End-to-end fine-tuning of a Protein Language Model + GOTermPredictor
head for multi-label GO term annotation (BPO / CCO / MFO).

Mirrors finetune_pring.py / finetune_davis.py:
  - Differential learning rates (backbone vs head)
  - Optional dynamic LR sweep (--lr_dynsweep) using lr_selector
    (single-LR sweep, shared between backbone and head)
  - Layer freezing, AMP, gradient accumulation, DDP support
  - 'single' (flat) or 'three' (per-namespace) head modes
  - Per-GO-term pos-weighting

Primary metric: protein-centric Fmax.
================================================================
"""

import argparse
import json
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

# ---- Imports from PredictionModule ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PredictionModule"))

from util_model import MODEL_REGISTRY, load_model
from util_data import (
    build_go_to_index,
    build_go_vocabulary,
    collect_cafa_protein_ids,
    compute_cafa_pos_weight,
    load_cafa_splits,
)
from util_helper import (
    auto_setup_ddp,
    cleanup_ddp,
    get_device,
    get_world_size,
    is_main_process,
    make_loader_generator,
    save_metrics_json,
    seed_worker,
    set_seed,
    setup_logging,
)
from predict_cafa import ARCH_CONFIGS, GOTermPredictor, compute_fmax

# ---- Imports from FineTuneModule ----
from util_finetune import (
    CAFASequenceDataset,
    build_param_groups,
    cafa_sequence_collate_single,
    cafa_sequence_collate_three,
    freeze_backbone_layers,
    get_amp_context,
    run_lr_dynsweep_finetune,
)
from util_finetune_model import finetune_forward, get_supported_ft_families


# ============================================================
# Argument Parsing
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="CAFA5 GO Term Fine-Tuning — end-to-end PLM + multi-label head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory with train.tsv, val.tsv, test.tsv "
                        "(columns: protein_id, sequence, output)")

    # ---- Model ----
    p.add_argument("--model_name", type=str, required=True,
                   choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--model_path", type=str, required=True)

    # ---- Prediction head ----
    p.add_argument("--head_mode", type=str, default="single",
                   choices=["single", "three"],
                   help="'single' = one flat output, 'three' = per-namespace heads")
    p.add_argument("--arch_type", type=str, default="original",
                   choices=["original", "type1", "type2", "type3"])
    p.add_argument("--hidden_dim", type=int, default=512,
                   help="Hidden dim (only used for 'original' arch_type)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate (only used for 'original' arch_type)")
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
                   choices=["none", "linear", "sqrt"],
                   help="Per-GO-term pos weighting for BCEWithLogitsLoss")
    p.add_argument("--pos_weight_cap", type=float, default=50.0)
    p.add_argument("--eval_splits", type=str, nargs="+",
                   default=["train", "val", "test"],
                   choices=["train", "val", "test"])

    p.add_argument("--eval_only", action="store_true")

    # ---- Output ----
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--log_name", type=str, default="ft_cafa")

    # ---- Misc ----
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=50)

    return p.parse_args()


# ============================================================
# End-to-End Model
# ============================================================
class EndToEndCAFAModel(nn.Module):
    """PLM backbone + GOTermPredictor head."""

    def __init__(self, model_ctx, head: GOTermPredictor):
        super().__init__()
        self.model_ctx = model_ctx
        self.backbone = model_ctx.model
        self.head = head

    def forward(self, seqs):
        emb = finetune_forward(self.model_ctx, seqs)
        return self.head(emb)


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
# Loss helpers (single/three)
# ============================================================
def _build_criterion(args, pos_weight, device):
    """Return (criterion_single, criteria_three_dict_or_None)."""
    if args.head_mode == "single":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw), None
    if pos_weight is not None:
        criteria = {
            ns: nn.BCEWithLogitsLoss(pos_weight=pos_weight[ns].to(device))
            for ns in ("BPO", "CCO", "MFO")
        }
    else:
        criteria = {ns: nn.BCEWithLogitsLoss() for ns in ("BPO", "CCO", "MFO")}
    return None, criteria


def _compute_loss(args, output, labels, device, criterion, criteria):
    if args.head_mode == "single":
        labels = labels.to(device, non_blocking=True)
        return criterion(output, labels)
    return sum(
        criteria[ns](output[ns], labels[ns].to(device, non_blocking=True))
        for ns in ("BPO", "CCO", "MFO")
    ) / 3.0


# ============================================================
# Training
# ============================================================
def train_model(
    e2e_model: EndToEndCAFAModel,
    train_dataset: CAFASequenceDataset,
    val_dataset: CAFASequenceDataset,
    args,
    device: torch.device,
    go_vocab,
    collate_fn,
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
            f"  Head: {args.head_mode}  Arch: {args.arch_type}  "
            f"Pos-weight: {args.pos_weight}  AMP: {args.use_amp}  "
            f"Early-stop: {'off' if es_patience == 0 else es_patience}"
        )
        logging.info("=" * 70)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate_fn, num_workers=nw,
        pin_memory=use_pin, drop_last=use_ddp,
        worker_init_fn=seed_worker, generator=make_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=collate_fn,
        num_workers=nw, pin_memory=use_pin, drop_last=False,
        worker_init_fn=seed_worker,
    )

    total_steps = len(train_loader)
    criterion, criteria = _build_criterion(args, pos_weight, device)

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

        for step, (seqs, labels, _) in enumerate(train_loader, 1):
            with autocast_ctx:
                output = e2e_model(seqs)
                loss = _compute_loss(args, output, labels, device, criterion, criteria) / accum_steps

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
        with torch.no_grad():
            for seqs, labels, _ in val_loader:
                output = e2e_model(seqs)
                epoch_val += _compute_loss(
                    args, output, labels, device, criterion, criteria
                ).item()

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

    if is_main_process() and best_state is not None:
        ckpt_path = join(args.save_dir, f"{args.log_name}_best_checkpoint.pt")
        torch.save({
            "backbone_state_dict": best_state["backbone_state_dict"],
            "head_state_dict": best_state["head_state_dict"],
            "model_name": args.model_name,
            "head_mode": args.head_mode,
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
    e2e_model: EndToEndCAFAModel,
    dataset: CAFASequenceDataset,
    split_name: str,
    go_vocab,
    args,
    device: torch.device,
    collate_fn,
):
    logging.info(f"--- Evaluating {split_name} ({len(dataset)} proteins) ---")
    raw = e2e_model.module if hasattr(e2e_model, "module") else e2e_model
    raw.eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    all_preds, all_labels, all_pids = [], [], []
    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (seqs, labels, pids) in enumerate(loader, 1):
            if batch_idx % 200 == 0 or batch_idx == n_batches:
                logging.info(f"  Inference: batch {batch_idx}/{n_batches}")
            output = raw(seqs)

            if args.head_mode == "single":
                probs = torch.sigmoid(output).cpu().numpy()
                labels_np = labels.numpy()
            else:
                probs = torch.cat(
                    [torch.sigmoid(output[ns]) for ns in ("BPO", "CCO", "MFO")],
                    dim=1,
                ).cpu().numpy()
                labels_np = torch.cat(
                    [labels[ns] for ns in ("BPO", "CCO", "MFO")], dim=1,
                ).numpy()

            all_preds.append(probs)
            all_labels.append(labels_np)
            all_pids.extend(pids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = {}
    fmax_overall, best_t = compute_fmax(all_labels, all_preds)
    metrics["overall_fmax"] = fmax_overall
    metrics["overall_threshold"] = best_t
    logging.info(f"  Overall Fmax: {fmax_overall:.4f} (threshold: {best_t:.2f})")

    ns_sizes = {ns: len(go_vocab[ns]) for ns in ("BPO", "CCO", "MFO")}
    offset = 0
    for ns in ("BPO", "CCO", "MFO"):
        ns_labels = all_labels[:, offset:offset + ns_sizes[ns]]
        ns_preds = all_preds[:, offset:offset + ns_sizes[ns]]
        has_annotation = ns_labels.sum(axis=1) > 0
        if has_annotation.sum() > 0:
            ns_fmax, ns_t = compute_fmax(
                ns_labels[has_annotation], ns_preds[has_annotation],
            )
            metrics[f"{ns}_fmax"] = ns_fmax
            metrics[f"{ns}_threshold"] = ns_t
            metrics[f"{ns}_proteins"] = int(has_annotation.sum())
            logging.info(
                f"  {ns} Fmax: {ns_fmax:.4f} (t={ns_t:.2f}, n={has_annotation.sum()})"
            )
        else:
            metrics[f"{ns}_fmax"] = None
            metrics[f"{ns}_threshold"] = None
            metrics[f"{ns}_proteins"] = 0
            logging.info(f"  {ns} Fmax: N/A (no proteins)")
        offset += ns_sizes[ns]

    metrics["num_proteins"] = len(all_pids)
    metrics["label_density"] = float(all_labels.sum() / max(all_labels.size, 1))
    metrics["avg_active_terms"] = float(all_labels.sum(axis=1).mean())

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
        logging.info("CAFA5 GO Term Fine-Tuning Pipeline")
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Model: {args.model_name} (family={spec.family})")
        logging.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            logging.info(f"  {k}: {v}")

    splits = load_cafa_splits(args.splits_dir)
    all_protein_ids = collect_cafa_protein_ids(splits)
    if is_main_process():
        logging.info(f"  Unique proteins across splits: {len(all_protein_ids)}")

    if is_main_process():
        logging.info("Building GO vocabulary...")
    go_vocab = build_go_vocabulary(args.splits_dir)
    ns_to_idx, flat_to_idx = build_go_to_index(go_vocab)
    total_terms = sum(len(v) for v in go_vocab.values())
    if is_main_process():
        logging.info(
            f"  BPO: {len(go_vocab['BPO'])}  CCO: {len(go_vocab['CCO'])}  "
            f"MFO: {len(go_vocab['MFO'])}  Total: {total_terms}"
        )
        vocab_path = join(args.save_dir, f"{args.log_name}_go_vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(go_vocab, f, indent=2)
        logging.info(f"  GO vocab saved: {vocab_path}")

    if is_main_process():
        logging.info(f"  Loading PLM backbone: {args.model_name}")
    model_ctx = load_model(args.model_name, args.model_path, device)
    embedding_dim = int(model_ctx.extras.get("embedding_dim", spec.embedding_dim))
    args.embedding_dim = embedding_dim
    if is_main_process():
        logging.info(f"  Backbone embedding dim: {embedding_dim}")

    if args.freeze_layers > 0:
        freeze_backbone_layers(model_ctx, args.freeze_layers)

    head = GOTermPredictor(
        embedding_dim=embedding_dim,
        go_vocab=go_vocab,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mode=args.head_mode,
        arch_type=args.arch_type,
    )
    e2e_model = EndToEndCAFAModel(model_ctx, head).to(device)

    total_params = sum(p.numel() for p in e2e_model.parameters())
    trainable_params = sum(p.numel() for p in e2e_model.parameters() if p.requires_grad)
    if is_main_process():
        logging.info(
            f"  Total params: {total_params:,}  Trainable: {trainable_params:,}"
        )
        if args.arch_type == "original":
            logging.info(
                f"  Architecture: {embedding_dim} → {args.hidden_dim} → output"
            )
        else:
            cfg = ARCH_CONFIGS[args.arch_type]
            dims = [str(embedding_dim)] + [str(c["out"]) for c in cfg] + ["output"]
            logging.info(f"  Architecture: {' → '.join(dims)}")

    collate_fn = (
        cafa_sequence_collate_three if args.head_mode == "three"
        else cafa_sequence_collate_single
    )

    datasets = {}
    for name in ("train", "val", "test"):
        if name in splits:
            tsv_path = join(args.splits_dir, f"{name}.tsv")
            datasets[name] = CAFASequenceDataset(
                split_tsv=tsv_path,
                go_vocab=go_vocab,
                ns_to_idx=ns_to_idx,
                flat_to_idx=flat_to_idx,
                label_mode=args.head_mode,
                split_name=name,
            )

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
                _, _, _, metrics = evaluate_split(
                    e2e_model, datasets[sn], sn, go_vocab, args, device,
                    collate_fn=collate_fn,
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
        pw = compute_cafa_pos_weight(
            datasets["train"], go_vocab,
            mode=args.head_mode,
            method=args.pos_weight,
            cap=args.pos_weight_cap,
        )

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
        sweep_criterion, sweep_criteria = _build_criterion(args, pw, device)

        def _sweep_forward(model, batch, _criterion_unused, dev):
            seqs, labels, _ = batch
            output = model(seqs)
            return _compute_loss(
                args, output, labels, dev, sweep_criterion, sweep_criteria,
            )

        # The lr_selector signature requires *some* criterion; pass a sentinel.
        sentinel_criterion = sweep_criterion if sweep_criterion is not None else nn.Identity()

        best_lr = run_lr_dynsweep_finetune(
            args, e2e_model,
            sweep_train_loader, sweep_val_loader,
            _sweep_forward, sentinel_criterion, device,
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
        go_vocab=go_vocab, collate_fn=collate_fn,
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
            logging.info("")
            _, _, _, metrics = evaluate_split(
                raw, datasets[sn], sn, go_vocab, args, device,
                collate_fn=collate_fn,
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
        logging.info(
            f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)"
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
