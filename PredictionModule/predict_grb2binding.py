#!/usr/bin/env python3
"""
================================================================
GRB2 Binding Score Prediction Pipeline
================================================================

Predicts continuous GRB2 SH2 domain peptide binding scores from
a Deep Mutational Scanning (DMS) dataset using protein embeddings.

Two operating modes for protein embeddings:
  1) Pre-extracted embeddings  – supply --emb_dir + --emb_suffix + --embedding_dim
  2) On-the-fly extraction     – supply --model_name + --model_path
     (optionally --save_embeddings_dir to persist extracted embeddings)

In both modes the script:
  • builds a GRB2Regressor prediction head (linear / mlp)
  • trains from scratch (or loads --checkpoint) with MSE or Huber loss
  • evaluates on train/val/test
  • saves predictions TSV, metrics JSON, training curves, and checkpoint

Primary metric: Spearman ρ.
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
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from util_model import MODEL_REGISTRY, extract_embeddings, load_model
from util_data import (
    GRB2Dataset,
    collect_grb2_protein_ids,
    collect_grb2_sequences,
    grb2_collate,
    load_grb2_splits,
    preload_embeddings,
    preload_foldvision_embeddings,
    save_embeddings,
)
from util_helper import (
    apply_sweep_lr,
    auto_setup_ddp,
    cleanup_ddp,
    compute_regression_metrics,
    get_device,
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
        description="GRB2 Binding Score Regression — pre-extracted or on-the-fly embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory with train.tsv, val.tsv, test.tsv "
                        "(columns: variant, num_mutations, score, sequence)")

    # ---- Mode 1: pre-extracted embeddings ----
    p.add_argument("--emb_dir", type=str, default="",
                   help="Directory of pre-extracted .pt embeddings (Mode 1)")
    p.add_argument("--emb_suffix", type=str, default="_per_tok.pt",
                   help="Filename suffix for embedding files")
    p.add_argument("--embedding_dim", type=int, default=0,
                   help="Embedding dimension (auto-detected in Mode 2)")
    p.add_argument("--emb_type", type=str, default="per_token",
                   choices=["per_token", "mean"],
                   help="Type of stored embeddings")
    p.add_argument("--foldvision_runs", type=int, default=0,
                   help="If > 0, loads multiple .npz runs for Test-Time Augmentation.")

    # ---- Mode 2: on-the-fly extraction ----
    p.add_argument("--model_name", type=str, default="",
                   choices=[""] + sorted(MODEL_REGISTRY.keys()),
                   help="PLM model name from registry (Mode 2)")
    p.add_argument("--model_path", type=str, default="",
                   help="Path / HuggingFace ID for PLM weights (Mode 2)")
    p.add_argument("--pdb_dir", type=str, default="",
                   help="Directory with .pdb files (required for structure models, e.g. ESM-IF)")
    p.add_argument("--extraction_batch_size", type=int, default=8,
                   help="Batch size for on-the-fly embedding extraction")
    p.add_argument("--save_embeddings_dir", type=str, default="",
                   help="If set, save extracted embeddings here for later reuse")

    # ---- Prediction head ----
    p.add_argument("--arch_type", type=str, default="mlp",
                   choices=["linear", "mlp"],
                   help="'mlp' = 2-layer MLP (Linear → LN → GELU → Dropout → Linear); "
                        "'linear' = direct linear probe")
    p.add_argument("--hidden_dim", type=int, default=512,
                   help="Hidden dimension (only used for 'mlp' arch_type)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate (only used for 'mlp' arch_type)")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to trained checkpoint (.pt). Empty → train from scratch")

    # ---- Training ----
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss_type", type=str, default="mse",
                   choices=["mse", "huber"],
                   help="Regression loss: 'mse' (standard) or 'huber' (robust to outliers)")
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
    p.add_argument("--log_name", type=str, default="grb2_pred")

    # ---- Misc ----
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help="0 = disabled; stops when val loss does not improve")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--log_every_n_steps", type=int, default=50,
                   help="Step-level logging frequency (0 = disabled)")

    return p.parse_args()


# ============================================================
# Prediction Head
# ============================================================
class GRB2Regressor(nn.Module):
    """
    Lightweight prediction head for GRB2 binding / fitness regression.

    'linear': embedding_dim → 1
        Direct linear projection — linear probe.  All predictive signal
        comes from the PLM representation, enabling fair cross-model
        comparison.  Default and recommended for benchmarking.

    'mlp':    embedding_dim → hidden_dim → 1
        Two-layer MLP with LayerNorm, GELU activation, and Dropout.
        Matches the unified 2-layer regression head format.

    No output activation — raw logits are returned and passed to
    MSELoss / HuberLoss directly during training.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        arch_type: str = "linear",
    ):
        super().__init__()
        if arch_type == "linear":
            self.head = nn.Linear(embedding_dim, 1)
        elif arch_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown arch_type: {arch_type!r}. Choose 'linear' or 'mlp'.")

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: (batch, embedding_dim)  mean-pooled protein embeddings
        Returns:
            scores: (batch,)  predicted binding / fitness scores
        """
        return self.head(emb).squeeze(-1)


# ============================================================
# Training
# ============================================================
def train_model(model, train_dataset, val_dataset, args, device, use_ddp=False):
    """Train GRB2Regressor from scratch; return (best_model, train_losses, val_losses)."""
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
        logging.info(f"  Arch: {args.arch_type}  Loss: {args.loss_type}  "
                     f"Early-stop: {'off' if es_patience == 0 else es_patience}")
        logging.info("=" * 70)

    model = model.to(device)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=grb2_collate, num_workers=nw, pin_memory=use_pin,
        persistent_workers=(nw > 0), drop_last=use_ddp,
        worker_init_fn=seed_worker, generator=make_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=grb2_collate,
        num_workers=nw, pin_memory=use_pin,
        persistent_workers=(nw > 0), drop_last=False,
        worker_init_fn=seed_worker,
    )

    total_steps = len(train_loader)

    criterion = nn.HuberLoss() if args.loss_type == "huber" else nn.MSELoss()

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

        for step, (emb, tgt, _) in enumerate(train_loader, 1):
            emb = emb.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(model(emb), tgt)
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
            for emb, tgt, _ in val_loader:
                emb = emb.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                if emb.ndim == 3:
                    B, N, D = emb.shape
                    emb = emb.view(B * N, D)
                    preds = model(emb).view(B, N).mean(dim=1)
                else:
                    preds = model(emb)

                epoch_val += criterion(preds, tgt).item()
                vp.append(preds.cpu().numpy())
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
            from scipy.stats import spearmanr
            vp_np = np.concatenate(vp)
            vt_np = np.concatenate(vt)
            spear_r, _ = spearmanr(vt_np, vp_np)
            vt_mean = float(np.mean(vt_np))
            ss_res = float(np.sum((vt_np - vp_np) ** 2))
            ss_tot = float(np.sum((vt_np - vt_mean) ** 2))
            r2 = float("nan") if ss_tot <= 0.0 else float(1.0 - (ss_res / ss_tot))
            elapsed = time.time() - t0
            logging.info(
                f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
                f"Train {avg_train:.6f} | Val {avg_val:.6f} | "
                f"Val Spearman ρ {spear_r:.4f} | Val R² {r2:.4f} | "
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
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "loss_type": args.loss_type,
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
def evaluate_split(model, dataset, split_name, args, device):
    """Evaluate on one split.  Returns (DataFrame, metrics_dict)."""
    import pandas as pd

    logging.info(f"--- Evaluating {split_name} ({len(dataset)} variants) ---")
    raw = model.module if hasattr(model, "module") else model
    raw = raw.to(device).eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=grb2_collate, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    all_preds, all_labels, all_variants = [], [], []
    with torch.no_grad():
        for emb, tgt, variants in loader:
            emb = emb.to(device, non_blocking=True)

            if emb.ndim == 3:
                # FoldVision Test-Time Augmentation: average raw scores across N runs
                B, N, D = emb.shape
                emb = emb.view(B * N, D)
                preds = raw(emb).view(B, N).mean(dim=1).cpu().numpy()
            else:
                preds = raw(emb).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(tgt.numpy())
            all_variants.extend(variants)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = compute_regression_metrics(all_labels, all_preds, n_predictors=args.embedding_dim)
    logging.info(
        f"  Spearman ρ={metrics['spearman_r']:.4f}  "
        f"R²={metrics['r2']:.4f}  Adj R²={metrics['adjusted_r2']:.4f}  "
        f"Pearson r={metrics['pearson_r']:.4f}  "
        f"MSE={metrics['mse']:.6f}  RMSE={metrics['rmse']:.6f}  "
        f"MAE={metrics['mae']:.6f}  N={metrics['n_samples']}"
    )

    # Predictions TSV
    df = pd.DataFrame({
        "variant": all_variants,
        "target_score": np.round(all_labels, 6),
        "predicted_score": np.round(all_preds, 6),
        "residual": np.round(all_labels - all_preds, 6),
    })
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

    # ---- Determine mode ----
    mode_preextracted = bool(args.emb_dir)
    mode_onthefly = bool(args.model_name)

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
        logging.info("GRB2 Binding Score Prediction Pipeline")
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Mode: {'pre-extracted' if mode_preextracted else 'on-the-fly'}")
        logging.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            if k == "lr":
                continue
            logging.info(f"  {k}: {v}")

    device = get_device(use_ddp)

    # ---- Load splits ----
    splits = load_grb2_splits(args.splits_dir)
    if not splits:
        logging.error(f"No split files found in: {args.splits_dir}")
        cleanup_ddp()
        sys.exit(1)

    all_protein_ids = collect_grb2_protein_ids(splits)
    if is_main_process():
        for sn, df in splits.items():
            logging.info(f"  {sn}: {len(df)} variants")
        logging.info(f"  Unique variants across splits: {len(all_protein_ids)}")

    # ============================================================
    # Obtain embeddings
    # ============================================================
    if mode_preextracted:
        if args.embedding_dim <= 0:
            logging.error("--embedding_dim is required in pre-extracted mode")
            cleanup_ddp()
            sys.exit(1)

        if is_main_process():
            logging.info(f"  Loading embeddings from: {args.emb_dir}")
        if args.foldvision_runs > 0:
            embeddings, missing = preload_foldvision_embeddings(
                all_protein_ids, args.emb_dir, args.foldvision_runs
            )
        else:
            embeddings, missing = preload_embeddings(
                all_protein_ids, args.emb_dir, args.emb_suffix,
                args.embedding_dim, args.emb_type,
            )
        embedding_dim = args.embedding_dim

    else:
        # --- Mode 2: on-the-fly extraction ---
        spec = MODEL_REGISTRY[args.model_name]
        embedding_dim = spec.embedding_dim
        args.embedding_dim = embedding_dim

        sequences = collect_grb2_sequences(splits)
        if is_main_process():
            logging.info(f"  Sequences collected: {len(sequences)}")

        model_ctx = load_model(args.model_name, args.model_path, device)
        runtime_embedding_dim = int(model_ctx.extras.get("embedding_dim", spec.embedding_dim))
        if runtime_embedding_dim != embedding_dim and is_main_process():
            logging.info(
                f"  Embedding dim override from model config: "
                f"{embedding_dim} -> {runtime_embedding_dim}"
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        missing = [p for p in all_protein_ids if p not in embeddings]

    if is_main_process():
        logging.info(f"  Embeddings available: {len(embeddings)}/{len(all_protein_ids)}")
        if missing:
            logging.warning(f"  Missing embeddings: {len(missing)}")

    # ============================================================
    # Build datasets
    # ============================================================
    datasets = {}
    for name in ("train", "val", "test"):
        if name in splits:
            tsv_path = join(args.splits_dir, f"{name}.tsv")
            datasets[name] = GRB2Dataset(tsv_path, embeddings, split_name=name)

    if is_main_process():
        for sn, ds in datasets.items():
            logging.info(f"  Dataset {sn}: {len(ds)} samples")

    # ============================================================
    # Prediction head model
    # ============================================================
    model = GRB2Regressor(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        arch_type=args.arch_type,
    )
    total_p = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logging.info(f"  GRB2Regressor: arch={args.arch_type}, params={total_p:,}")

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

        # ---- Dynamic LR Sweep (Rule 4a / 4b selector — sweep-only mode) ----
        if args.lr_dynsweep:
            sweep_train_loader = DataLoader(
                datasets["train"], batch_size=args.batch_size,
                shuffle=True, collate_fn=grb2_collate,
                num_workers=0, pin_memory=False, drop_last=False,
                generator=make_loader_generator(args.seed),
            )
            sweep_val_loader = DataLoader(
                datasets["val"], batch_size=args.batch_size,
                shuffle=False, collate_fn=grb2_collate,
                num_workers=0, pin_memory=False, drop_last=False,
            )
            sweep_criterion = (
                nn.HuberLoss() if args.loss_type == "huber" else nn.MSELoss()
            )

            _emb_dim_ref = embedding_dim

            def _sweep_model_factory():
                return GRB2Regressor(
                    embedding_dim=_emb_dim_ref,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    arch_type=args.arch_type,
                )

            def _sweep_forward(model, batch, criterion, dev):
                emb, tgt, _ = batch
                emb = emb.to(dev, non_blocking=True)
                tgt = tgt.to(dev, non_blocking=True)
                if emb.ndim == 3:
                    B, N, D = emb.shape
                    emb = emb.view(B * N, D)
                    preds = model(emb).view(B, N).mean(dim=1)
                    return criterion(preds, tgt)
                else:
                    return criterion(model(emb), tgt)

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
            model, datasets["train"], datasets["val"], args, device, use_ddp=use_ddp,
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
            _, metrics = evaluate_split(raw, datasets[sn], sn, args, device)
            all_metrics[sn] = metrics

        if val_losses is not None:
            all_metrics["best_epoch"] = int(np.argmin(val_losses)) + 1
            all_metrics["total_epochs"] = len(val_losses)

        save_metrics_json(
            all_metrics, vars(args),
            join(args.save_dir, f"{args.log_name}_metrics.json"),
        )

        # Summary table
        logging.info("")
        logging.info(
            f"{'Split':<8} {'Spearman ρ':>12} {'R²':>8} {'Adj R²':>8} {'Pearson r':>10} "
            f"{'MSE':>10} {'RMSE':>10} {'MAE':>10} {'N':>8}"
        )
        logging.info("-" * 92)
        for sn, m in all_metrics.items():
            if not isinstance(m, dict):
                continue
            logging.info(
                f"{sn:<8} {m['spearman_r']:>12.4f} {m['r2']:>8.4f} {m['adjusted_r2']:>8.4f} {m['pearson_r']:>10.4f} "
                f"{m['mse']:>10.6f} {m['rmse']:>10.6f} {m['mae']:>10.6f} "
                f"{m['n_samples']:>8}"
            )
        logging.info("=" * 70)
        logging.info(f"All outputs in: {args.save_dir}")
        logging.info("Done.")

    if is_main_process():
        total_elapsed = time.time() - main_t0
        logging.info(f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    cleanup_ddp()


if __name__ == "__main__":
    main()
