"""
================================================================
Fine-Tuning Utilities for Protein Language Models
================================================================

Sequence-based dataset classes and collate functions for end-to-end
PLM fine-tuning, plus shared infrastructure: differential-LR
parameter groups, family-aware layer freezing, AMP helpers, and
a glue wrapper around lr_selector for the dynamic LR sweep.

Components:
  Datasets / collates (return raw sequences, not pre-computed embeddings):
    - PPISequenceDataset / ppi_sequence_collate
    - DAVISSequenceDataset / davis_sequence_collate
    - GRB2SequenceDataset / grb2_sequence_collate
    - CAFASequenceDataset / cafa_sequence_collate_single|three
    - SPOTSequenceDataset / spot_sequence_collate

  Shared:
    - build_param_groups        — differential LR for backbone vs head
    - freeze_backbone_layers    — family-aware layer freezing
    - get_amp_context           — mixed-precision context manager (CUDA / fallback)
    - make_finetune_factory     — snapshot init weights + factory for lr_selector
    - run_lr_dynsweep_finetune  — single-LR dynamic sweep (Rule 4a / 4b)
================================================================
"""

import ast
import logging
from os.path import join
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ============================================================
# PPI Sequence Dataset (PRING)
# ============================================================
class PPISequenceDataset(Dataset):
    """
    Dataset for PRING PPI fine-tuning that returns raw amino-acid sequences.

    Expects a TSV with columns: protein_a, protein_b, ..., output
    """

    def __init__(
        self,
        split_tsv: str,
        sequences: Dict[str, str],
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")

        self.pairs: List[Tuple[str, str, int]] = []
        skipped = 0
        for _, row in df.iterrows():
            pid_a = str(row["protein_a"])
            pid_b = str(row["protein_b"])
            if pid_a in sequences and pid_b in sequences:
                self.pairs.append((pid_a, pid_b, int(row["output"])))
            else:
                skipped += 1

        self.sequences = sequences

        unique = set()
        for a, b, _ in self.pairs:
            unique.add(a)
            unique.add(b)

        logger.info(
            f"  {split_name}: {len(self.pairs)}/{len(df)} pairs loaded "
            f"({skipped} skipped), {len(unique)} unique proteins"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid_a, pid_b, target = self.pairs[idx]
        return (
            self.sequences[pid_a],
            self.sequences[pid_b],
            target,
            pid_a,
            pid_b,
        )

    def get_protein_ids(self):
        """Return (pids_a, pids_b, targets) lists."""
        pids_a = [p[0] for p in self.pairs]
        pids_b = [p[1] for p in self.pairs]
        targets = [p[2] for p in self.pairs]
        return pids_a, pids_b, targets


def ppi_sequence_collate(batch):
    """
    Returns (seqs_a, seqs_b, targets, pids_a, pids_b).
    """
    seqs_a = [b[0] for b in batch]
    seqs_b = [b[1] for b in batch]
    targets = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    pids_a = [b[3] for b in batch]
    pids_b = [b[4] for b in batch]
    return seqs_a, seqs_b, targets, pids_a, pids_b


# ============================================================
# DAVIS Sequence Dataset (drug–target affinity regression)
# ============================================================
class DAVISSequenceDataset(Dataset):
    """
    Dataset for DAVIS fine-tuning. The PLM-side input is a raw protein
    sequence; the molecule side is a pre-computed (frozen) embedding
    keyed by SMILES.

    Expects TSV columns: protein_id, sequence, SMILES, output
    """

    def __init__(
        self,
        split_tsv: str,
        sequences: Dict[str, str],
        mol_embeddings: Dict[str, torch.Tensor],
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")
        self.pairs: List[Tuple[str, str, float]] = []
        skipped_prot = 0
        skipped_mol = 0

        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            smi = str(row["SMILES"])
            if pid not in sequences:
                skipped_prot += 1
                continue
            if smi not in mol_embeddings:
                skipped_mol += 1
                continue
            self.pairs.append((pid, smi, float(row["output"])))

        self.sequences = sequences
        self.mol_embeddings = mol_embeddings

        unique_prots = set(p[0] for p in self.pairs)
        unique_mols = set(p[1] for p in self.pairs)
        logger.info(
            f"  {split_name}: {len(self.pairs)}/{len(df)} pairs loaded "
            f"(skipped {skipped_prot} missing prot, {skipped_mol} missing mol), "
            f"{len(unique_prots)} proteins, {len(unique_mols)} drugs"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid, smi, score = self.pairs[idx]
        return (
            self.sequences[pid],
            self.mol_embeddings[smi],
            torch.tensor(score, dtype=torch.float32),
            pid,
            smi,
        )

    def get_ids(self):
        pids = [p[0] for p in self.pairs]
        smiles = [p[1] for p in self.pairs]
        scores = [p[2] for p in self.pairs]
        return pids, smiles, scores


def davis_sequence_collate(batch):
    """Returns (seqs, mol_embs, targets, pids, smiles)."""
    seqs = [b[0] for b in batch]
    mol_embs = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    pids = [b[3] for b in batch]
    smiles = [b[4] for b in batch]
    return seqs, mol_embs, targets, pids, smiles


# ============================================================
# GRB2 Sequence Dataset (single-protein regression)
# ============================================================
class GRB2SequenceDataset(Dataset):
    """
    Dataset for GRB2 binding/abundance fine-tuning. Each row is one
    variant (e.g. "D28Y,N49Y") and a continuous score.

    Expects TSV columns: variant, sequence, output
    """

    def __init__(
        self,
        split_tsv: str,
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")
        self.entries: List[Tuple[str, str, float]] = []
        for _, row in df.iterrows():
            vid = str(row["variant"])
            seq = str(row["sequence"])
            self.entries.append((vid, seq, float(row["output"])))

        logger.info(
            f"  {split_name}: {len(self.entries)} variants loaded"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        vid, seq, score = self.entries[idx]
        return seq, torch.tensor(score, dtype=torch.float32), vid

    def get_variant_ids(self) -> List[str]:
        return [e[0] for e in self.entries]


def grb2_sequence_collate(batch):
    """Returns (seqs, scores, variants)."""
    seqs = [b[0] for b in batch]
    scores = torch.stack([b[1] for b in batch])
    variants = [b[2] for b in batch]
    return seqs, scores, variants


# ============================================================
# CAFA Sequence Dataset (multi-label GO term classification)
# ============================================================
class CAFASequenceDataset(Dataset):
    """
    Dataset for CAFA5 fine-tuning. Returns raw protein sequences and
    encoded GO label vector(s).

    Expects TSV columns: protein_id, sequence, output
    The "output" column is a stringified dict like
    "{'BPO': [...], 'CCO': [...], 'MFO': [...]}".
    """

    def __init__(
        self,
        split_tsv: str,
        go_vocab: Dict[str, List[str]],
        ns_to_idx: Dict[str, Dict[str, int]],
        flat_to_idx: Dict[str, int],
        label_mode: str = "single",
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")
        self.protein_ids: List[str] = []
        self.sequences_list: List[str] = []
        self.targets: List[dict] = []
        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            seq = str(row["sequence"])
            tgt = ast.literal_eval(row["output"])
            self.protein_ids.append(pid)
            self.sequences_list.append(seq)
            self.targets.append(tgt)

        self.go_vocab = go_vocab
        self.ns_to_idx = ns_to_idx
        self.flat_to_idx = flat_to_idx
        self.label_mode = label_mode

        logger.info(
            f"  {split_name}: {len(self.protein_ids)}/{len(df)} proteins loaded, "
            f"label_mode={label_mode}"
        )

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        seq = self.sequences_list[idx]
        tgt_dict = self.targets[idx]

        if self.label_mode == "single":
            total_dim = sum(len(self.go_vocab[ns]) for ns in ("BPO", "CCO", "MFO"))
            label = np.zeros(total_dim, dtype=np.float32)
            for ns in ("BPO", "CCO", "MFO"):
                if ns in tgt_dict:
                    for term in tgt_dict[ns]:
                        if term in self.flat_to_idx:
                            label[self.flat_to_idx[term]] = 1.0
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = {}
            for ns in ("BPO", "CCO", "MFO"):
                vec = np.zeros(len(self.go_vocab[ns]), dtype=np.float32)
                if ns in tgt_dict:
                    for term in tgt_dict[ns]:
                        if term in self.ns_to_idx[ns]:
                            vec[self.ns_to_idx[ns][term]] = 1.0
                label[ns] = torch.tensor(vec, dtype=torch.float32)

        return seq, label, pid


def cafa_sequence_collate_single(batch):
    """Returns (seqs, labels_tensor, pids)."""
    seqs = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    pids = [b[2] for b in batch]
    return seqs, labels, pids


def cafa_sequence_collate_three(batch):
    """Returns (seqs, {ns: tensor}, pids)."""
    seqs = [b[0] for b in batch]
    pids = [b[2] for b in batch]
    labels = {
        ns: torch.stack([b[1][ns] for b in batch])
        for ns in ("BPO", "CCO", "MFO")
    }
    return seqs, labels, pids


# ============================================================
# SPOT Sequence Dataset (binary protein–molecule interaction)
# ============================================================
class SPOTSequenceDataset(Dataset):
    """
    Dataset for SPOT fine-tuning (TSV format, mirrors DAVIS).

    PLM-side input: raw protein sequence.
    Molecule-side input: pre-computed (frozen) embedding keyed by SMILES.

    Expects TSV columns: protein_id, sequence, SMILES, output
    """

    def __init__(
        self,
        split_tsv: str,
        sequences: Dict[str, str],
        mol_embeddings: Dict[str, torch.Tensor],
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")
        self.pairs: List[Tuple[str, str, int]] = []
        skipped_prot = 0
        skipped_mol = 0

        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            smi = str(row["SMILES"])
            if pid not in sequences:
                skipped_prot += 1
                continue
            if smi not in mol_embeddings:
                skipped_mol += 1
                continue
            self.pairs.append((pid, smi, int(row["output"])))

        self.sequences = sequences
        self.mol_embeddings = mol_embeddings

        unique_prots = set(p[0] for p in self.pairs)
        unique_mols = set(p[1] for p in self.pairs)
        logger.info(
            f"  {split_name}: {len(self.pairs)}/{len(df)} pairs loaded "
            f"(skipped {skipped_prot} missing prot, {skipped_mol} missing mol), "
            f"{len(unique_prots)} proteins, {len(unique_mols)} molecules"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid, smi, target = self.pairs[idx]
        return (
            self.sequences[pid],
            self.mol_embeddings[smi],
            torch.tensor(target, dtype=torch.float32),
            pid,
            smi,
        )

    def get_ids(self):
        pids = [p[0] for p in self.pairs]
        smiles = [p[1] for p in self.pairs]
        targets = [p[2] for p in self.pairs]
        return pids, smiles, targets


def spot_sequence_collate(batch):
    """Returns (seqs, mol_embs, targets, pids, smiles)."""
    seqs = [b[0] for b in batch]
    mol_embs = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    pids = [b[3] for b in batch]
    smiles = [b[4] for b in batch]
    return seqs, mol_embs, targets, pids, smiles


# ============================================================
# Differential Learning Rate — Parameter Groups
# ============================================================
def build_param_groups(
    backbone: nn.Module,
    head: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> List[dict]:
    """
    Build optimizer parameter groups with differential learning rates.

    The backbone (PLM) gets a lower LR to preserve pre-trained knowledge,
    while the prediction head gets a higher LR for faster convergence.
    """
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]

    param_groups = [
        {
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head",
        },
    ]

    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    logger.info(
        f"  Param groups: backbone={n_bb:,} params (lr={backbone_lr}), "
        f"head={n_hd:,} params (lr={head_lr})"
    )

    return param_groups


# ============================================================
# Layer Freezing (family-aware)
# ============================================================
_LAYER_PATTERNS = {
    "esm1b": "model.layers",
    "esm2": "model.layers",
    "prostt5": "model.encoder.block",
    "prott5": "model.encoder.block",
    "ankh": "model.encoder.block",
    "protbert": "model.encoder.layer",
    "venusplm": "model.layers",
    "xtrimopglm": "model.transformer.layers",
    "carp": "model.layers",
}


def _get_layer_modules(model: nn.Module, dot_path: str) -> Optional[nn.ModuleList]:
    """Navigate model attributes by dot-separated path."""
    obj = model
    for attr in dot_path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    if isinstance(obj, (nn.ModuleList, list)):
        return obj
    return None


def freeze_backbone_layers(model_ctx, n_freeze: int) -> int:
    """
    Freeze the first *n_freeze* transformer layers of the PLM backbone.

    Embeddings are also frozen when n_freeze > 0. Returns the number of
    layers actually frozen.
    """
    if n_freeze <= 0:
        return 0

    family = model_ctx.spec.family
    model = model_ctx.model

    layer_path = _LAYER_PATTERNS.get(family)
    if layer_path is None:
        logger.warning(
            f"  No layer pattern for family '{family}' — skipping layer freezing"
        )
        return 0

    layers = _get_layer_modules(model, layer_path)
    if layers is None:
        logger.warning(
            f"  Could not access '{layer_path}' on {family} model — "
            f"skipping layer freezing"
        )
        return 0

    total_layers = len(layers)
    actual_freeze = min(n_freeze, total_layers)

    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ("embed", "wte", "wpe", "position")):
            param.requires_grad = False

    for i in range(actual_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False

    frozen_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(
        f"  Froze {actual_freeze}/{total_layers} layers in {family} backbone "
        f"({frozen_params:,}/{total_params:,} params frozen, "
        f"{total_params - frozen_params:,} trainable)"
    )

    return actual_freeze


# ============================================================
# Mixed Precision (AMP) Context Helper
# ============================================================
def get_amp_context(device: torch.device):
    """
    Return (autocast_ctx, grad_scaler).

    - CUDA: torch.amp.autocast('cuda') + GradScaler
    - MPS / CPU: nullcontext + None (AMP not supported)
    """
    from contextlib import nullcontext

    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
        logger.info("  AMP enabled: float16 autocast + GradScaler (CUDA)")
        return autocast_ctx, scaler
    logger.info(f"  AMP disabled on {device.type} — using float32")
    return nullcontext(), None


# ============================================================
# lr_selector glue for fine-tuning
# ============================================================
def make_finetune_factory(e2e_model: nn.Module) -> Callable[[], nn.Module]:
    """
    Snapshot the current weights of *e2e_model* and return a zero-arg
    factory that restores those weights and returns the same model.

    This lets lr_selector.run_lr_sweep iterate over LRs without
    reloading the PLM from disk between trials. A single AdamW-with-
    one-LR optimizer is built per trial inside lr_selector itself,
    which matches the user's "single sweep over both backbone and
    head" choice.
    """
    init_state = {
        k: v.detach().cpu().clone()
        for k, v in e2e_model.state_dict().items()
    }

    def _factory() -> nn.Module:
        e2e_model.load_state_dict(init_state)
        return e2e_model

    return _factory


def run_lr_dynsweep_finetune(
    args,
    e2e_model: nn.Module,
    train_loader,
    val_loader,
    forward_fn: Callable,
    criterion,
    device: torch.device,
) -> float:
    """
    Run the dynamic LR sweep on rank 0 over the e2e (PLM + head) model
    and return the chosen LR. The same LR is applied to both backbone
    and head (single-LR sweep for now).

    Saves sweep_results.json + table.csv via lr_selector helpers.
    Other ranks return 0.0; the caller must broadcast the chosen LR.
    """
    # Local imports to keep this module importable without the
    # PredictionModule on sys.path during pure-utility usage.
    from lr_selector import find_best_lrs, DEFAULT_LR_GRID
    from util_helper import is_main_process

    if not is_main_process():
        return 0.0

    grid = getattr(args, "lr_dynsweep_grid", None) or DEFAULT_LR_GRID
    n_epochs = int(getattr(args, "lr_dynsweep_epochs", 2))
    alpha = float(getattr(args, "lr_dynsweep_alpha", 0.5))
    threshold = float(getattr(args, "lr_dynsweep_threshold", 0.9))
    sweep_rule = getattr(args, "sweep_rule", "4a")

    factory = make_finetune_factory(e2e_model)

    res = find_best_lrs(
        model_factory=factory,
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

    logger.info(
        f"  Dynamic sweep chosen LR (Rule {sweep_rule}): {chosen_lr:.6e}"
    )
    return float(chosen_lr)
