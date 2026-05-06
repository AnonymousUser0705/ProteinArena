"""
================================================================
Fine-Tuning Forward Functions for Protein Language Models
================================================================

Per-family differentiable forward pass functions that mirror the
extraction logic in PredictionModule/util_model.py but:
  1. Run WITH gradients (no torch.no_grad)
  2. Return ON-DEVICE tensors (not .cpu())
  3. Process a single batch of sequences (not the full protein list)

Each function takes a ModelContext + list of sequences and returns
mean-pooled embeddings as a (batch, embedding_dim) tensor.

Supported families (sequence-only):
  CARP, ESM1b, ESM2, ESMC, ProstT5, ProtT5,
  VenusPLM, xTrimoPGLM, Ankh, ProtBERT

Usage:
    from util_finetune_model import finetune_forward

    # Returns (batch, embedding_dim) tensor with gradients
    embeddings = finetune_forward(model_ctx, ["MKTL...", "MGSS..."])
================================================================
"""

import logging
import re
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


# ============================================================
# Shared T5 preprocessing (mirrors util_model._preprocess_t5_seq)
# ============================================================
def _preprocess_t5_seq(seq: str, add_prefix: bool = False) -> str:
    """Space-separate amino acids and replace rare AAs with X."""
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = " ".join(list(seq))
    if add_prefix:
        seq = "<AA2fold> " + seq
    return seq


# ============================================================
# CARP — backward-safe transpose for fine-tuning
# ============================================================
#
# The sequence_models library (MaskedConv1d, PositionFeedForward) uses
# .transpose(1,2) around Conv1d calls. Forward works fine, but during
# backward the gradient flowing through .transpose() becomes non-contiguous.
# When Conv1d.backward() or LayerNorm.backward() then call .view() on
# that non-contiguous gradient, it crashes.
#
# .contiguous() in forward does NOT fix this — it only affects the forward
# tensor. The backward gradient through transpose is independently
# non-contiguous.
#
# Fix: a custom autograd Function that guarantees contiguity in BOTH
# forward AND backward, used to replace every .transpose(1,2) call.
# ============================================================


class _SafeTranspose(torch.autograd.Function):
    """Transpose that returns contiguous tensors in both forward and backward."""

    @staticmethod
    def forward(ctx, x, dim0, dim1):
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        return x.transpose(dim0, dim1).contiguous()

    @staticmethod
    def backward(ctx, grad):
        return grad.transpose(ctx.dim0, ctx.dim1).contiguous(), None, None


def _safe_transpose(x, dim0, dim1):
    """Transpose that is safe for backward — always returns contiguous tensor."""
    return _SafeTranspose.apply(x, dim0, dim1)


_carp_patched = False


def _patch_carp_modules():
    """One-time monkey-patch of sequence_models to fix non-contiguous backward."""
    global _carp_patched
    if _carp_patched:
        return
    _carp_patched = True

    import torch.nn as nn
    from sequence_models.convolutional import MaskedConv1d
    from sequence_models.layers import PositionFeedForward

    # Patch MaskedConv1d.forward
    # Original: return super().forward(x.transpose(1, 2)).transpose(1, 2)
    def _masked_conv1d_forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return _safe_transpose(nn.Conv1d.forward(self, _safe_transpose(x, 1, 2)), 1, 2)

    MaskedConv1d.forward = _masked_conv1d_forward

    # Patch PositionFeedForward.forward
    # Original (non-factorized): return self.conv(x.transpose(1, 2)).transpose(1, 2)
    def _pff_forward(self, x):
        if self.factorized:
            w = self.u @ self.v
            return x @ w.t() + self.bias
        else:
            return _safe_transpose(self.conv(_safe_transpose(x, 1, 2)), 1, 2)

    PositionFeedForward.forward = _pff_forward

    logger.info("  Patched MaskedConv1d and PositionFeedForward with backward-safe transpose")


def _ft_forward_carp(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for CARP models.

    Applies a one-time monkey-patch to fix non-contiguous tensor issues
    in the sequence_models library, then runs through the CARP wrapper
    normally.

    Returns: (batch, embedding_dim) on device.
    """
    _patch_carp_modules()

    model, collater, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device

    batch_seqs = [[seq] for seq in sequences]
    batch_lens = [len(seq) for seq in sequences]

    x = collater(batch_seqs)[0].to(device)
    results = model(x, repr_layers=[-1], logits=False)

    # Extract from the single repr layer
    embeddings = []
    for _, rep in results["representations"].items():
        for r, ell in zip(rep, batch_lens):
            embeddings.append(r[:ell].mean(dim=0))

    return torch.stack(embeddings)  # (batch, dim)


# ============================================================
# ESM1b
# ============================================================
def _ft_forward_esm1b(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for ESM-1b.

    Max 1022 residues + 2 special tokens. Mean-pools over residue
    positions (excluding CLS and EOS tokens).

    Returns: (batch, 1280) on device.
    """
    model, batch_converter, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device
    repr_layer = model_ctx.extras["repr_layer"]

    # ESM1b max 1022 residues
    data = [(f"seq_{i}", seq[:1022]) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = results["representations"][repr_layer]

    embeddings = []
    for j, seq in enumerate(sequences):
        seq_len = min(len(seq), 1022)
        # tokens: [CLS] seq [EOS] [PAD...] — exclude CLS and EOS
        token_repr = reps[j, 1 : seq_len + 1, :]
        embeddings.append(token_repr.mean(dim=0))

    return torch.stack(embeddings)


# ============================================================
# ESM2
# ============================================================
def _ft_forward_esm2(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for ESM-2 models.

    Max 1022 residues + 2 special tokens. Mean-pools over residue
    positions (excluding CLS and EOS tokens).

    Returns: (batch, embedding_dim) on device.
    """
    model, batch_converter, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device
    repr_layer = model_ctx.extras["repr_layer"]

    data = [(f"seq_{i}", seq[:1022]) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = results["representations"][repr_layer]

    embeddings = []
    for j, seq in enumerate(sequences):
        seq_len = min(len(seq), 1022)
        token_repr = reps[j, 1 : seq_len + 1, :]
        embeddings.append(token_repr.mean(dim=0))

    return torch.stack(embeddings)


# ============================================================
# ESMC
# ============================================================
def _ft_forward_esmc(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for ESMC (ESM++).

    Uses ESM SDK's encode + logits API to get per-token embeddings,
    then mean-pools excluding BOS/EOS.

    Returns: (batch, 960) on device.
    """
    from esm.sdk.api import ESMProtein, LogitsConfig

    client, device = model_ctx.model, model_ctx.device

    embeddings = []
    for seq in sequences:
        protein = ESMProtein(sequence=seq[:1024])
        pt = client.encode(protein)
        out = client.logits(
            pt,
            LogitsConfig(return_embeddings=True, return_mean_embedding=False),
        )
        if out.embeddings is not None:
            emb = out.embeddings
            if isinstance(emb, torch.Tensor):
                if emb.dim() == 3:
                    emb = emb.squeeze(0)
                # Mean-pool excluding BOS and EOS
                emb = emb[1:-1].mean(dim=0)
            embeddings.append(emb)
        else:
            # Fallback: zero vector (should not happen in practice)
            embeddings.append(torch.zeros(960, device=device))

    return torch.stack(embeddings)


# ============================================================
# ProstT5
# ============================================================
def _ft_forward_prostt5(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for ProstT5.

    Space-separates amino acids, adds <AA2fold> prefix.
    Mean-pools with attention mask.

    Returns: (batch, 1024) on device.
    """
    model, tokenizer, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device

    batch_seqs = [
        _preprocess_t5_seq(seq[:1024], add_prefix=True)
        for seq in sequences
    ]

    encoded = tokenizer.batch_encode_plus(
        batch_seqs, add_special_tokens=True,
        padding="longest", return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    output = model(**encoded)

    last_hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return mean_pooled  # (batch, 1024)


# ============================================================
# ProtT5
# ============================================================
def _ft_forward_prott5(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for ProtT5-XL-UniRef50.

    Space-separates amino acids (no prefix). Mean-pools with attention mask.

    Returns: (batch, 1024) on device.
    """
    model, tokenizer, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device

    batch_seqs = [
        _preprocess_t5_seq(seq[:1024], add_prefix=False)
        for seq in sequences
    ]

    encoded = tokenizer.batch_encode_plus(
        batch_seqs, add_special_tokens=True,
        padding="longest", return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    output = model(**encoded)

    last_hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return mean_pooled  # (batch, 1024)


# ============================================================
# VenusPLM
# ============================================================
def _ft_forward_venusplm(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for VenusPLM.

    Uses sequence packing with 1-indexed attention masks to
    distinguish sequences in the packed batch.

    Returns: (batch, 1024) on device.
    """
    model, tokenizer, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device

    encoded = tokenizer(
        sequences,
        padding=False,
        return_attention_mask=False,
        return_length=True,
    )
    lengths = encoded["length"]

    input_ids = torch.cat(
        [torch.tensor(seq, device=device) for seq in encoded["input_ids"]],
        dim=0,
    ).to(torch.long).unsqueeze(0)

    attention_mask = torch.cat(
        [torch.ones(ell, device=device) + idx
         for idx, ell in enumerate(lengths)],
        dim=0,
    ).to(torch.long).unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=False,
    )

    hidden = outputs.hidden_states[-1].squeeze(0)  # (total_L, dim)

    embeddings = []
    offset = 0
    for ell in lengths:
        seq_hidden = hidden[offset : offset + ell]
        embeddings.append(seq_hidden.mean(dim=0))
        offset += ell

    return torch.stack(embeddings)


# ============================================================
# xTrimoPGLM
# ============================================================
def _ft_forward_xtrimopglm(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for xTrimoPGLM models.

    Tokenizes with HuggingFace AutoTokenizer, extracts last hidden state,
    mean-pools excluding boundary special tokens.

    Returns: (batch, embedding_dim) on device.
    """
    model, tokenizer, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device
    max_length = int(model_ctx.extras.get("max_length", 2048))

    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        output_hidden_states=True,
        return_last_hidden_state=True,
        return_dict=True,
    )

    hidden = getattr(outputs, "hidden_states", None)
    if hidden is None:
        raise RuntimeError("xTrimoPGLM output does not contain hidden states")
    # xTrimoPGLM returns sequence-first layout (S, B, H).
    last_hidden = hidden

    attn = encoded["attention_mask"]
    embeddings = []
    for j in range(len(sequences)):
        seq_len = int(attn[j].sum().item())
        if seq_len <= 0:
            embeddings.append(torch.zeros(last_hidden.shape[-1], device=device, dtype=last_hidden.dtype))
            continue
        # Strip only the trailing <eos> token (matches official example).
        end = seq_len - 1 if seq_len > 1 else seq_len
        token_repr = last_hidden[:end, j, :]
        if token_repr.numel() == 0:
            token_repr = last_hidden[:seq_len, j, :]
        embeddings.append(token_repr.mean(dim=0))

    return torch.stack(embeddings)


# ============================================================
# Ankh
# ============================================================
def _ft_forward_ankh(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for Ankh models (T5-based).

    Tokenizes each sequence as a list of characters (official Ankh API).
    Mean-pools with attention mask.

    Returns: (batch, embedding_dim) on device.
    """
    model, tokenizer, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device

    batch_seqs = [list(seq) for seq in sequences]

    encoded = tokenizer(
        batch_seqs, add_special_tokens=True,
        padding=True, is_split_into_words=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    output = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    last_hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return mean_pooled


# ============================================================
# ProtBERT (HuggingFace / Rostlab)
# ============================================================
def _ft_forward_protbert(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Differentiable forward for ProtBERT / ProtBERT-BFD.

    Space-separates amino acids, replaces rare AAs with X.
    Mean-pools excluding [CLS] and [SEP] tokens.

    Returns: (batch, 1024) on device.
    """
    model, tokenizer, device = model_ctx.model, model_ctx.tokenizer, model_ctx.device

    batch_seqs = [
        " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        for seq in sequences
    ]

    encoded = tokenizer(
        batch_seqs, add_special_tokens=True,
        padding=True, truncation=True,
        max_length=1024, return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    output = model(**encoded)
    last_hidden = output.last_hidden_state  # (batch, seq_len, 1024)

    embeddings = []
    for j in range(len(sequences)):
        seq_len = int(encoded["attention_mask"][j].sum().item())
        # tokens: [CLS] aa1 aa2 ... [SEP] [PAD...] — exclude CLS and SEP
        token_repr = last_hidden[j, 1 : seq_len - 1, :]
        embeddings.append(token_repr.mean(dim=0))

    return torch.stack(embeddings)


# ============================================================
# Dispatch table
# ============================================================
_FT_FORWARDS = {
    "carp": _ft_forward_carp,
    "esm1b": _ft_forward_esm1b,
    "esm2": _ft_forward_esm2,
    "esmc": _ft_forward_esmc,
    "prostt5": _ft_forward_prostt5,
    "prott5": _ft_forward_prott5,
    "venusplm": _ft_forward_venusplm,
    "xtrimopglm": _ft_forward_xtrimopglm,
    "ankh": _ft_forward_ankh,
    "protbert": _ft_forward_protbert,
}

# Families that are NOT supported for fine-tuning (structure / TF)
_UNSUPPORTED_FT_FAMILIES = {"esmif", "proteinbert"}


def finetune_forward(model_ctx, sequences: List[str]) -> torch.Tensor:
    """
    Unified differentiable forward pass for any supported PLM family.

    Dispatches to the appropriate family-specific function based on
    model_ctx.spec.family.

    Args:
        model_ctx: ModelContext from util_model.load_model()
        sequences: List of amino acid strings (one per protein)

    Returns:
        Tensor of shape (len(sequences), embedding_dim) on model device,
        with gradients attached for backpropagation.
    """
    family = model_ctx.spec.family

    if family in _UNSUPPORTED_FT_FAMILIES:
        raise ValueError(
            f"Family '{family}' ({model_ctx.spec.name}) is not supported for "
            f"fine-tuning. Structure-based and TensorFlow models are excluded."
        )

    fn = _FT_FORWARDS.get(family)
    if fn is None:
        raise ValueError(
            f"No fine-tuning forward function for family '{family}'. "
            f"Supported: {sorted(_FT_FORWARDS.keys())}"
        )

    return fn(model_ctx, sequences)


def get_supported_ft_families() -> List[str]:
    """Return list of PLM families that support fine-tuning."""
    return sorted(_FT_FORWARDS.keys())
