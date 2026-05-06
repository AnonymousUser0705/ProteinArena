"""
================================================================
Model Registry & Embedding Extraction for Protein Language Models
================================================================

Unified interface for loading pretrained PLMs and extracting
mean-pooled protein embeddings. Each model family registers a
loader and an extractor; the generic `load_model` / `extract_embeddings`
functions dispatch automatically.

Supported model families:
  CARP, ESM1b, ESM2, ESMC, ESM-IF, ProstT5, ProtT5,
    VenusPLM, xTrimoPGLM, Ankh, ProteinBERT

Usage:
    from util_model import load_model, extract_embeddings, MODEL_REGISTRY

    ctx = load_model("esm2_650M", "/path/to/model", device)
    embs = extract_embeddings(ctx, protein_ids, sequences, batch_size=8)
    # embs: {protein_id: Tensor(embedding_dim,)}
================================================================
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================
@dataclass
class ModelSpec:
    """Metadata for a PLM model variant."""
    name: str
    family: str              # e.g. "carp", "esm2", "esmc", ...
    input_type: str          # "sequence" or "structure"
    embedding_dim: int
    description: str = ""
    framework: str = "pytorch"  # "pytorch" or "tensorflow"
    default_path: str = ""
    load_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelContext:
    """Container returned by load_model() — holds everything needed for extraction."""
    model: Any
    tokenizer: Any           # tokenizer / collater / batch_converter / input_encoder
    spec: ModelSpec
    device: torch.device
    extras: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Model Registry
# ============================================================
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # --- CARP ---
    "carp_600k": ModelSpec(
        name="carp_600k", family="carp", input_type="sequence",
        embedding_dim=128, description="CARP 600K ByteNet",
        default_path="carp_600k",
    ),
    "carp_38M": ModelSpec(
        name="carp_38M", family="carp", input_type="sequence",
        embedding_dim=1024, description="CARP 38M ByteNet",
        default_path="carp_38M",
    ),
    "carp_640M": ModelSpec(
        name="carp_640M", family="carp", input_type="sequence",
        embedding_dim=1280, description="CARP 640M ByteNet",
        default_path="carp_640M",
    ),
    # --- ESM1b ---
    "esm1b": ModelSpec(
        name="esm1b", family="esm1b", input_type="sequence",
        embedding_dim=1280, description="ESM-1b 650M (t33, UR50S)",
        load_kwargs={"repr_layer": 33},
    ),
    # --- ESM2 ---
    "esm2_150M": ModelSpec(
        name="esm2_150M", family="esm2", input_type="sequence",
        embedding_dim=640, description="ESM-2 150M (t30)",
        load_kwargs={"repr_layer": 30},
    ),
    "esm2_650M": ModelSpec(
        name="esm2_650M", family="esm2", input_type="sequence",
        embedding_dim=1280, description="ESM-2 650M (t33)",
        default_path="/gpfs/project/projects/CompCellBio/Models/ESM2/esm2_t33_650M_UR50D.pt",
        load_kwargs={"repr_layer": 33},
    ),
    # --- ESMC (ESM++) ---
    "esmc": ModelSpec(
        name="esmc", family="esmc", input_type="sequence",
        embedding_dim=960, description="ESM++ / ESMC 300M",
    ),
    # --- ESM-IF (structure-based) ---
    "esmif": ModelSpec(
        name="esmif", family="esmif", input_type="structure",
        embedding_dim=512, description="ESM-IF1 inverse folding (structure)",
    ),
    # --- ESM-GearNet (joint sequence+structure, torchdrug) ---
    "esmgearnet": ModelSpec(
        name="esmgearnet", family="esmgearnet", input_type="structure",
        embedding_dim=4352,
        description="ESM-GearNet (ESM-2-650M + GearNet, series fusion, SiamDiff pretrained)",
    ),
    # --- ProstT5 ---
    "prostt5": ModelSpec(
        name="prostt5", family="prostt5", input_type="sequence",
        embedding_dim=1024, description="ProstT5 encoder",
    ),
    # --- ProtT5 ---
    "prott5": ModelSpec(
        name="prott5", family="prott5", input_type="sequence",
        embedding_dim=1024, description="ProtT5-XL-UniRef50 encoder",
    ),
    # --- VenusPLM ---
    "venusplm_300M": ModelSpec(
        name="venusplm_300M", family="venusplm", input_type="sequence",
        embedding_dim=1024, description="VenusPLM 300M masked LM",
        default_path="AI4Protein/VenusPLM-300M",
    ),
    # --- xTrimoPGLM ---
    "xtrimopglm_100b_int4": ModelSpec(
        name="xtrimopglm_100b_int4", family="xtrimopglm", input_type="sequence",
        embedding_dim=10240, description="xTrimoPGLM 100B INT4",
        default_path="biomap-research/xtrimopglm-100b-int4",
        load_kwargs={"max_length": 2048},
    ),
    "xtrimopglm_1b": ModelSpec(
        name="xtrimopglm_1b", family="xtrimopglm", input_type="sequence",
        embedding_dim=2560, description="xTrimoPGLM 1B (proteinglm-1b-mlm)",
        default_path="Bo1015/proteinglm-1b-mlm",
        load_kwargs={"max_length": 2048},
    ),
    # --- Ankh ---
    "ankh_base": ModelSpec(
        name="ankh_base", family="ankh", input_type="sequence",
        embedding_dim=768, description="Ankh Base T5 encoder",
    ),
    "ankh_large": ModelSpec(
        name="ankh_large", family="ankh", input_type="sequence",
        embedding_dim=1536, description="Ankh Large T5 encoder",
    ),
    "ankh3_xl": ModelSpec(
        name="ankh3_xl", family="ankh", input_type="sequence",
        embedding_dim=2560, description="Ankh3 XL T5 encoder",
    ),
    # --- ProteinBERT (TensorFlow) ---
    "proteinbert": ModelSpec(
        name="proteinbert", family="proteinbert", input_type="sequence",
        embedding_dim=640, description="ProteinBERT (TF/Keras, 512-d local + 128-d global)",
        framework="tensorflow",
    ),
    # --- ProtBERT (HuggingFace / Rostlab) ---
    "protbert": ModelSpec(
        name="protbert", family="protbert", input_type="sequence",
        embedding_dim=1024, description="ProtBERT (Rostlab/prot_bert)",
        default_path="Rostlab/prot_bert",
    ),
    "protbert_bfd": ModelSpec(
        name="protbert_bfd", family="protbert", input_type="sequence",
        embedding_dim=1024, description="ProtBERT-BFD (Rostlab/prot_bert_bfd)",
        default_path="Rostlab/prot_bert_bfd",
    ),
}


# ============================================================
# Public helpers
# ============================================================
def list_models() -> None:
    """Print all registered models."""
    print(f"{'Name':<20s} {'Dim':>6s} {'Input':<12s} {'Description'}")
    print("-" * 70)
    for name, s in MODEL_REGISTRY.items():
        print(f"  {name:<20s} {s.embedding_dim:>5d}  {s.input_type:<12s} {s.description}")


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name]


# ============================================================
# Generic Load / Extract  (dispatch on spec.family)
# ============================================================
def load_model(model_name: str, model_path: str, device: torch.device) -> ModelContext:
    """Load a PLM by registry name.  Returns a ModelContext."""
    spec = get_model_spec(model_name)
    path = model_path or spec.default_path
    logger.info(f"Loading model: {spec.name} ({spec.description})")
    logger.info(f"  Path:          {path}")
    logger.info(f"  Embedding dim: {spec.embedding_dim}")

    loader = _LOADERS.get(spec.family)
    if loader is None:
        raise ValueError(f"No loader for family '{spec.family}'")
    return loader(spec, path, device)


def extract_embeddings(
    model_ctx: ModelContext,
    protein_ids: List[str],
    sequences: Optional[Dict[str, str]] = None,
    pdb_dir: Optional[str] = None,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Extract mean-pooled embeddings for a list of protein IDs.

    For sequence-based models supply *sequences* (dict protein_id → AA string).
    For structure-based models supply *pdb_dir* (directory of {id}.pdb files).

    When DDP is active, the protein list is automatically sharded across ranks
    and results are gathered so every rank gets the complete dict.

    Returns: {protein_id: Tensor(embedding_dim,)} on CPU.
    """
    import torch.distributed as dist

    spec = model_ctx.spec
    extractor = _EXTRACTORS.get(spec.family)
    if extractor is None:
        raise ValueError(f"No extractor for family '{spec.family}'")

    use_ddp = dist.is_initialized()

    # ---- Shard protein IDs across ranks ----
    if use_ddp:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_ids = [pid for i, pid in enumerate(protein_ids) if i % world == rank]
        logger.info(f"  [Rank {rank}] Extracting {len(local_ids)}/{len(protein_ids)} proteins")
    else:
        local_ids = protein_ids

    # ---- Run family-specific extractor on local shard ----
    if spec.input_type == "structure":
        if pdb_dir is None:
            raise ValueError(f"Model '{spec.name}' requires pdb_dir for structure input")
        local_embs = extractor(model_ctx, local_ids, pdb_dir=pdb_dir, batch_size=batch_size)
    else:
        if sequences is None:
            raise ValueError(f"Model '{spec.name}' requires sequences for sequence input")
        local_embs = extractor(model_ctx, local_ids, sequences=sequences, batch_size=batch_size)

    # ---- Gather across ranks ----
    if use_ddp:
        import pickle
        data = pickle.dumps(local_embs)
        size = torch.tensor([len(data)], dtype=torch.long, device=model_ctx.device)

        # Collect sizes from all ranks
        all_sizes = [torch.zeros(1, dtype=torch.long, device=model_ctx.device)
                     for _ in range(world)]
        dist.all_gather(all_sizes, size)

        max_size = max(s.item() for s in all_sizes)
        buf = torch.zeros(max_size, dtype=torch.uint8, device=model_ctx.device)
        buf[:len(data)] = torch.tensor(list(data), dtype=torch.uint8, device=model_ctx.device)

        all_bufs = [torch.zeros(max_size, dtype=torch.uint8, device=model_ctx.device)
                    for _ in range(world)]
        dist.all_gather(all_bufs, buf)

        # Merge all shards
        embeddings = {}
        for i in range(world):
            shard_data = bytes(all_bufs[i][:all_sizes[i].item()].cpu().tolist())
            shard_embs = pickle.loads(shard_data)
            embeddings.update(shard_embs)

        logger.info(f"  [Rank {dist.get_rank()}] Gathered {len(embeddings)} "
                    f"total embeddings from {world} ranks")
        return embeddings

    return local_embs


# ============================================================
# CARP
# ============================================================
def _load_carp(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from sequence_models.pretrained import load_model_and_alphabet
    model, collater = load_model_and_alphabet(model_path)
    model = model.to(device).eval()
    logger.info(f"  CARP model loaded on {device}")
    return ModelContext(model=model, tokenizer=collater, spec=spec, device=device)


def _extract_carp(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, collater, device = ctx.model, ctx.tokenizer, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [[sequences[pid]] for pid in batch_ids]
        batch_lens = [len(sequences[pid]) for pid in batch_ids]

        x = collater(batch_seqs)[0].to(device)
        with torch.no_grad():
            results = model(x, repr_layers=[-1], logits=False)

        for layer, rep in results["representations"].items():
            for pid, r, ell in zip(batch_ids, rep, batch_lens):
                embeddings[pid] = r[:ell].mean(dim=0).cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  CARP extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} CARP embeddings")
    return embeddings


# ============================================================
# ESM1b
# ============================================================
def _load_esm1b(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    import argparse
    import esm

    # PyTorch ≥2.6 defaults weights_only=True; ESM checkpoints store argparse.Namespace
    # objects (model args), so we must allowlist that type before loading.
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])

    if model_path and os.path.exists(model_path):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    repr_layer = spec.load_kwargs.get("repr_layer", 33)
    logger.info(f"  ESM1b loaded, repr_layer={repr_layer}")
    return ModelContext(
        model=model, tokenizer=batch_converter, spec=spec, device=device,
        extras={"alphabet": alphabet, "repr_layer": repr_layer},
    )


def _extract_esm1b(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, batch_converter, device = ctx.model, ctx.tokenizer, ctx.device
    repr_layer = ctx.extras["repr_layer"]
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        # ESM1b max 1022 residues + 2 special tokens = 1024
        data = [(pid, sequences[pid][:1022]) for pid in batch_ids]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        reps = results["representations"][repr_layer]
        for j, pid in enumerate(batch_ids):
            seq_len = min(len(sequences[pid]), 1022)
            # tokens layout: [CLS] seq [EOS] [PAD...]
            token_repr = reps[j, 1 : seq_len + 1, :]  # exclude CLS and EOS
            embeddings[pid] = token_repr.mean(dim=0).cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM1b extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ESM1b embeddings")
    return embeddings


# ============================================================
# ESM2
# ============================================================
def _load_esm2(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    import argparse
    import esm

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])

    if model_path and os.path.exists(model_path):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        hub = {
            "esm2_8M": esm.pretrained.esm2_t6_8M_UR50D,
            "esm2_35M": esm.pretrained.esm2_t12_35M_UR50D,
            "esm2_150M": esm.pretrained.esm2_t30_150M_UR50D,
            "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
            "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
        }
        fn = hub.get(spec.name)
        if fn is None:
            raise ValueError(f"No ESM2 hub loader for '{spec.name}'. Provide model_path.")
        model, alphabet = fn()

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    repr_layer = spec.load_kwargs.get("repr_layer", 33)
    logger.info(f"  ESM2 loaded, repr_layer={repr_layer}")
    return ModelContext(
        model=model, tokenizer=batch_converter, spec=spec, device=device,
        extras={"alphabet": alphabet, "repr_layer": repr_layer},
    )


def _extract_esm2(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, batch_converter, device = ctx.model, ctx.tokenizer, ctx.device
    repr_layer = ctx.extras["repr_layer"]
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        # ESM2 max 1022 residues + 2 special tokens = 1024
        data = [(pid, sequences[pid][:1022]) for pid in batch_ids]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        reps = results["representations"][repr_layer]
        for j, pid in enumerate(batch_ids):
            seq_len = min(len(sequences[pid]), 1022)
            # tokens layout: [CLS] seq [EOS] [PAD...]
            token_repr = reps[j, 1 : seq_len + 1, :]  # exclude CLS and EOS
            embeddings[pid] = token_repr.mean(dim=0).cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM2 extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ESM2 embeddings")
    return embeddings


# ============================================================
# ESMC
# ============================================================
def _load_esmc(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from esm.models.esmc import ESMC
    from esm.tokenization import EsmSequenceTokenizer

    if model_path and os.path.exists(model_path) and not os.path.isdir(model_path):
        # Local .pth checkpoint — use correct architecture and tokenizer from SDK
        from esm.tokenization import get_esmc_model_tokenizers
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        d_model = 960
        n_heads = 15
        n_layers = 30
        with torch.device("cpu"):
            client = ESMC(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=False,
            ).eval()
        client.load_state_dict(state_dict)
        client = client.to(device)
        logger.info(f"  ESMC loaded from local checkpoint: {model_path}")
        logger.info(f"    d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    else:
        # Registry / HF hub name
        name = model_path or "esmc_300m_2024_12_v0"
        client = ESMC.from_pretrained(name).to(device)
        logger.info(f"  ESMC loaded from registry: {name}")

    client.eval()
    return ModelContext(model=client, tokenizer=None, spec=spec, device=device)


def _extract_esmc(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    from esm.sdk.api import ESMProtein, LogitsConfig

    client, device = ctx.model, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [sequences[pid][:1024] for pid in batch_ids]

        with torch.no_grad():
            proteins = [ESMProtein(sequence=seq) for seq in batch_seqs]
            protein_tensors = [client.encode(p) for p in proteins]

            # Process one protein at a time — local ESMC does not return mean_embedding
            # so we get per-token embeddings and mean-pool ourselves (excluding BOS/EOS)
            mean_embs = []
            for pt in protein_tensors:
                out = client.logits(
                    pt,
                    LogitsConfig(return_embeddings=True, return_mean_embedding=False)
                )
                if out.embeddings is not None:
                    emb = out.embeddings  # shape: (1, seq_len, d_model) or (seq_len, d_model)
                    if isinstance(emb, torch.Tensor):
                        if emb.dim() == 3:
                            emb = emb.squeeze(0)  # -> (seq_len, d_model)
                        # Mean-pool over sequence positions excluding first (BOS) and last (EOS)
                        emb = emb[1:-1].mean(dim=0)  # -> (d_model,)
                    mean_embs.append(emb)
                else:
                    mean_embs.append(None)

        for j, pid in enumerate(batch_ids):
            if j < len(mean_embs) and mean_embs[j] is not None:
                emb = mean_embs[j]
                embeddings[pid] = emb.cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb).cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESMC extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ESMC embeddings (ESM SDK API)")
    return embeddings


# ============================================================
# ESM-IF  (structure-based)
# ============================================================
def _load_esmif(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    import argparse
    import esm

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])

    if model_path and os.path.exists(model_path):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    model = model.to(device).eval()
    logger.info("  ESM-IF loaded")
    return ModelContext(
        model=model, tokenizer=None, spec=spec, device=device,
        extras={"alphabet": alphabet},
    )


def _extract_esmif(ctx: ModelContext, protein_ids, *, pdb_dir=None, batch_size=1, **kw):
    import esm.inverse_folding.util
    import esm.inverse_folding.multichain_util

    model, device = ctx.model, ctx.device
    alphabet = ctx.extras["alphabet"]
    embeddings = {}

    for idx, pid in enumerate(protein_ids):
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        if not os.path.exists(pdb_path):
            logger.warning(f"  PDB not found: {pdb_path} — skipping {pid}")
            continue
        try:
            # Load all chains to auto-detect single vs multi-chain
            structure = esm.inverse_folding.util.load_structure(pdb_path, chain=None)
            coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
            chain_ids = list(coords.keys())

            with torch.no_grad():
                if len(chain_ids) == 1:
                    # Single chain: use simple encoder output
                    rep = esm.inverse_folding.util.get_encoder_output(
                        model, alphabet, coords[chain_ids[0]]
                    )
                    embeddings[pid] = rep.mean(dim=0).cpu()
                else:
                    # Multi-chain: get encoder output for each chain
                    # conditioned on the full complex, then pool across all
                    chain_reps = []
                    for cid in chain_ids:
                        rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(
                            model, alphabet, coords, cid
                        )
                        chain_reps.append(rep)
                    all_reps = torch.cat(chain_reps, dim=0)  # (total_residues, 512)
                    embeddings[pid] = all_reps.mean(dim=0).cpu()
        except Exception as e:
            logger.warning(f"  Error processing {pid}: {e}")

        if idx % 50 == 0:
            logger.info(f"  ESM-IF extraction: {idx + 1}/{len(protein_ids)}")

    logger.info(f"  Extracted {len(embeddings)} ESM-IF embeddings")
    return embeddings


# ============================================================
# Shared T5 preprocessing
# ============================================================
def _preprocess_t5_seq(seq: str, add_prefix: bool = False) -> str:
    """Space-separate amino acids and replace rare AAs with X."""
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = " ".join(list(seq))
    if add_prefix:
        seq = "<AA2fold> " + seq
    return seq


# ============================================================
# ProstT5
# ============================================================
def _load_prostt5(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    logger.info(f"  ProstT5 loaded from {model_path}")
    return ModelContext(model=model, tokenizer=tokenizer, spec=spec, device=device)


def _extract_prostt5(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, tokenizer, device = ctx.model, ctx.tokenizer, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [
            _preprocess_t5_seq(sequences[pid][:2048], add_prefix=True)
            for pid in batch_ids
        ]

        encoded = tokenizer.batch_encode_plus(
            batch_seqs, add_special_tokens=True,
            padding="longest", return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        last_hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        for j, pid in enumerate(batch_ids):
            embeddings[pid] = mean_pooled[j].cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ProstT5 extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ProstT5 embeddings")
    return embeddings


# ============================================================
# ProtT5
# ============================================================
def _load_prott5(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    logger.info(f"  ProtT5 loaded from {model_path}")
    return ModelContext(model=model, tokenizer=tokenizer, spec=spec, device=device)


def _extract_prott5(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, tokenizer, device = ctx.model, ctx.tokenizer, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [
            _preprocess_t5_seq(sequences[pid][:2048], add_prefix=False)
            for pid in batch_ids
        ]

        encoded = tokenizer.batch_encode_plus(
            batch_seqs, add_special_tokens=True,
            padding="longest", return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        last_hidden = output.last_hidden_state
        # Drop the EOS (</s>) token at the last non-pad position of each sequence
        # before mean-pooling, per the ProtT5 HF model card recipe.
        mask = encoded["attention_mask"].clone()
        last_idx = mask.sum(dim=1, keepdim=True) - 1
        mask.scatter_(1, last_idx, 0)
        mask = mask.unsqueeze(-1).to(last_hidden.dtype)
        mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        for j, pid in enumerate(batch_ids):
            embeddings[pid] = mean_pooled[j].float().cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ProtT5 extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ProtT5 embeddings")
    return embeddings


# ============================================================
# VenusPLM
# ============================================================
def _load_venusplm(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from vplm import TransformerConfig, TransformerForMaskedLM, VPLMTokenizer

    config = TransformerConfig.from_pretrained(model_path, attn_impl="sdpa") # or "flash_attn" if installed
    model = TransformerForMaskedLM.from_pretrained(model_path, config=config)
    tokenizer = VPLMTokenizer.from_pretrained(model_path)
    model = model.to(device).eval()
    logger.info(f"  VenusPLM loaded from {model_path}")
    return ModelContext(model=model, tokenizer=tokenizer, spec=spec, device=device)


def _extract_venusplm(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, tokenizer, device = ctx.model, ctx.tokenizer, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [sequences[pid] for pid in batch_ids]

        # Sequence packing: tokenize without padding, then concat
        encoded = tokenizer(
            batch_seqs,
            padding=False,
            return_attention_mask=False,
            return_length=True,
        )
        lengths = encoded["length"]

        # concat the input_ids along the length dimension
        input_ids = torch.cat(
            [torch.tensor(seq, device=device) for seq in encoded["input_ids"]],
            dim=0,
        ).to(torch.long).unsqueeze(0)  # [1, total_L]

        # attention_mask with 1-indexed sequence offsets to distinguish sequences
        attention_mask = torch.cat(
            [torch.ones(ell, device=device) + idx
             for idx, ell in enumerate(lengths)],
            dim=0,
        ).to(torch.long).unsqueeze(0)  # [1, total_L]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
            )

        hidden = outputs.hidden_states[-1].squeeze(0)  # (total_L, dim)

        # Split by sequence lengths and mean-pool each
        offset = 0
        for j, pid in enumerate(batch_ids):
            seq_hidden = hidden[offset : offset + lengths[j]]
            embeddings[pid] = seq_hidden.mean(dim=0).cpu()
            offset += lengths[j]

        if (i // batch_size) % 50 == 0:
            logger.info(f"  VenusPLM extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} VenusPLM embeddings")
    return embeddings


# ============================================================
# xTrimoPGLM
# ============================================================
def _load_xtrimopglm(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    if hasattr(config, "is_causal"):
        config.is_causal = False

    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    hidden_size = int(getattr(config, "hidden_size", spec.embedding_dim))
    if hidden_size != spec.embedding_dim:
        logger.warning(
            "  xTrimoPGLM hidden_size=%d differs from registry embedding_dim=%d",
            hidden_size,
            spec.embedding_dim,
        )

    logger.info(f"  xTrimoPGLM loaded from {model_path}")
    return ModelContext(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        device=device,
        extras={
            "max_length": int(spec.load_kwargs.get("max_length", 2048)),
            "embedding_dim": hidden_size,
        },
    )


def _extract_xtrimopglm(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=1, **kw):
    model, tokenizer, device = ctx.model, ctx.tokenizer, ctx.device
    max_length = int(ctx.extras.get("max_length", 2048))

    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [sequences[pid] for pid in batch_ids]

        encoded = tokenizer(
            batch_seqs,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

        with torch.no_grad():
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
        for j, pid in enumerate(batch_ids):
            seq_len = int(attn[j].sum().item())
            if seq_len <= 0:
                continue

            # Strip only the trailing <eos> token (matches official example).
            end = seq_len - 1 if seq_len > 1 else seq_len
            token_repr = last_hidden[:end, j, :]
            if token_repr.numel() == 0:
                token_repr = last_hidden[:seq_len, j, :]

            embeddings[pid] = token_repr.float().mean(dim=0).cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  xTrimoPGLM extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} xTrimoPGLM embeddings")
    return embeddings


# ============================================================
# Ankh
# ============================================================
def _load_ankh(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    # Mirrors the official ankh library exactly (T5EncoderModel + tokenizer),
    # but loads from a local path so no internet / ankh package needed.
    from transformers import T5EncoderModel, AutoTokenizer, T5Tokenizer

    # ankh3 variants use T5Tokenizer; ankh base/large use AutoTokenizer
    if spec.name.startswith("ankh3"):
        tokenizer = T5Tokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = T5EncoderModel.from_pretrained(model_path)
    model = model.to(device).eval()
    logger.info(f"  Ankh loaded: {spec.name} from {model_path}")
    return ModelContext(model=model, tokenizer=tokenizer, spec=spec, device=device)


def _extract_ankh(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, tokenizer, device = ctx.model, ctx.tokenizer, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        # Official ankh tokenization: split each sequence into a list of characters
        batch_seqs = [list(sequences[pid]) for pid in batch_ids]

        encoded = tokenizer(
            batch_seqs, add_special_tokens=True,
            padding=True, is_split_into_words=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(input_ids=encoded['input_ids'],
                           attention_mask=encoded['attention_mask'])

        last_hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        for j, pid in enumerate(batch_ids):
            embeddings[pid] = mean_pooled[j].cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  Ankh extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} Ankh embeddings")
    return embeddings


# ============================================================
# ProteinBERT  (TensorFlow / Keras)
# ============================================================
def _load_proteinbert(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from proteinbert import load_pretrained_model

    if model_path and os.path.isdir(model_path):
        model_generator, input_encoder = load_pretrained_model(
            local_model_dump_dir=model_path,
            download_model_dump_if_not_exists=False,
            validate_downloading=False,
        )
    else:
        model_generator, input_encoder = load_pretrained_model(
            download_model_dump_if_not_exists=True,
            validate_downloading=False,
        )

    logger.info("  ProteinBERT loaded (TensorFlow/Keras)")
    return ModelContext(
        model=model_generator, tokenizer=input_encoder,
        spec=spec, device=device,
    )


def _extract_proteinbert(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=32, **kw):
    from tensorflow import keras

    model_generator = ctx.model
    input_encoder = ctx.tokenizer
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]

    if not ids_list:
        return embeddings

    all_seqs = [sequences[pid] for pid in ids_list]
    max_seq_len = min(max(len(s) for s in all_seqs) + 2, 1026)  # +2 for START/END

    # Extract only the final hidden states (inputs to the output Dense layers).
    # get_model_with_hidden_layers_as_outputs concatenates ALL LayerNorm layers across
    # all 6 blocks, producing (6*2*128 + vocab_size) + (512 + 6*2*512 + n_annotations)
    # = 17161-d. We want only the last hidden_seq (128-d) + hidden_global (512-d) = 640-d.
    base_model = model_generator.create_model(max_seq_len)
    final_local = base_model.get_layer('output-seq').input    # (batch, seq_len, 128)
    final_global = base_model.get_layer('output-annotations').input  # (batch, 512)
    model = keras.models.Model(inputs=base_model.inputs, outputs=[final_local, final_global])

    total = len(ids_list)
    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        batch_seqs = [sequences[pid][:max_seq_len - 2] for pid in batch_ids]
        encoded_x = input_encoder.encode_X(batch_seqs, max_seq_len)

        local_representations, global_representations = model.predict(
            encoded_x, batch_size=len(batch_ids), verbose=0
        )
        # local_representations shape: (batch, seq_len, 128)
        # global_representations shape: (batch, 512)

        for j, pid in enumerate(batch_ids):
            seq_len_actual = min(len(sequences[pid]), max_seq_len - 2)
            # Mean-pool local representations over residue positions (skip padding)
            local_mean = local_representations[j, :seq_len_actual, :].mean(axis=0)
            global_vec = global_representations[j]
            combined = np.concatenate([local_mean, global_vec], axis=0)
            embeddings[pid] = torch.from_numpy(combined.copy()).float()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ProteinBERT extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ProteinBERT embeddings")
    return embeddings


# ============================================================
# ProtBERT  (HuggingFace / Rostlab)
# ============================================================
def _load_protbert(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    from transformers import BertModel, BertTokenizer

    path = model_path or spec.default_path
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
    model = BertModel.from_pretrained(path).to(device).eval()
    logger.info(f"  ProtBERT loaded from {path}")
    return ModelContext(model=model, tokenizer=tokenizer, spec=spec, device=device)


def _extract_protbert(ctx: ModelContext, protein_ids, *, sequences=None, batch_size=8, **kw):
    model, tokenizer, device = ctx.model, ctx.tokenizer, ctx.device
    embeddings = {}
    ids_list = [p for p in protein_ids if p in sequences]
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        # ProtBERT expects space-separated amino acids, rare AAs replaced with X
        batch_seqs = [
            " ".join(list(re.sub(r"[UZOB]", "X", sequences[pid])))
            for pid in batch_ids
        ]

        encoded = tokenizer(
            batch_seqs, add_special_tokens=True,
            padding=True, truncation=True,
            max_length=2048, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        last_hidden = output.last_hidden_state  # (batch, seq_len, 1024)

        for j, pid in enumerate(batch_ids):
            seq_len = int(encoded["attention_mask"][j].sum().item())
            # tokens layout: [CLS] aa1 aa2 ... aaN [SEP] [PAD...]
            # exclude [CLS] (index 0) and [SEP] (index seq_len-1)
            token_repr = last_hidden[j, 1 : seq_len - 1, :]
            embeddings[pid] = token_repr.mean(dim=0).cpu()

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ProtBERT extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ProtBERT embeddings")
    return embeddings


# ============================================================
# ESM-GearNet  (DeepGraphLearning/ESM-GearNet, torchdrug-based)
# ============================================================
ESM2_DIR_DEFAULT = "/gpfs/project/projects/CompCellBio/Models/ESM2"


def _load_esmgearnet(spec: ModelSpec, model_path: str, device: torch.device) -> ModelContext:
    """
    Build the ESM-2-650M + GearNet FusionNetwork (series fusion) and load
    the SiamDiff-pretrained checkpoint via `model.load_state_dict(...)` —
    matching the recipe at ESM-GearNet/util.py:159-160.

    Requires the `esm_gearnet` conda env (torchdrug, torch_scatter, the ESM-GearNet
    repo's `gearnet` package on PYTHONPATH).
    """
    # Lazy imports — torchdrug stack only exists in the esm_gearnet env
    from torchdrug import models, layers
    from torchdrug.layers import geometry
    from gearnet.model import FusionNetwork

    sequence_model = models.ESM(
        path=ESM2_DIR_DEFAULT,
        model="ESM-2-650M",
    )
    structure_model = models.GearNet(
        input_dim=1280,
        hidden_dims=[512, 512, 512, 512, 512, 512],
        batch_norm=True,
        concat_hidden=True,
        short_cut=True,
        readout="sum",
        num_relation=7,
    )
    model = FusionNetwork(sequence_model, structure_model, fusion="series")

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SiamDiff checkpoint not found: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"  ESM-GearNet checkpoint loaded from {model_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        logger.warning(
            "  No --model_path provided for esmgearnet; using uninitialized "
            "FusionNetwork weights (ESM-2 sub-model still pretrained)"
        )

    graph_construction = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
        ],
    )

    model = model.to(device).eval()
    graph_construction = graph_construction.to(device)
    logger.info(f"  ESM-GearNet ready (output_dim={model.output_dim})")

    return ModelContext(
        model=model, tokenizer=None, spec=spec, device=device,
        extras={"graph_construction": graph_construction},
    )


def _extract_esmgearnet(ctx: ModelContext, protein_ids, *, pdb_dir=None, batch_size=4, **kw):
    from torchdrug import data, transforms

    if not pdb_dir:
        raise ValueError("ESM-GearNet requires --pdb_dir (structure-based model)")

    model, device = ctx.model, ctx.device
    graph_construction = ctx.extras["graph_construction"]
    embeddings: Dict[str, torch.Tensor] = {}

    ESM_MAX_RESIDUES = 1022
    truncate = transforms.TruncateProtein(max_length=ESM_MAX_RESIDUES, random=False)
    logger.info(
        f"  ESM-GearNet truncation policy: max_length={ESM_MAX_RESIDUES}, random=False "
        f"(deterministic N-terminal slice — required so GearNet adjacency matches ESM's 1022-residue cap)"
    )

    ids_list = list(protein_ids)
    total = len(ids_list)
    truncated_count = 0

    for i in range(0, total, batch_size):
        batch_ids = ids_list[i : i + batch_size]
        proteins, valid_ids = [], []
        for pid in batch_ids:
            pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
            if not os.path.exists(pdb_path):
                logger.warning(f"  Missing PDB for {pid}: {pdb_path}")
                continue
            try:
                p = data.Protein.from_pdb(
                    pdb_path,
                    atom_feature="position",
                    bond_feature=None,
                    residue_feature="default",
                )
                if p.num_residue > ESM_MAX_RESIDUES:
                    orig_len = p.num_residue
                    p = truncate({"graph": p})["graph"]
                    truncated_count += 1
                    logger.warning(
                        f"  Truncated {pid}: {orig_len} -> {ESM_MAX_RESIDUES} residues "
                        f"(C-terminal {orig_len - ESM_MAX_RESIDUES} residues dropped)"
                    )
                proteins.append(p)
                valid_ids.append(pid)
            except Exception as e:
                logger.warning(f"  Failed to parse PDB for {pid}: {e}")

        if not proteins:
            continue

        graph = data.Protein.pack(proteins).to(device)
        graph = graph_construction(graph)
        residue_features = graph.residue_feature.float()

        with torch.no_grad():
            output = model(graph, residue_features)

        graph_feature = output["graph_feature"].cpu()  # (B, 4352)
        for j, pid in enumerate(valid_ids):
            embeddings[pid] = graph_feature[j]

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM-GearNet extraction: {min(i + batch_size, total)}/{total}")

    logger.info(f"  Extracted {len(embeddings)} ESM-GearNet embeddings")
    if truncated_count:
        logger.info(
            f"  ESM-GearNet truncation summary: {truncated_count}/{total} proteins "
            f"exceeded {ESM_MAX_RESIDUES} residues and were truncated to N-terminal {ESM_MAX_RESIDUES}"
        )
    return embeddings


# ============================================================
# Dispatch tables
# ============================================================
_LOADERS = {
    "carp": _load_carp,
    "esm1b": _load_esm1b,
    "esm2": _load_esm2,
    "esmc": _load_esmc,
    "esmif": _load_esmif,
    "prostt5": _load_prostt5,
    "prott5": _load_prott5,
    "venusplm": _load_venusplm,
    "xtrimopglm": _load_xtrimopglm,
    "ankh": _load_ankh,
    "proteinbert": _load_proteinbert,
    "protbert": _load_protbert,
    "esmgearnet": _load_esmgearnet,
}

_EXTRACTORS = {
    "carp": _extract_carp,
    "esm1b": _extract_esm1b,
    "esm2": _extract_esm2,
    "esmc": _extract_esmc,
    "esmif": _extract_esmif,
    "prostt5": _extract_prostt5,
    "prott5": _extract_prott5,
    "venusplm": _extract_venusplm,
    "xtrimopglm": _extract_xtrimopglm,
    "ankh": _extract_ankh,
    "proteinbert": _extract_proteinbert,
    "protbert": _extract_protbert,
    "esmgearnet": _extract_esmgearnet,
}
