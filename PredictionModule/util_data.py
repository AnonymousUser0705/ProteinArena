"""
================================================================
Data Handling for PredictionModule
================================================================

Shared data utilities for PRING PPI prediction and CAFA5 GO term
prediction.  Loads split files, collects unique proteins, provides
PyTorch Datasets/collates, and handles embedding I/O.
================================================================
"""

import ast
import logging
import os
import sys
from os.path import join
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ============================================================
# Split Loading
# ============================================================
def load_pring_splits(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load train.tsv / val.tsv / test.tsv from *splits_dir*.

    Returns: {split_name: DataFrame} for each file that exists.
    Expected TSV columns: protein_a, protein_b, sequence_a, sequence_b, output
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = join(splits_dir, f"{name}.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            splits[name] = df
            logger.info(f"  Loaded {name} split: {len(df)} pairs")
        else:
            logger.warning(f"  Split not found: {path}")
    return splits


def collect_unique_protein_ids(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return the set of all unique protein IDs across all splits."""
    ids = set()
    for df in splits.values():
        ids.update(df["protein_a"].astype(str))
        ids.update(df["protein_b"].astype(str))
    return ids


def collect_unique_proteins(splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Build {protein_id: sequence} from all splits.

    Uses sequence_a / sequence_b columns.  If a protein appears
    multiple times the first occurrence is kept.
    """
    proteins: Dict[str, str] = {}
    for df in splits.values():
        for _, row in df.iterrows():
            pid_a = str(row["protein_a"])
            pid_b = str(row["protein_b"])
            if pid_a not in proteins and "sequence_a" in row:
                proteins[pid_a] = str(row["sequence_a"])
            if pid_b not in proteins and "sequence_b" in row:
                proteins[pid_b] = str(row["sequence_b"])
    return proteins


# ============================================================
# Embedding Pre-loading from Disk
# ============================================================
def preload_foldvision_embeddings(
    protein_ids,
    emb_dir: str,
    n_runs: int,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Loads pre-computed FoldVision test-time augmentation runs.
    Expects files named embeddings_run00.npz, embeddings_run01.npz, etc.
    Returns:
        embeddings: Dict[str, torch.Tensor] of shape (n_runs, emb_dim)
        missing: List[str]
    """
    import numpy as np
    embeddings = {}
    missing = []
    
    run_data = []
    for r in range(n_runs):
        path = os.path.join(emb_dir, f"embeddings_run{r:02d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"FoldVision run file missing: {path}")
        
        data = np.load(path, allow_pickle=True)
        pids = data["protein_ids"]
        embs = data["embeddings"]
        pid_to_emb = {str(k): torch.from_numpy(v).float() for k, v in zip(pids, embs)}
        run_data.append(pid_to_emb)
        
    for pid in protein_ids:
        if all(pid in rd for rd in run_data):
            stacked = torch.stack([rd[pid] for rd in run_data], dim=0) # (n_runs, dim)
            embeddings[pid] = stacked
        else:
            missing.append(pid)
            
    return embeddings, missing

def preload_embeddings(
    protein_ids,
    emb_dir: str,
    emb_suffix: str,
    embedding_dim: int,
    emb_type: str = "per_token",
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Load pre-extracted .pt embeddings from *emb_dir*.

    Per-token tensors (seq_len, dim) are mean-pooled to (dim,).

    Returns:
        embeddings  – {protein_id: Tensor(embedding_dim,)}
        missing     – list of IDs with no file on disk
    """
    embeddings: Dict[str, torch.Tensor] = {}
    missing: List[str] = []

    for pid in protein_ids:
        emb_path = join(emb_dir, f"{pid}{emb_suffix}")
        if not os.path.exists(emb_path):
            missing.append(pid)
            continue

        emb = torch.load(emb_path, map_location="cpu", weights_only=True)

        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        elif emb.dim() != 1:
            logger.warning(f"Unexpected shape {emb.shape} for {pid} — skipping")
            missing.append(pid)
            continue

        if emb.shape[0] != embedding_dim:
            logger.warning(
                f"Dim mismatch for {pid}: expected {embedding_dim}, got {emb.shape[0]} — skipping"
            )
            missing.append(pid)
            continue

        embeddings[pid] = emb

    return embeddings, missing


# ============================================================
# Save Embeddings to Disk
# ============================================================
def save_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_dir: str,
    model_name: str,
) -> None:
    """Save each embedding as  {protein_id}_{model_name}_mean.pt  inside *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    for pid, emb in embeddings.items():
        path = join(output_dir, f"{pid}_{model_name}_mean.pt")
        torch.save(emb, path)
    logger.info(f"  Saved {len(embeddings)} embeddings to {output_dir}")


# ============================================================
# FASTA Parsing  (for extract_embeddings.py)
# ============================================================
def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """Parse a FASTA file into {header: sequence}."""
    proteins: Dict[str, str] = {}
    current_header = None
    current_seq: list = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    proteins[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            proteins[current_header] = "".join(current_seq)

    return proteins


# ============================================================
# PyTorch Dataset for PPI Pairs
# ============================================================
class PPIPairDataset(Dataset):
    """
    Dataset for PPI binary classification using pre-loaded embeddings.

    Expects a TSV with columns: protein_a, protein_b, ..., output
    and a pre-loaded embeddings dictionary.
    """

    def __init__(self, split_tsv: str, embeddings: Dict[str, torch.Tensor],
                 split_name: str = "unknown"):
        df = pd.read_csv(split_tsv, sep="\t")

        self.pairs: List[Tuple[str, str, int]] = []
        self.split_name = split_name
        skipped = 0
        for _, row in df.iterrows():
            pid_a = str(row["protein_a"])
            pid_b = str(row["protein_b"])
            if pid_a in embeddings and pid_b in embeddings:
                self.pairs.append((pid_a, pid_b, int(row["output"])))
            else:
                skipped += 1

        self.embeddings = embeddings

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
        
        emb_a = self.embeddings[pid_a]
        emb_b = self.embeddings[pid_b]
        
        if emb_a.ndim == 2 and self.split_name == "train":
            run_idx_a = torch.randint(0, emb_a.shape[0], (1,)).item()
            emb_a = emb_a[run_idx_a]
        if emb_b.ndim == 2 and self.split_name == "train":
            run_idx_b = torch.randint(0, emb_b.shape[0], (1,)).item()
            emb_b = emb_b[run_idx_b]
            
        return (
            emb_a,
            emb_b,
            torch.tensor(target, dtype=torch.float32),
            pid_a,
            pid_b,
        )

    def get_protein_ids(self):
        """Return (pids_a, pids_b, targets) lists."""
        pids_a = [p[0] for p in self.pairs]
        pids_b = [p[1] for p in self.pairs]
        targets = [p[2] for p in self.pairs]
        return pids_a, pids_b, targets


# ============================================================
# Collate
# ============================================================
def ppi_collate(batch):
    """Custom collate for PPI pairs (handles string protein IDs)."""
    emb_a = torch.stack([b[0] for b in batch])
    emb_b = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    pids_a = [b[3] for b in batch]
    pids_b = [b[4] for b in batch]
    return emb_a, emb_b, targets, pids_a, pids_b


# ################################################################
#
#  CAFA5 GO Term Prediction — Data Utilities
#
# ################################################################

# ============================================================
# CAFA Split Loading
# ============================================================
def load_cafa_splits(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load train.tsv / val.tsv / test.tsv for CAFA5.

    Expected columns: protein_id, sequence, output
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = join(splits_dir, f"{name}.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            splits[name] = df
            logger.info(f"  Loaded {name} split: {len(df)} proteins")
        else:
            logger.warning(f"  Split not found: {path}")
    return splits


def collect_cafa_protein_ids(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return all unique protein IDs across CAFA splits."""
    ids = set()
    for df in splits.values():
        ids.update(df["protein_id"].astype(str))
    return ids


def collect_cafa_sequences(splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Build {protein_id: sequence} from CAFA splits."""
    proteins: Dict[str, str] = {}
    for df in splits.values():
        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            if pid not in proteins and "sequence" in row:
                proteins[pid] = str(row["sequence"])
    return proteins


# ============================================================
# GO Vocabulary & Label Encoding
# ============================================================
def build_go_vocabulary(splits_dir: str) -> Dict[str, List[str]]:
    """
    Scan all CAFA splits and collect every unique GO term per namespace.

    Returns {namespace: sorted_list_of_terms} for BPO, CCO, MFO.
    """
    go_terms: Dict[str, set] = {"BPO": set(), "CCO": set(), "MFO": set()}

    for split_file in ("train.tsv", "val.tsv", "test.tsv"):
        path = join(splits_dir, split_file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        for target_str in df["output"]:
            target = ast.literal_eval(target_str)
            for ns in ("BPO", "CCO", "MFO"):
                if ns in target:
                    go_terms[ns].update(target[ns])

    go_vocab = {ns: sorted(terms) for ns, terms in go_terms.items()}
    return go_vocab


def build_go_to_index(go_vocab: Dict[str, List[str]]):
    """
    Create term → index mappings.

    Returns:
        ns_to_idx  – {namespace: {term: local_index}}
        flat_to_idx – {term: global_flat_index}  (BPO | CCO | MFO concatenated)
    """
    ns_to_idx: Dict[str, Dict[str, int]] = {}
    for ns in ("BPO", "CCO", "MFO"):
        ns_to_idx[ns] = {term: i for i, term in enumerate(go_vocab[ns])}

    flat_to_idx: Dict[str, int] = {}
    offset = 0
    for ns in ("BPO", "CCO", "MFO"):
        for term in go_vocab[ns]:
            flat_to_idx[term] = offset
            offset += 1

    return ns_to_idx, flat_to_idx


def encode_go_labels(target_dict, go_vocab, ns_to_idx, flat_to_idx, mode="single"):
    """
    Encode a protein's GO annotations into binary vector(s).

    mode='single': flat vector of len(BPO)+len(CCO)+len(MFO)
    mode='three':  dict {ns: vector} per namespace
    """
    if mode == "single":
        total_dim = sum(len(go_vocab[ns]) for ns in ("BPO", "CCO", "MFO"))
        label = np.zeros(total_dim, dtype=np.float32)
        for ns in ("BPO", "CCO", "MFO"):
            if ns in target_dict:
                for term in target_dict[ns]:
                    if term in flat_to_idx:
                        label[flat_to_idx[term]] = 1.0
        return label
    elif mode == "three":
        labels = {}
        for ns in ("BPO", "CCO", "MFO"):
            vec = np.zeros(len(go_vocab[ns]), dtype=np.float32)
            if ns in target_dict:
                for term in target_dict[ns]:
                    if term in ns_to_idx[ns]:
                        vec[ns_to_idx[ns][term]] = 1.0
            labels[ns] = vec
        return labels
    else:
        raise ValueError(f"mode must be 'single' or 'three', got '{mode}'")


# ============================================================
# CAFA Dataset — pre-loaded embeddings in memory
# ============================================================
class CAFADataset(Dataset):
    """
    Dataset for CAFA5 GO term prediction using in-memory embeddings.

    Works with both Mode 1 (pre-extracted from disk → dict) and
    Mode 2 (on-the-fly extracted → dict).
    """

    def __init__(
        self,
        split_tsv: str,
        embeddings: Dict[str, torch.Tensor],
        go_vocab: Dict[str, List[str]],
        ns_to_idx: Dict[str, Dict[str, int]],
        flat_to_idx: Dict[str, int],
        label_mode: str = "single",
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")

        self.protein_ids: List[str] = []
        self.targets: List[dict] = []
        self.split_name = split_name
        skipped = 0

        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            if pid in embeddings:
                self.protein_ids.append(pid)
                self.targets.append(ast.literal_eval(row["output"]))
            else:
                skipped += 1

        self.embeddings = embeddings
        self.go_vocab = go_vocab
        self.ns_to_idx = ns_to_idx
        self.flat_to_idx = flat_to_idx
        self.label_mode = label_mode

        logger.info(
            f"  {split_name}: {len(self.protein_ids)}/{len(df)} proteins loaded "
            f"({skipped} skipped), label_mode={label_mode}"
        )

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        emb = self.embeddings[pid]
        
        if emb.ndim == 2 and self.split_name == "train":
            run_idx = torch.randint(0, emb.shape[0], (1,)).item()
            emb = emb[run_idx]

        label = encode_go_labels(
            self.targets[idx], self.go_vocab,
            self.ns_to_idx, self.flat_to_idx,
            mode=self.label_mode,
        )

        if self.label_mode == "single":
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = {ns: torch.tensor(vec, dtype=torch.float32) for ns, vec in label.items()}

        return emb, label, pid


# ============================================================
# CAFA Collate
# ============================================================
def cafa_collate_single(batch):
    """Default collate for single-head CAFA (emb, label_tensor, pid)."""
    embs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pids = [b[2] for b in batch]
    return embs, labels, pids


def cafa_collate_three(batch):
    """Collate for three-head CAFA (emb, {ns: tensor}, pid)."""
    embs = torch.stack([b[0] for b in batch])
    pids = [b[2] for b in batch]
    labels = {ns: torch.stack([b[1][ns] for b in batch]) for ns in ("BPO", "CCO", "MFO")}
    return embs, labels, pids


# ============================================================
# CAFA Pos-Weight (per GO term)
# ============================================================
def compute_cafa_pos_weight(
    dataset: "CAFADataset",
    go_vocab: Dict[str, List[str]],
    mode: str = "single",
    method: str = "linear",
    cap: float = 50.0,
):
    """
    Compute per-GO-term positive class weight from training labels.

    method: 'linear' (neg/pos) or 'sqrt' (sqrt(neg/pos))
    cap:    maximum weight to prevent instability

    Returns:
        single mode → Tensor(total_terms,)
        three mode  → {ns: Tensor(ns_terms,)}
    """
    n = len(dataset)
    logger.info(f"  Computing CAFA pos_weight ({method}, cap={cap}) from {n} proteins...")

    if mode == "single":
        total_dim = sum(len(go_vocab[ns]) for ns in ("BPO", "CCO", "MFO"))
        pos_counts = np.zeros(total_dim, dtype=np.float64)
        for i in range(n):
            label = encode_go_labels(
                dataset.targets[i], go_vocab,
                dataset.ns_to_idx, dataset.flat_to_idx,
                mode="single",
            )
            pos_counts += label

        neg_counts = n - pos_counts
        ratio = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
        if method == "sqrt":
            ratio = np.sqrt(ratio)
        ratio = np.clip(ratio, a_min=1.0, a_max=cap)
        pw = torch.tensor(ratio, dtype=torch.float32)
        logger.info(f"    pos_weight range: [{pw.min().item():.2f}, {pw.max().item():.2f}], "
                     f"mean: {pw.mean().item():.2f}")
        return pw

    elif mode == "three":
        pw_dict = {}
        for ns in ("BPO", "CCO", "MFO"):
            ns_dim = len(go_vocab[ns])
            pos_counts = np.zeros(ns_dim, dtype=np.float64)
            idx_map = dataset.ns_to_idx[ns]
            for i in range(n):
                target = dataset.targets[i]
                if ns in target:
                    for term in target[ns]:
                        if term in idx_map:
                            pos_counts[idx_map[term]] += 1.0

            neg_counts = n - pos_counts
            ratio = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
            if method == "sqrt":
                ratio = np.sqrt(ratio)
            ratio = np.clip(ratio, a_min=1.0, a_max=cap)
            pw_dict[ns] = torch.tensor(ratio, dtype=torch.float32)
            logger.info(f"    {ns} pos_weight range: [{pw_dict[ns].min().item():.2f}, "
                         f"{pw_dict[ns].max().item():.2f}]")
        return pw_dict
    else:
        raise ValueError(f"mode must be 'single' or 'three', got '{mode}'")


# ################################################################
#
#  SPOT Protein–Small Molecule Interaction — Data Utilities
#
# ################################################################

# ============================================================
# SPOT Split Loading
# ============================================================
def load_spot_splits(
    splits_dir: str,
    labels_dict: Dict[str, int],
) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    Load SPOT splits from {train,val,test}_names.npy files.

    Each *_names.npy contains an array of "UID_MID" strings.
    Labels come from *labels_dict* (``{"UID_MID": 0|1}``).

    Returns ``{"train": [(uid, mid, target), ...], ...}``.
    """
    mapping = {"train": "train_names.npy", "val": "val_names.npy", "test": "test_names.npy"}
    splits: Dict[str, List[Tuple[str, str, int]]] = {}
    for name, fname in mapping.items():
        path = join(splits_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"  Split file not found: {path}")
            continue
        names = np.load(path, allow_pickle=True)
        pairs: List[Tuple[str, str, int]] = []
        missing_labels = 0
        malformed_keys = 0
        for key in names:
            key = str(key)
            if "_" not in key:
                malformed_keys += 1
                continue
            if key not in labels_dict:
                missing_labels += 1
                continue
            uid, mid = key.rsplit("_", 1)
            target = int(labels_dict[key])
            pairs.append((uid, mid, target))
        splits[name] = pairs
        logger.info(f"  Loaded {name} split: {len(pairs)} pairs")
        if missing_labels > 0:
            logger.warning(
                f"  {name}: {missing_labels} pair keys missing from labels_dict and were skipped"
            )
        if malformed_keys > 0:
            logger.warning(
                f"  {name}: {malformed_keys} malformed keys in {fname} and were skipped"
            )
    return splits


def collect_spot_protein_ids(splits: Dict[str, List[Tuple[str, str, int]]]) -> Set[str]:
    """Return all unique protein UIDs across SPOT splits."""
    ids: Set[str] = set()
    for pairs in splits.values():
        ids.update(uid for uid, _, _ in pairs)
    return ids


def collect_spot_mids(splits: Dict[str, List[Tuple[str, str, int]]]) -> Set[str]:
    """Return all unique molecule IDs across SPOT splits."""
    mids: Set[str] = set()
    for pairs in splits.values():
        mids.update(mid for _, mid, _ in pairs)
    return mids


def load_spot_sequences(splits_dir: str) -> Dict[str, str]:
    """
    Load {UID: amino-acid-sequence} from SPOT sequence maps.

    Preferred file: UID_to_Seq.npy ({UID: sequence}).
    Fallback file:  Seq_to_UID.npy ({sequence: UID}) — inverted on load.
    """
    uid_to_seq_path = join(splits_dir, "UID_to_Seq.npy")
    seq_to_uid_path = join(splits_dir, "Seq_to_UID.npy")

    if os.path.exists(uid_to_seq_path):
        raw = np.load(uid_to_seq_path, allow_pickle=True).item()
        seqs = {str(k): str(v) for k, v in raw.items()}
        logger.info(f"  Loaded {len(seqs)} protein sequences from {uid_to_seq_path}")
        return seqs

    if os.path.exists(seq_to_uid_path):
        raw = np.load(seq_to_uid_path, allow_pickle=True).item()
        seqs: Dict[str, str] = {}
        conflicts = 0
        for seq, uid in raw.items():
            uid = str(uid)
            seq = str(seq)
            if uid in seqs and seqs[uid] != seq:
                conflicts += 1
                continue
            seqs[uid] = seq
        logger.info(
            f"  Loaded {len(seqs)} protein sequences by inverting {seq_to_uid_path}"
        )
        if conflicts > 0:
            logger.warning(
                f"  Seq_to_UID conflicts for {conflicts} UIDs; kept first sequence per UID"
            )
        return seqs

    raise FileNotFoundError(
        f"No sequence map found in {splits_dir}. Expected UID_to_Seq.npy or Seq_to_UID.npy"
    )


# ============================================================
# Molecule Embedding Loading
# ============================================================
def load_mol_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed molecule embeddings from .npy or .pt file.

    Expected format: dict mapping molecule ID (e.g. 'M0') → vector.
    """
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu", weights_only=True)
    elif path.endswith(".npy") or path.endswith(".npz"):
        # Compat: numpy 2.x renamed `numpy.core` -> `numpy._core`. Files pickled
        # by numpy >= 2.0 reference `numpy._core.*`; alias it back so a numpy 1.x
        # interpreter (as in the esm_gearnet env) can still unpickle them.
        if "numpy._core" not in sys.modules:
            import numpy.core
            sys.modules["numpy._core"] = numpy.core
            for _sub in ("multiarray", "umath", "numeric", "_multiarray_umath"):
                _m = getattr(numpy.core, _sub, None)
                if _m is not None:
                    sys.modules[f"numpy._core.{_sub}"] = _m
        data = np.load(path, allow_pickle=True)
        if hasattr(data, "item"):
            data = data.item()
    else:
        raise ValueError(f"Unsupported molecule embedding format: {path}")

    embeddings: Dict[str, torch.Tensor] = {}
    for key, vec in data.items():
        if isinstance(vec, np.ndarray):
            vec = torch.tensor(vec, dtype=torch.float32)
        elif not isinstance(vec, torch.Tensor):
            vec = torch.tensor(np.array(vec), dtype=torch.float32)
        embeddings[str(key)] = vec

    logger.info(f"  Loaded {len(embeddings)} molecule embeddings from {path}")
    return embeddings


# ============================================================
# SPOT Dataset
# ============================================================
class SPOTPairDataset(Dataset):
    """
    Dataset for SPOT binary protein–molecule interaction classification.

    Pairs are (UID, MID, target).  Protein embeddings are keyed by UID,
    molecule embeddings are keyed by MID.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        protein_embeddings: Dict[str, torch.Tensor],
        mol_features: Dict[str, torch.Tensor],
        split_name: str = "unknown",
    ):
        self.pairs: List[Tuple[str, str, int]] = []
        skipped = 0
        total = len(pairs)
        for uid, mid, target in pairs:
            if uid in protein_embeddings and mid in mol_features:
                self.pairs.append((uid, mid, target))
            else:
                skipped += 1

        self.protein_embeddings = protein_embeddings
        self.mol_features = mol_features

        unique_prots = set(p[0] for p in self.pairs)
        unique_mols = set(p[1] for p in self.pairs)

        logger.info(
            f"  {split_name}: {len(self.pairs)}/{total} pairs loaded "
            f"({skipped} skipped), {len(unique_prots)} proteins, "
            f"{len(unique_mols)} molecules"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid, mid, target = self.pairs[idx]
        return (
            self.protein_embeddings[uid],
            self.mol_features[mid],
            torch.tensor(target, dtype=torch.float32),
            uid,
            mid,
        )

    def get_ids(self):
        """Return (protein_ids, mol_ids, targets) lists."""
        pids = [p[0] for p in self.pairs]
        mids = [p[1] for p in self.pairs]
        targets = [p[2] for p in self.pairs]
        return pids, mids, targets


# ============================================================
# SPOT Collate
# ============================================================
def spot_collate(batch):
    """Custom collate for SPOT protein–molecule pairs."""
    prot_embs = torch.stack([b[0] for b in batch])
    mol_feats = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    pids = [b[3] for b in batch]
    mids = [b[4] for b in batch]
    return prot_embs, mol_feats, targets, pids, mids


# ============================================================
# SPOT v2 — TSV-based pipeline (uniform with DAVIS)
# ============================================================
def load_spot_splits_v2(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load train.tsv / val.tsv / test.tsv for SPOT (v2 format).

    Expected columns: protein_id, sequence, SMILES, output
    Returns: {split_name: DataFrame} for each file that exists.
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = join(splits_dir, f"{name}.tsv")
        if os.path.exists(path):
            splits[name] = pd.read_csv(path, sep="\t")
            logger.info(f"  Loaded {name} split: {len(splits[name])} pairs")
        else:
            logger.warning(f"  Split not found: {path}")
    return splits


def collect_spot_protein_ids_v2(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return all unique protein_ids across SPOT v2 splits."""
    ids: Set[str] = set()
    for df in splits.values():
        ids.update(df["protein_id"].astype(str))
    return ids


def collect_spot_sequences_v2(splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Build {protein_id: sequence} from SPOT v2 splits.

    First occurrence wins when a protein appears in multiple splits.
    """
    seq_map: Dict[str, str] = {}
    for df in splits.values():
        for pid, seq in zip(df["protein_id"].astype(str), df["sequence"].astype(str)):
            if pid not in seq_map:
                seq_map[pid] = seq
    return seq_map


def collect_spot_smiles(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return all unique SMILES strings across SPOT v2 splits."""
    smiles: Set[str] = set()
    for df in splits.values():
        smiles.update(df["SMILES"].astype(str))
    return smiles


class SPOTPairDatasetV2(Dataset):
    """
    SPOT binary classification dataset — v2 TSV-based pipeline.

    Mirrors DAVISDataset: protein embeddings keyed by protein_id,
    molecule embeddings keyed by SMILES string.
    """

    def __init__(
        self,
        split_tsv: str,
        protein_embeddings: Dict[str, torch.Tensor],
        mol_embeddings: Dict[str, torch.Tensor],
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")
        self.pairs: List[Tuple[str, str, int]] = []
        self.split_name = split_name
        skipped_prot = 0
        skipped_mol = 0
        total = len(df)

        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            smi = str(row["SMILES"])
            if pid not in protein_embeddings:
                skipped_prot += 1
                continue
            if smi not in mol_embeddings:
                skipped_mol += 1
                continue
            self.pairs.append((pid, smi, int(row["output"])))

        self.protein_embeddings = protein_embeddings
        self.mol_embeddings = mol_embeddings

        unique_prots = set(p[0] for p in self.pairs)
        unique_mols = set(p[1] for p in self.pairs)

        logger.info(
            f"  {split_name}: {len(self.pairs)}/{total} pairs loaded "
            f"(skipped {skipped_prot} missing prot, {skipped_mol} missing mol), "
            f"{len(unique_prots)} proteins, {len(unique_mols)} molecules"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid, smi, target = self.pairs[idx]
        
        prot_emb = self.protein_embeddings[pid]
        if prot_emb.ndim == 2 and self.split_name == "train":
            run_idx = torch.randint(0, prot_emb.shape[0], (1,)).item()
            prot_emb = prot_emb[run_idx]
            
        return (
            prot_emb,
            self.mol_embeddings[smi],
            torch.tensor(target, dtype=torch.float32),
            pid,
            smi,
        )

    def get_ids(self):
        """Return (protein_ids, smiles_list, targets) lists."""
        pids = [p[0] for p in self.pairs]
        smiles = [p[1] for p in self.pairs]
        targets = [p[2] for p in self.pairs]
        return pids, smiles, targets


def spot_collate_v2(batch):
    """Custom collate for SPOT v2 protein–molecule pairs (SMILES-keyed)."""
    prot_embs = torch.stack([b[0] for b in batch])
    mol_embs = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    pids = [b[3] for b in batch]
    smiles = [b[4] for b in batch]
    return prot_embs, mol_embs, targets, pids, smiles


# ################################################################
#
#  GRB2 Binding / Abundance — Data Utilities
#
# ################################################################

# ============================================================
# GRB2 Split Loading
# ============================================================
def load_grb2_splits(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load train.tsv / val.tsv / test.tsv for a GRB2 DMS task.

    Expected columns: variant, variant_pdb_id, sequence, output
    Returns: {split_name: DataFrame} for each file that exists.
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = join(splits_dir, f"{name}.tsv")
        if os.path.exists(path):
            splits[name] = pd.read_csv(path, sep="\t")
    return splits


def collect_grb2_protein_ids(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return all unique variant names (used as protein IDs) across splits."""
    ids: Set[str] = set()
    for df in splits.values():
        ids.update(df["variant"].astype(str).tolist())
    return ids


def collect_grb2_sequences(splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Build {variant: sequence} from GRB2 splits.

    The variant name (e.g. "D28Y,N49Y") is used as the protein ID.
    First occurrence wins when a variant appears in multiple splits.
    """
    seq_map: Dict[str, str] = {}
    for df in splits.values():
        for vid, seq in zip(df["variant"].astype(str), df["sequence"].astype(str)):
            if vid not in seq_map:
                seq_map[vid] = seq
    return seq_map


# ============================================================
# GRB2 Dataset
# ============================================================
class GRB2Dataset(Dataset):
    """
    Dataset for GRB2 regression using pre-loaded, mean-pooled embeddings.

    Each sample corresponds to one variant (e.g. "D28Y,N49Y").  The
    variant name doubles as the protein ID for embedding lookup.

    Returns:
        embedding  – Tensor(embedding_dim,)
        score      – Tensor(scalar float32)
        variant    – str  (variant name)
    """

    def __init__(
        self,
        split_tsv: str,
        embeddings: Dict[str, torch.Tensor],
        split_name: str = "unknown",
    ):
        self.pairs: List[Tuple[torch.Tensor, float, str]] = []
        self.split_name = split_name
        df = pd.read_csv(split_tsv, sep="\t")
        missing = 0
        for _, row in df.iterrows():
            vid = str(row["variant"])
            if vid not in embeddings:
                missing += 1
                continue
            self.pairs.append((embeddings[vid], float(row["output"]), vid))
        if missing:
            logger.warning(
                f"  [{split_name}] {missing}/{len(df)} variants "
                "missing embeddings — skipped"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        emb, score, vid = self.pairs[idx]
        
        if emb.ndim == 2 and self.split_name == "train":
            run_idx = torch.randint(0, emb.shape[0], (1,)).item()
            emb = emb[run_idx]
            
        return emb, torch.tensor(score, dtype=torch.float32), vid

    def get_variant_ids(self) -> List[str]:
        """Return variant names in dataset order."""
        return [p[2] for p in self.pairs]


# ============================================================
# GRB2 Collate
# ============================================================
def grb2_collate(batch):
    """Custom collate for GRB2 single-sequence regression."""
    embs = torch.stack([b[0] for b in batch])
    scores = torch.stack([b[1] for b in batch])
    variants = [b[2] for b in batch]
    return embs, scores, variants


# ################################################################
#
#  DAVIS Drug–Target Affinity — Data Utilities
#
# ################################################################

# ============================================================
# DAVIS Split Loading
# ============================================================
def load_davis_splits(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load train.tsv / val.tsv / test.tsv for a DAVIS DTA split.

    Expected columns: protein_id, sequence, SMILES, target, output
    Returns: {split_name: DataFrame} for each file that exists.
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = join(splits_dir, f"{name}.tsv")
        if os.path.exists(path):
            splits[name] = pd.read_csv(path, sep="\t")
            logger.info(f"  Loaded {name} split: {len(splits[name])} pairs")
        else:
            logger.warning(f"  Split not found: {path}")
    return splits


def collect_davis_protein_ids(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return all unique protein_ids across DAVIS splits."""
    ids: Set[str] = set()
    for df in splits.values():
        ids.update(df["protein_id"].astype(str))
    return ids


def collect_davis_sequences(splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Build {protein_id: sequence} from DAVIS splits.

    First occurrence wins when a protein appears in multiple splits.
    """
    seq_map: Dict[str, str] = {}
    for df in splits.values():
        for pid, seq in zip(df["protein_id"].astype(str), df["sequence"].astype(str)):
            if pid not in seq_map:
                seq_map[pid] = seq
    return seq_map


def collect_davis_smiles(splits: Dict[str, pd.DataFrame]) -> Set[str]:
    """Return all unique SMILES strings across DAVIS splits."""
    smiles: Set[str] = set()
    for df in splits.values():
        smiles.update(df["SMILES"].astype(str))
    return smiles


# ============================================================
# DAVIS Dataset
# ============================================================
class DAVISDataset(Dataset):
    """
    Dataset for DAVIS drug–target affinity regression.

    Each sample is a (protein, drug) pair with a continuous pKd score.
    Protein embeddings are keyed by protein_id, molecule embeddings
    are keyed by SMILES string.

    Returns:
        prot_emb   – Tensor(protein_embedding_dim,)
        mol_emb    – Tensor(mol_embedding_dim,)
        score      – Tensor(scalar float32)  pKd value
        protein_id – str
        smiles     – str
    """

    def __init__(
        self,
        split_tsv: str,
        protein_embeddings: Dict[str, torch.Tensor],
        mol_embeddings: Dict[str, torch.Tensor],
        split_name: str = "unknown",
    ):
        df = pd.read_csv(split_tsv, sep="\t")
        self.pairs: List[Tuple[str, str, float]] = []
        self.split_name = split_name
        skipped_prot = 0
        skipped_mol = 0
        total = len(df)

        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            smi = str(row["SMILES"])
            if pid not in protein_embeddings:
                skipped_prot += 1
                continue
            if smi not in mol_embeddings:
                skipped_mol += 1
                continue
            self.pairs.append((pid, smi, float(row["output"])))

        self.protein_embeddings = protein_embeddings
        self.mol_embeddings = mol_embeddings

        unique_prots = set(p[0] for p in self.pairs)
        unique_mols = set(p[1] for p in self.pairs)

        logger.info(
            f"  {split_name}: {len(self.pairs)}/{total} pairs loaded "
            f"(skipped {skipped_prot} missing prot, {skipped_mol} missing mol), "
            f"{len(unique_prots)} proteins, {len(unique_mols)} drugs"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid, smi, score = self.pairs[idx]
        
        prot_emb = self.protein_embeddings[pid]
        if prot_emb.ndim == 2 and getattr(self, "split_name", "unknown") == "train":
            run_idx = torch.randint(0, prot_emb.shape[0], (1,)).item()
            prot_emb = prot_emb[run_idx]
            
        return (
            prot_emb,
            self.mol_embeddings[smi],
            torch.tensor(score, dtype=torch.float32),
            pid,
            smi,
        )

    def get_ids(self):
        """Return (protein_ids, smiles_list, scores) lists."""
        pids = [p[0] for p in self.pairs]
        smiles = [p[1] for p in self.pairs]
        scores = [p[2] for p in self.pairs]
        return pids, smiles, scores


# ============================================================
# DAVIS Collate
# ============================================================
def davis_collate(batch):
    """Custom collate for DAVIS drug–target affinity pairs."""
    prot_embs = torch.stack([b[0] for b in batch])
    mol_embs = torch.stack([b[1] for b in batch])
    scores = torch.stack([b[2] for b in batch])
    pids = [b[3] for b in batch]
    smiles = [b[4] for b in batch]
    return prot_embs, mol_embs, scores, pids, smiles
