# ProteinArena

We introduce **ProteinArena**, an interactive benchmark suite for evaluating open-weight protein encoders under a standardized protocol. ProteinArena evaluates protein encoders across six tasks spanning protein function prediction, protein–small-molecule interaction, protein–protein interaction, and variant-effect prediction. Each encoder is assessed in both **frozen** and **end-to-end fine-tuning** settings.

Beyond aggregate test metrics, ProteinArena provides split-aware, subset-aware, and cost-aware analyses through an interactive website: [www.ProteinArena.org](http://www.ProteinArena.org). It is designed as a living benchmark: datasets, splits, and code are publicly accessible, benchmark results are easy to reproduce, and newly published open-weight encoders can be easily added.

ProteinArena operates through two parallel modules:

- **PredictionModule** — frozen-encoder evaluation. The PLM is used as a fixed feature extractor; only a lightweight prediction head is trained on top of its embeddings. Cheap, fast, and consistent across models.
- **FineTuneModule** — end-to-end fine-tuning. The PLM backbone is updated jointly with the prediction head under differential learning rates.

This README walks through a complete run on a single task (**PRING**, protein–protein interaction) using **ESM-2 150M** as the PLM in both settings. Every other task follows the same pattern with a different `predict_*.py` / `finetune_*.py` script.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Data](#data)
3. [Environment](#environment)
4. [Run quickstart](#run-quickstart)
   - [Frozen embeddings (PredictionModule)](#frozen-embeddings-predictionmodule)
   - [End-to-end fine-tuning (FineTuneModule)](#end-to-end-fine-tuning-finetunemodule)
5. [Essential Flags](#essential-flags)
6. [Other Tasks](#other-tasks)
7. [Structure-Based Models (FoldVision, ESM-IF, ESM-GearNet)](#structure-based-models-foldvision-esm-if-esm-gearnet)
8. [Submitting a New Model](#submitting-a-new-model)

---

## Repository Layout

```
Code/
├── README.md                          # this file
│
├── PredictionModule/
│   ├── predict_pring.py               # PRING — Protein–Protein Interaction (binary)
│   ├── predict_cafa.py                # CAFA5 — GO Term Prediction (multi-label)
│   ├── predict_davis.py               # DAVIS — Drug–Target Affinity (regression)
│   ├── predict_spot.py                # SPOT — Protein–Small Molecule (binary)
│   ├── predict_grb2binding.py         # GRB2 Binding (regression)
│   ├── predict_grb2abundance.py       # GRB2 Abundance (regression)
│   │
│   ├── util_model.py                  # MODEL_REGISTRY + family-specific loaders / extractors
│   ├── util_data.py                   # split loaders, Datasets, embedding I/O
│   ├── util_helper.py                 # DDP, logging, metrics, checkpointing
│   └── lr_selector.py                 # dynamic LR sweep (Rules 4a / 4b)
│
└── FineTuneModule/
    ├── finetune_pring.py              # PRING — end-to-end PLM + head
    ├── finetune_cafa.py               # CAFA5 — end-to-end PLM + multi-label head
    ├── finetune_davis.py              # DAVIS — end-to-end PLM + DTA head
    ├── finetune_spot.py               # SPOT — end-to-end PLM + binary head
    ├── finetune_grb2binding.py        # GRB2 Binding — end-to-end PLM + regression head
    ├── finetune_grb2abundance.py      # GRB2 Abundance — end-to-end PLM + regression head
    │
    ├── util_finetune.py               # sequence Datasets, param groups, layer freezing, sweep glue
    └── util_finetune_model.py         # per-family differentiable forward functions
```

The FineTuneModule scripts reuse `MODEL_REGISTRY`, the data utilities, the helper module, and `lr_selector.py` from PredictionModule via relative imports — so the two modules share their model registry, split loaders, metrics, and sweep algorithm.

---

## Data

The dataset splits and pre-computed MolFormer-XL drug embeddings are hosted on Zenodo:

> **Zenodo download:** [Zenedo | ProteinArena](https://zenodo.org/records/19918666)

After extracting the archive you should have something like:

```
ProteinArena_Data/
├── PRING/Splits/
│   ├── train.tsv         # protein_a, protein_b, sequence_a, sequence_b, output
│   ├── val.tsv
│   └── test.tsv
├── DAVIS/
│   ├── Splits/
│   │   ├── random_split/{train,val,test}.tsv      # protein_id, sequence, SMILES, target, output
│   │   ├── cold_drug_split/...
│   │   ├── cold_target_split/...
│   │   └── cold_drug_cold_target_split/...
│   └── SMILES_MolFormer_XL.npy                     # SMILES → 768-d MolFormer embedding
├── SPOT/Splits/{train,val,test}.tsv  +  SMILES_MolFormer_XL.npy
├── CAFA5/Splits/{train,val,test}.tsv
├── GRB2_Binding/Splits/{train,val,test}.tsv
└── GRB2_Abundance/Splits/{train,val,test}.tsv
```

Both modules consume the same splits — there is no separate "fine-tuning split".

---

## Environment

ProteinArena itself doesn't require any complex dependencies. Each model pulls in its own dependency (Fair-ESM, `transformers`, `torchdrug`, etc.). One should look at the installation requirements from the GitHub repository of the respective models only. For illustration, we use ESM-2; the official requirement for ESM-2 is:

```bash
pip install fair-esm
```

It is recommended to run the pipeline on GPU with PyTorch and CUDA and use `torchrun` for multi-GPU runs, whereas for a single-GPU run, just use `python`.

---

## Run quickstart

The two modules share the same flag conventions. Set these once for the rest of this section:

```bash
export PRING_SPLITS=/path/to/ProteinArena_Data/PRING/Splits_v2
export ESM2_WEIGHTS=/path/to/esm2_t30_150M_UR50D.pt    # local .pt; omit to auto-download
```

### Frozen embeddings (PredictionModule)

This single command runs an LR sweep, trains the lightweight head with the chosen LR, and then evaluates on `train` / `val` / `test` — all from one invocation. The protein encoder weights stay frozen throughout.

```bash
export OUT_DIR=/path/to/runs/esm2_150M_pring_pred
cd Code/PredictionModule

torchrun --nproc_per_node=4 predict_pring.py \
    --splits_dir   $PRING_SPLITS \
    --model_name   esm2_150M \
    --model_path   $ESM2_WEIGHTS \
    --arch_type    symmetric \
    --lr_dynsweep \
    --sweep_rule   4a \
    --num_epochs   200 \
    --batch_size   128 \
    --extraction_batch_size 32 \
    --early_stopping_patience 15 \
    --log_every_n_steps 50 \
    --seed         42 \
    --save_dir     $OUT_DIR
```

Drop `torchrun --nproc_per_node=N` for a single-GPU run.

> **Note:** `--lr_dynsweep` is single-process by design. The wrapper handles broadcasting the chosen LR to all DDP ranks before training begins, but the sweep itself runs on rank 0 only.

> **Why two sweep rules?** Rule 4a picks the LR that minimises the train/val gap; Rule 4b adds an α-weighted stability term. Use `4a` as the default; `4b` if val curves are noisy.

### End-to-end fine-tuning (FineTuneModule)

The FineTuneModule mirrors the PredictionModule but updates the PLM backbone alongside the prediction head. The same dynamic LR sweep is applied to **both** backbone and head as a single shared LR (manual differential control via `--backbone_lr` / `--head_lr` is also exposed). Because the backbone is now in the gradient graph, you typically want a smaller `--batch_size` paired with `--grad_accum_steps`, and `--use_amp` on CUDA:

```bash
export OUT_DIR=/path/to/runs/esm2_150M_pring_ft
cd Code/FineTuneModule

torchrun --nproc_per_node=4 finetune_pring.py \
    --splits_dir   $PRING_SPLITS \
    --model_name   esm2_150M \
    --model_path   $ESM2_WEIGHTS \
    --arch_type    symmetric \
    --lr_dynsweep \
    --sweep_rule   4a \
    --num_epochs   10 \
    --batch_size   8 \
    --grad_accum_steps 4 \
    --max_grad_norm 1.0 \
    --use_amp \
    --early_stopping_patience 5 \
    --log_every_n_steps 50 \
    --seed         42 \
    --save_dir     $OUT_DIR
```

Drop `torchrun --nproc_per_node=N` for a single-GPU run.

### Outputs in `--save_dir`

Both modules write a consistent set of artifacts:

| File | Contents |
|---|---|
| `<log_name>.log` | full run log |
| `<log_name>_best_checkpoint.pt` | best-val checkpoint + config<sup>†</sup> |
| `<log_name>_curves.npz` | per-epoch train/val loss |
| `<log_name>_<split>_predictions.tsv` (or `.npy` for CAFA) | per-sample predictions |
| `<log_name>_<split>_roc.npz` | ROC curves (binary tasks only) |
| `<log_name>_metrics.json` | summary metrics + run config |
| `<log_name>_dynsweep_results.json` (sweep runs only) | LR sweep table + Rule 4a / 4b winners |
| `<log_name>_dynsweep_table.csv` (sweep runs only) | same sweep table as CSV |

`<log_name>` defaults to a task-specific value: PredictionModule uses `pring_pred`, `davis_pred`, …; FineTuneModule uses `ft_pring`, `ft_davis`, …. Override with `--log_name`.

<sup>†</sup> **Checkpoint differences across modules.** The PredictionModule checkpoint stores a single `model_state_dict` (the head). The FineTuneModule checkpoint stores `backbone_state_dict` + `head_state_dict` separately, plus the per-group `backbone_lr` / `head_lr` used during training, so the two halves can be loaded independently.

You can also save embeddings as a side-effect of an on-the-fly PredictionModule run by adding `--save_embeddings_dir $EMB_CACHE` to any `predict_*.py` invocation. (FineTuneModule doesn't have an analogous flag — its forward pass is differentiable per batch and isn't designed to materialise a reusable embedding cache.)

---

## Essential Flags

Every script takes the same core set of flags. The "Module" column marks which scripts each flag applies to (`pred` = PredictionModule, `ft` = FineTuneModule, `both` = both). The ones you almost always need to set:

| Flag | Module | Required | What it does |
|---|---|---|---|
| `--splits_dir` | both | yes | directory with `train.tsv` / `val.tsv` / `test.tsv` |
| `--model_name` | both | yes | registered name in `MODEL_REGISTRY` (e.g. `esm2_150M`) |
| `--model_path` | both | yes | local path or HuggingFace ID for PLM weights |
| `--save_dir` | both | yes | output directory; created if missing |
| `--mol_emb_path` | both | yes for DAVIS / SPOT | path to `SMILES_MolFormer_XL.npy` |
| `--pdb_dir` | pred | yes for structure PLMs | directory of `{protein_id}.pdb` files |
| `--arch_type` | both | recommended | head architecture |
| `--batch_size` | both | recommended | per-GPU batch size (use a much smaller value for `ft` than `pred`) |
| `--extraction_batch_size` | pred | recommended | batch size for the on-the-fly PLM forward pass |
| `--num_epochs` | both | recommended | training epoch budget |
| `--early_stopping_patience` | both | recommended | stop if val loss does not improve for N epochs |
| `--seed` | both | recommended | reproducibility |
| `--backbone_lr` / `--head_lr` | ft | recommended | differential LRs (PLM vs head); overridden by sweep when `--lr_dynsweep` is set |
| `--freeze_layers` | ft | optional | freeze the first N transformer layers of the backbone |
| `--use_amp` | ft | optional | mixed-precision training (CUDA only) |
| `--grad_accum_steps` | ft | optional | gradient accumulation for a larger effective batch |
| `--max_grad_norm` | ft | optional | gradient clipping norm (`0` to disable) |

LR-related flags (use one of the modes):

| Flag | Module | What it does |
|---|---|---|
| `--lr_dynsweep` | both | run the sweep first, then train with the selected LR (single-process; rank 0 only) |
| `--lr_from_sweep <path>` | both | load an LR from a prior sweep's JSON (overrides the static LR) |
| `--sweep_rule {4a,4b}` | both | which selector rule to read from the JSON (default `4a`) |
| `--lr` | pred | static LR if you skip the sweep entirely |

For FineTuneModule, the sweep produces a single LR that is then assigned to **both** `backbone_lr` and `head_lr` (single-LR sweep — separate-LR sweeps are a planned extension).

Run `python predict_pring.py --help` (or `python finetune_pring.py --help`) for the full flag list.

---

## Other Tasks

Same pattern as PRING; only the script name and the head architecture differ.

For DAVIS / SPOT the extra flag is just one line:

```bash
torchrun --nproc_per_node=4 predict_davis.py \
    --splits_dir    $DAVIS_SPLITS/random_split \
    --mol_emb_path  $DAVIS_DATA/SMILES_MolFormer_XL.npy \
    --model_name    esm2_150M \
    --model_path    $ESM2_WEIGHTS \
    --arch_type     concat \
    --lr_dynsweep --sweep_rule 4a \
    --num_epochs 200 --batch_size 128 --extraction_batch_size 32 \
    --early_stopping_patience 15 --seed 42 \
    --save_dir      $OUT_DIR/esm2_150M_davis_random
```

DAVIS ships four split designs (`random_split`, `cold_drug_split`, `cold_target_split`, `cold_drug_cold_target_split`) — point `--splits_dir` at the one you want to evaluate against.

The same applies to FineTuneModule — substitute `finetune_davis.py` and the fine-tune-specific flag set (smaller `--batch_size`, `--grad_accum_steps`, `--use_amp`, etc.) shown in the Quickstart above.

---

## Structure-Based Models (FoldVision, ESM-IF, ESM-GearNet)

Sequence models can read straight from the TSV `sequence` columns. Structure models need PDB files (or another structural representation) prepared up front:

- **ESM-IF / ESM-GearNet** — pass `--pdb_dir` pointing at a directory of `{protein_id}.pdb` files. The pipeline takes care of feature extraction.
- **FoldVision** and other models that ship their own preprocessing (custom feature files, multi-run inference embeddings, etc.) — **follow the official repository of that model** to generate the required input files first.

Once the model's official pipeline has produced per-protein vectors, plug them into ProteinArena via `--emb_dir`. FoldVision, for example, exposes test-time-augmentation runs as `embeddings_run00.npz`, `embeddings_run01.npz`, …; pass them via:

```bash
python predict_davis.py \
    --splits_dir       $DAVIS_SPLITS/random_split \
    --emb_dir          /path/to/foldvision/embeddings_nruns_5 \
    --foldvision_runs  5 \
    --embedding_dim    1024 \
    --mol_emb_path     $DAVIS_DATA/SMILES_MolFormer_XL.npy \
    --arch_type        concat \
    --lr_dynsweep \
    --sweep_rule 4a \
    --num_epochs 200 --batch_size 64 \
    --early_stopping_patience 15 \
    --seed 42 \
    --save_dir         $OUT_DIR/foldvision_davis_random
```

`--foldvision_runs N` triggers test-time averaging across the N stored runs at inference / training time.

---

## Submitting a New Model

We accept new PLMs through GitHub issues. Please **do not open a PR directly**; we want to triage scope and reproducibility first.

To submit a new model, use the following as checklist and include:

1. **Model name** as it should appear in the registry (e.g. `mymodel_300M`).
2. **Family** — sequence-based, structure-based, or hybrid.
3. **Embedding dimension** of the mean-pooled representation.
4. **Weights** — Hugging Face hub ID, Zenodo DOI, or other public mirror. We cannot accept weights behind authenticated download.
5. **License** of the weights and code.
6. **Reference loader code** — minimal Python snippet showing how to (a) load the model, (b) tokenize one sequence, (c) produce a per-residue or mean-pooled vector, including the package versions the model has been tested on. For FineTuneModule support, include a differentiable forward (`forward(model, sequences) -> (batch, embedding_dim)` with gradients attached).
7. **Preprocessing requirements** — any structural inputs, MSAs, residue features, or non-standard tokenizers that ProteinArena would have to call into.
8. **Sanity check** — for a small held-out FASTA, include the mean-pooled vector for one or two proteins so we can verify our integration matches yours.

We will then either (a) wire the model into `MODEL_REGISTRY` + add a `_load_<family>` / `_extract_<family>` pair in `util_model.py` (and a `_ft_forward_<family>` in `util_finetune_model.py` if applicable), or (b) ask you to send a PR following the same pattern. Either way, the resulting benchmark numbers will land on the public results dashboard.

If your model needs heavy preprocessing, please provide proper documentation and supporting code files for the same.
