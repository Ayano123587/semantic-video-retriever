# Semantic Video Retriever

Retrieve timestamped video moments from natural-language queries using CLIP embeddings and contiguous temporal candidates. The project includes real-data preparation, full-corpus encoding, residual text-adapter training, validation-based policy selection, and a frozen final evaluation.

The selected policy is **frozen CLIP**. All 21 adapter fits completed, and 16 advanced scoring policies were evaluated. None passed the preregistered validation promotion gate; the project makes no learned-improvement claim.

## Data

The retained DiDeMo research cohort contains **7,454 unique videos**: 5,916 training, 797 validation, and 741 test videos. It includes 23,202 training descriptions, 3,132 validation queries, and 2,972 eligible final-test queries. Source aliases are resolved to canonical media hashes.

This is the complete retained licensed, available cohort, not the entire official benchmark. Data preparation records source URLs, license categories, download receipts, exact duplicates, and decoder geometry. Noncommercial restrictions and source attribution still apply. Footage, captions, feature arrays, and checkpoints are excluded from GitHub. See [data sources](docs/datasets.md).

## Results

Final evaluation uses all 2,972 eligible queries, 741 videos, and 15,039 temporal candidates.

| Metric | Result |
|---|---:|
| Corpus consensus R@1, IoU ≥ 0.5 | 6.49% |
| Corpus consensus R@5, IoU ≥ 0.5 | 12.95% |
| Within-video official R@1 | 12.72% |
| Within-video official R@5 | 34.96% |
| Within-video official mIoU | 35.83% |

Corpus retrieval requires both the correct video and temporal agreement. Within-video metrics assume the correct video is known. The full saved-rank export independently reproduces all eight reported metrics. Production verification also re-encoded every validation query and checked search across all retained media.

## Run

Use a separate environment for this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m src.models.research_search --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead. To search your own real video, use footage you own or are licensed to process:

```bash
video-index path/to/licensed-video.mp4 --output data/processed/my-index --interval 2 --device cpu
video-search "person opening a door" --index data/processed/my-index --top-k 5 --suppress-seconds 4
```

This builds a pretrained CLIP frame index and returns matching timestamps; it does not fine-tune the encoder or reproduce the temporal-retrieval benchmark. The first run downloads the pinned encoder weights.

Reproducing the audited research search requires the matching prepared-data manifest, receipts, feature cache, selected checkpoint, and SHA-bound approval. The loader checks those identities and the frozen numerical runtime before loading model or array data.

```bash
python -m src.models.research_search "person opening a door" \
  --cache PATH_TO_CACHE --prepared PATH_TO_PREPARED \
  --receipts PATH_TO_RECEIPTS --checkpoint PATH_TO_SELECTED_CHECKPOINT \
  --approval PATH_TO_APPROVAL --approval-sha256 APPROVAL_SHA256 \
  --split all --top-k 5
```

The audit ran on CPU with Python 3.13.15, PyTorch 2.11.0+cpu, NumPy 2.1.3, and Transformers 4.57.6. Exact package, encoder-asset, and CPU/runtime identities are recorded in the evidence. Private research artifacts must be obtained or reproduced under their source licenses; the command does not download a ready-made benchmark bundle.

## Notebooks

| Notebook | Contents |
|---|---|
| [00 — Complete workflow](notebooks/00_complete_workflow.ipynb) | Real-data contracts, feature/training implementation, completed experiments, selection, and final results |
| [01 — Source exploration](notebooks/01_real_source_eda.ipynb) | Real source coverage, descriptions, and temporal annotation distributions |
| [02 — Training](notebooks/02_full_training.ipynb) | All 21 fit histories, early stopping, 16 policies, and selected-policy validation errors |
| [03 — Final evaluation](notebooks/03_final_evaluation.ipynb) | Production verification, full-cohort rank replay, localization errors, and citations |
| [04 — Corpus encoding](notebooks/04_full_corpus_encoding.ipynb) | Encoding coverage and temporal geometry for all 7,454 retained videos |
| [05 — Advanced experiments](notebooks/05_advanced_training_analysis.ipynb) | Twelve advanced fits and all 16 validation policies |

The analysis reads authenticated actual reports and rank artifacts. It does not rerun model fitting. See [Report](Report.md) for the evaluation protocol and interpretation.

Run the saved analyses in their locked environment:

```bash
uv sync --project notebook-environment --locked
```

This uses Python 3.12.15. Read each notebook's input cells for its matching
licensed data and recorded evidence; reproducing the saved analyses does not
silently start model training.

## Limits

Static CLIP frame pooling can miss motion order, brief actions, and small objects. Exact media hashes do not detect every near duplicate. Pretraining exposure is unknown. Some test queries were inspected during earlier development, which limits held-out independence. The final selected policy and scoring rules were frozen before the complete final release; no final-test tuning was performed.
