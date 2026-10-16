# Retrieval experiment report

## Current evidence

The real-data encoder and adapter readiness checks completed. Full DiDeMo adapter training is running; validation selection, full learning curves and held-out scores are pending. Historical movie-demo results remain separate.

| Actual completed check | Result |
|---|---:|
| Real videos encoded | 2 |
| Encoding time, excluding model loading | 5.97 seconds |
| Encoder serialization replay, maximum absolute difference | 2.98e-8 |
| Real training queries in adapter smoke | 2 |
| Training loss, epoch 1 → 2 | 1.615436 → 1.569930 |
| Adapter checkpoint replay, maximum absolute difference | 0 |

These are mechanics checks. They do not estimate retrieval accuracy. Exact receipts, source commit, model weight hash and empty loading-key diagnostics are in [video-smoke.json](docs/video-smoke.json).

## Source and split

DiDeMo metadata is pinned to `LisaAnne/LocalizingMoments` revision `b6a555c8134581305d0ed4716fbc192860e0b88c`. License metadata and deterministic SHA-ordered availability selection precede scoring. The initial fixed candidate cohort failed its availability minimum; the prospective expansion and original failures are preserved in the [availability amendment](docs/didemo-access-amendment.md).

Identical media within validation are coalesced to one candidate video while retaining every query. No exact media content crosses source splits. Near duplicates, common uploaders and unknown pretrained exposure remain possible. After complete decoder geometry inspection, the current frozen cohort is:

| Split | Unique videos | Human queries staged |
|---|---:|---:|
| Train | 364 | 1,390 |
| Validation | 73 | 261 |
| Test | 47 | None |

Official JSON was read for test video metadata; this is not a claim that the file was never parsed. No test descriptions, annotation IDs or target intervals enter feature preparation or model selection. The v2 staging manifest records exact file and media hashes.

Full-run version 1 rejected a third training source with declared six bins and actual duration 24.17 seconds, before any training or retrieval metric. All 485 sources were then inspected; only that training source failed. The whole source and three queries were excluded before the corrected run. Validation/test were unchanged. There was no fabricated padding, target remapping or outcome-based replacement. [Decoder receipt](docs/decoder-preflight.json).

## Representation, loss and selection

Frozen `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` is checked against weight SHA `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. Missing, unexpected or mismatched loading keys are rejected.

Two frames per five-second bin are sampled at quarter and three-quarter positions within the actual duration. Their CLIP features are pooled, then every contiguous bin interval becomes a normalized candidate vector. The final partial bin uses real frames only. Test media features may be extracted without test descriptions or targets.

The learned text residual is 512 → 64 → 512 with GELU and a zero-initialized output layer. Multi-positive contrastive loss uses consensus-positive moments in the correct video and frozen visual negatives. Queries without a valid positive are excluded only from training; they stay in evaluation denominators. Semantically similar cross-video moments can be false negatives.

Three optimizer procedures each run seeds 42, 1337 and 2026. Equal-seed mean scores compete with frozen CLIP, chronological and fixed-seed random controls. Promotion requires at least .01 absolute primary validation recall gain and no more than .01 within-video mean-IoU loss; ties prefer the frozen model, then listed order. Every seed and failed setting is retained. Selection is serialized before deliberate test-label release. The [adapter protocol](docs/didemo-adapter-plan.md) retains its preregistration and decoder amendment.

## Scoring and reproduction

Custom corpus consensus R@1 at IoU ≥ .5 requires both correct video and temporal agreement. Consensus averages the three largest human-reference overlaps. Official within-video average-best-three exact rank and mean IoU are reported separately. Stable candidate order breaks ties without targets. This available subset cannot be compared directly with full official benchmark scores.

The [main notebook](notebooks/00_main_video_retrieval.ipynb) contains the complete working algorithms, real run identities and actual checkpoint replay. Supporting notebooks retain executed aggregate plots. Training and encoding opt-ins use fresh output directories; held-out scoring is never automatic. Production modules are authoritative:

```powershell
python -m src.models.moment_encoder --help
python -m src.models.moment_training --help
```

Cloud execution uses readable mounted Python source and private licensed inputs. Completed full-run output will be hash-verified before its curves and validation comparisons are added. Private descriptions, downloaded footage, weights and credentials are not published.
