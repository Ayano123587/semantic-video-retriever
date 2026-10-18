# Video moment retrieval

The system embeds natural-language queries and video frames with a pinned CLIP encoder, pools contiguous temporal candidates, and ranks them by cosine similarity. Residual text adapters were trained on human descriptions to test whether supervised adaptation improved retrieval.

## Dataset and splits

The complete retained DiDeMo cohort contains 7,454 unique licensed, available videos. Canonical SHA256 identities and source aliases define disjoint prepared splits:

| Split | Unique videos | Queries |
|---|---:|---:|
| Train | 5,916 | 23,202 |
| Validation | 797 | 3,132 |
| Final test | 741 | 2,972 |

The final test contains 761 source IDs resolved to 741 canonical videos. Its cohort identity was frozen before temporal targets were opened for the final release. Five-second segment geometry produces 15,039 final-test candidates. The original prepared-data manifest is preserved; the test cohort is a separate authenticated sidecar.

The source, licensing, availability, and decoder checks are part of the data pipeline. The retained cohort excludes unavailable media and unsupported license categories, so these scores do not represent the complete official DiDeMo benchmark. Raw footage, descriptions, features, and checkpoints remain outside the public repository.

## Training and selection

Nine first-phase fits and twelve advanced fits completed on the full training split. The advanced study evaluated all sixteen localization-loss and adapted-score policies. Configurations used seeds 42, 1337, and 2026, with validation-based epoch selection and early stopping. The notebooks show the actual per-epoch losses and validation histories.

Selection used all 3,132 validation queries and the fixed promotion rules. No trained candidate passed every required gate. The global winner therefore remained frozen CLIP, with an adapted-score weight of zero. Adapter training completed, but it did not establish an eligible improvement over the baseline.

## Production verification

The selected implementation was checked against every validation query and all 7,454 retained media. Batch-32 query encoding matched the cached validation embeddings exactly. Maximum scalar-versus-batch and scalar-versus-cache error was `5.0664e-7`, below the declared `1e-5` tolerance.

All 3,132 same-query-vector searches matched the stable all-split top-five reference, and all 15,660 returned citations passed geometry checks. Full validation metrics were replayed. Source, input, package, encoder-asset, and numerical-runtime checks passed before and after execution. This verifies production consistency; it does not introduce a separate all-corpus benchmark score.

## Final evaluation

The model and scoring policy were frozen before a single final evaluation on all 2,972 eligible test queries. A one-shot marker was created before test annotations were parsed. The same execution exported candidate identities, stable top-five ranks, within-video order, and source citations without caption text.

| Metric | Value |
|---|---:|
| Corpus consensus R@1, IoU ≥ 0.5 | 0.0649394 |
| Corpus consensus R@5, IoU ≥ 0.5 | 0.1295424 |
| Correct-video R@1 | 0.1692463 |
| Correct-video R@5 | 0.2163526 |
| Within-video official R@1 | 0.1271871 |
| Within-video official R@5 | 0.3495962 |
| Within-video official mIoU | 0.3583128 |
| Consensus attainable fraction | 1.0000000 |

An independent implementation replayed all eight metrics from all saved query ranks and matched the reported aggregates within `1e-12`. It also checked every query identity, all 15,039 candidates against the original media geometry, and all 741 citations. The dense score matrix was fingerprinted during execution, rather than independently recomputed on the host.

Corpus metrics require retrieving the correct video and a matching temporal interval. Within-video metrics measure localization when the video is already known. They answer different questions and should not be compared as interchangeable recall scores.

## Reproduction and notebooks

The public source covers preparation, encoder caching, training, policy selection, authenticated query inference, and evaluation. Reproduction requires the exact prepared manifests, licensed media receipts, cache, selected checkpoint, and numerical environment recorded in the evidence. The research loader refuses mismatched artifacts or runtime identities.

Start with `00_complete_workflow.ipynb`. Source exploration and encoding are in notebooks 01 and 04; full training and policy analysis are in 02 and 05; production checks and final saved-rank error analysis are in 03. Analysis notebooks consume actual frozen artifacts and do not silently refit models. Execution and figure inventories are recorded with the final notebook outputs.

## Limitations

The available licensed cohort differs from the full benchmark. Some test queries were inspected in earlier development, so held-out independence is limited. Repeated validation selection and unknown CLIP pretraining exposure also constrain interpretation. Frame pooling lacks an explicit motion model, and exact-duplicate filtering does not eliminate every near duplicate. No final-test result was used to retune the selected model or scoring policy.
