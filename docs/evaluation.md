# Evaluation

The current benchmark is the fixed licensed DiDeMo subset described in
[datasets](datasets.md). All descriptions of a video stay in its official split.
Candidate moments are contiguous inclusive five-second bins. The primary custom metric
is corpus R@1 requiring both the correct video and consensus temporal IoU >= 0.5.
R@5, video retrieval, and official within-video best-three-annotator rank/mIoU are
reported separately. Stable candidate order resolves ties without consulting targets.

Model selection uses 261 validation queries. Phase 1 fitted three adapter configurations
across three seeds. The preregistered follow-up fitted four conditional-loss weights
across three seeds and evaluated sixteen fixed frozen/adapted score mixtures.
Promotion requires at least one percentage point in primary recall with no more than
one percentage point loss in within-video mIoU. Neither study qualified; frozen CLIP
remains selected at R@1 0.153257 and mIoU 0.376905. This is repeated validation selection,
not proof of a globally optimal model.

The final policy binds prepared-source, cache, checkpoint and official annotation hashes
before test access. The release command creates an exclusive source-directory marker
before parsing test targets and refuses a second report path or changed artifacts.
This provides consistency, not adversarial authentication. Test results are pending.

The earlier eight-query Sintel demo is retained in `models/heldout-evaluation.json`.
Its Hit@1 0.625 and Hit@3 0.875 describe a small manually annotated animation study;
they are not DiDeMo scores and are not used for current model selection.

Tests independently cover temporal metrics, multiple positives, conditional loss,
exact score mixtures, source/cache identity, checkpoint loading and one-shot release.
Software fixtures test mechanics; measured model results use real human annotations.
