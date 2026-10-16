# Evaluation protocol

`docs/demo-queries.json` contains four manually authored descriptions and relevant timestamp intervals. Labels were assigned by visually inspecting the sampled contact sheet before observing retrieval output: clouds/meadow 2-10s, stream 12-14s, bird 16-20s, tree hollow 24-30s. Black opening frames and title cards remain indexed; they are genuine distractors.

The experiment uses pretrained CLIP ViT-B/32, a fixed two-second sampling interval, within-shot cosine deduplication threshold .985, exact normalized cosine search and four-second temporal suppression. Hit@1 and Hit@3 count a result as relevant if its timestamp falls within the annotated interval. These are query-level hit rates, not frame-average recall or a large benchmark's recall metric.

There is no model fitting, validation-driven threshold tuning, or hidden search over queries. Four selected opening-sequence queries on one clip cannot estimate broad generalization. CLIP may have seen related imagery during pretraining; pretraining overlap is unverified. All model scores and timestamps are retained in `models/demo-evaluation.json`, including failures. The repository tests cover timestamp schedules, normalization/shape constraints, persistence/integrity, temporal suppression, scene-aware deduplication, empty query handling and revision compatibility without pretending those tests evaluate semantics.

To extend evaluation, annotate multiple licensed videos with distractor content, reserve query sets before tuning, include brief actions/negation/rare objects and out-of-domain material, and report precision, interval recall and confidence intervals separately. Add variable-frame-rate decoding with presentation timestamps before promising frame-perfect timing on arbitrary video.

## Controlled study beyond the diagnostic

See the prespecified [experiment plan](experiment-plan.md). Validation: eight queries across two films. Final test: eight manually annotated queries from a distinct Sintel film, with both validation films retained in the search index as distractors. Query relevance requires correct film identity and interval membership. Annotation files and manifests are hashed before the final run. Selection maximizes validation Hit@1 then Hit@3, breaking ties toward the cheaper two-second sampling and raw query. All four configuration results, including equal outcomes, are retained. MRR@3 is descriptive and was not the selection metric.

Uniform random baseline averages 1000 seeded candidate permutations with the same temporal suppression. Earliest-first baseline uses timestamps with the same suppression. Final test runs only after freezing the selection, and the script refuses to replace an existing final result. This protocol limits test-guided optimization but cannot eliminate unknown CLIP pretraining overlap, annotation subjectivity, or narrow-domain bias. No finetuning or new CLIP training occurred.

## Recorded held-out result

Chosen configuration: 2s sampling, raw query, fixed CLIP ViT-B/32. Three-film index: 61 sampled frames, 55 retained. Eight held-out Sintel queries: Hit@1 0.6250, Hit@3 0.8750, MRR@3 0.7500. Uniform-random baseline: Hit@1 0.0464, Hit@3 0.1269; chronological baseline: Hit@1 0.0000, Hit@3 0.0000. These results describe only the prespecified small open-animation study.
