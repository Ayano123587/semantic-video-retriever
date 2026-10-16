# Frozen retrieval experiment plan

Recorded before validation comparisons or Sintel retrieval output. Model is fixed pretrained CLIP ViT-B/32 at pinned revision; no finetuning or training claim. Validation media: Big Buck Bunny opening excerpt and Elephants Dream excerpt. Held-out test media: Sintel trailer, disjoint film. All query annotations were written from visual contact sheets without model rankings.

Controlled configurations: frame sampling intervals 1s and 2s crossed with text prompt prefixes empty and `a video frame of `. All use scene-aware .985 dedup and four-second temporal suppression. Select the highest validation Hit@1, then Hit@3; ties prefer the cheaper 2s index and unmodified query. Freeze selection and query/annotation/index hashes in a selection manifest. Run selected configuration once on held-out test annotations; retain failures. No test-guided changes to model, interval, queries or relevance intervals.

Report query-level Hit@1, Hit@3 and MRR@3. Baselines are earliest-first retrieval and seeded uniform random ranking with identical temporal suppression, averaged over1000 draws. Comparison concerns retrieval configuration on a fixed pretrained representation, not a model training experiment. Small samples and the same broad animated-film domain limit conclusions; unknown CLIP pretraining exposure remains.

The final search index includes all three films, so the two validation films serve as distractors for the eight Sintel test queries. Test relevance requires both the correct film identity and an annotated time interval.
