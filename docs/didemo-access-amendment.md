# Prospective source-availability amendment

The original fixed650video candidate cohort completed before training or model
scoring. Available:81train,8validation,30test videos (646.55MB). The pinned original
metadata and licenses were accessible, but most mapped official AWS video objects
failed. Exact media hashes show no cross-split duplicate among available files.
The original cohort fails the predeclared minimum100/20/40available gate.

No captions were inspected for selection and no score exists. Prospectively expand
the same license-filtered SHA256-ordered candidate prefix to3000train,500validation,
500test videos (or all license-eligible videos if a split contains fewer). Preserve
the original selection/failures and store a new immutable expanded selection and
receipt journal. No successful video is dropped based on captions, labels or model
outputs; no individual failed ID is replaced selectively. Original videos fall
within the expanded prefix. All official split boundaries remain unchanged.

Reuse preserved downloaded source bytes and record new receipts. Four workers,
128MiB ingestion limit and the100/20/40minimum remain. Report all unavailable IDs
privately and public aggregate availability counts. The available subset can be
biased by source deletion/availability; do not claim full-benchmark results.
No generated imagery or synthetic targets fill missing data.
