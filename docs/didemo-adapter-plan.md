# DiDeMo source preflight and frozen adapter study

The fixed expanded access cohort yielded 365 training, 74 validation and 47 test
source video IDs. Exact SHA256 audit found two validation IDs with identical media;
there are therefore 73 unique validation videos. Before any model outcomes, we
coalesce identical-media aliases to one canonical filename (lexicographic order),
retain every human query and require consistent annotated segment geometry.
This prevents two copies of the same physical video becoming false negatives.
No exact media content crosses splits. Near-duplicate content and common owners
remain possible; unknown CLIP pretraining overlap is not resolved by this audit.

The staged source contains 1,393 training and 261 validation human descriptions.
Test media identity and segment count are staged, but no test descriptions,
annotation IDs or target intervals. Reading official JSON for video metadata is
not a claim that the file was never parsed; test labels do not enter model choice.
Downloaded footage/captions remain ignored and private. Availability filtering is
substantial and results cannot be presented as the full official benchmark.

## Scorer

Candidate intervals enumerate all contiguous inclusive five-second bins, up to
six bins (21 moments). Inclusive IoU is independently verified against set overlap.
Consensus IoU averages the three largest human-reference IoUs (or all if fewer).
The primary **custom corpus consensus R@1 at IoU >= .5** requires the correct
video and consensus overlap. R@5, video-only recall and within-video official
average-best-three exact ranks/mIoU are secondary. This corpus metric is distinct
from official within-video localization. Ties preserve candidate order, never
use ground truth. Queries with no consensus-positive candidate remain in all
validation/test denominators; report attainable fraction. They are excluded only
from contrastive training because there is no valid positive target.

Reference: [pinned official scorer](https://github.com/LisaAnne/LocalizingMoments/blob/b6a555c8134581305d0ed4716fbc192860e0b88c/utils/eval.py).

## Frozen representation and search

Use pinned CLIP ViT-B/32 (identity in src/config.py) with real loading-key checks
and actual cached weight SHA verification. No random substitute. Freeze image and
text encoders. Sample two frames per available five-second bin at quarter and
three-quarter positions, clipped to actual last partial-bin duration. Restrict
footage to the first30seconds; reject inconsistent duration/decoder failure before
training and record any prospective availability amendment, without outcome-based
replacements. Normalize each frame embedding, average within a candidate interval,
then normalize. No video filenames, annotation IDs or timestamps are learned
features. Retain source hashes, frame geometry, cache hashes and text truncations.

Fit a residual text adapter:512->64->512 GELU, zero-initialized up projection,
normalized output. Three configurations: AdamW LR.001/weightdecay.01;
LR.0003/weightdecay.01; LR.001/weightdecay.1. Seeds42,1337,2026 each. Batch64,
50epochs maximum, minimum5epochs, patience8 on validation primary metric;
checkpoint ties earliest. Multi-positive contrastive loss compares each training
query to all training moments, counting every correct-video consensus-positive
interval as positive. Temperature.07 fixed. Unknown semantically similar queries
across different videos can still create false negatives and are a limitation.
No test captions/targets are needed for feature extraction or training.

Compare frozen CLIP, chronological and fixed-seed uniform-random controls, then
all three adapter procedures by equal-seed mean cosine scores, never best seed.
Promote only if primary validation recall gains at least.01 absolute and mean
within-video IoU does not drop more than.01; otherwise retain pretrained CLIP.
Ties prefer frozen, then the earlier listed adapter. Freeze serialized selection
before any one-shot test-label release. Preserve every configuration, seed history,
negative outcome and checkpoint. Further validation hypotheses must be preregistered
before execution; no holdout-driven redesign. A gain does not establish an optimum.

Free private CPU cloud is the current compute path because the account's weekly
GPU quota is exhausted. There is no user-imposed training-duration cap; the normal
12hour per-kernel service ceiling applies. Use a real decoding/encoder/train/save/
reload smoke before the full job, and plain mounted Python modules, never encoded
source literals. Extraction may include test media only; test query features are
created only after final selection and deliberate label release.
