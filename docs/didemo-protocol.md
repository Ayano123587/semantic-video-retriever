# Natural-video training and evaluation plan

The user resumed this project and requires real datasets, trained solutions, and
iterative validation. The earlier open-animation demo remains historical.
DiDeMo provides real personal videos with human-written temporal descriptions.
Use official train/validation/test assignments; all captions of a video remain
together. Do not use animation, generated footage, generated captions, or fabricated
targets for measured training/evaluation. Tiny software fixtures remain mechanics only.

Before inspecting question text or model scores, select videos by deterministic
SHA256 order within official splits and verified Creative Commons license metadata.
Target500train/50validation/100test videos, retaining all original annotations for
selected videos. Log missing downloads without replacing them using model scores.
No test captions or labels enter training or validation selection. Fingerprint
metadata/media/model identity, preserve download failures and report the actual
available cohort. Dataset research and access checks precede costly training.

Compare pinned pretrained CLIP with learned regularized text/image projection
adapters using real train pairs only. Primary validation objective: corpus moment
retrieval R@1 at temporal IoU>=0.5; also report R@5, video retrieval and localization
separately. A five-second clip representation averages sampled frames within each
moment. Include chronological/uniform random baselines. Freeze scorer, sampling,
hyperparameter grid and stopping rules after source/access preflight but before
the first model comparison. Test is opened once after final validation selection.
Multiple annotations/overlapping moments must not become false contrastive negatives.

Access preflight succeeded on the first deterministic training video (4.14MB),
using the official AWS media mapping. Restrict the initial cohort to CC BY2.0 and
CC BY-SA2.0 videos with attached public source attribution. Downloads use four
workers and retain failed IDs/partial bytes privately; no replacements. A128MiB
per-source safety limit is an explicit availability filter, not a GPU budget.
Verify exact media hashes across splits before training. Require at least100train,
20validation and40test available videos for this study; otherwise revise the
access plan transparently before model scoring. Metadata parsing stages IDs/gold
privately in advance; human inspection/model choice never uses test captions/gold.

Source: [official DiDeMo repository](https://github.com/LisaAnne/LocalizingMoments),
revision b6a555c8134581305d0ed4716fbc192860e0b88c. Video-specific CC license and
attribution remain attached privately; no footage or captions are redistributed.
Code and aggregate evidence only are published. Claims are limited to the fixed
licensed available subset, not the entire official benchmark or unknown pretraining
overlap. GPU service quotas still apply; user removed artificial time caps.
