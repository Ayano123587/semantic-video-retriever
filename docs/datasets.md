# Data and model provenance

The main study uses real videos and human temporal descriptions from
[DiDeMo](https://github.com/LisaAnne/LocalizingMoments), pinned at
`b6a555c8134581305d0ed4716fbc192860e0b88c`. Videos are selected deterministically
within official splits using verified CC BY 2.0 or CC BY-SA 2.0 metadata.
Per-video attribution, license and media hashes remain with the private source records;
footage and descriptions are not redistributed in this repository.

The historical first cohort contains 364 training videos/1,390 queries,
73 validation videos/261 queries and 47 test videos. The MP4-only filename defect excluded 2,200 eligible sources before HTTP; these were not established access failures. Corrected ingestion now attempts every eligible source. Exact-media
aliases and one invalid training video's geometry are recorded in the source and decoder
preflight receipts. Test descriptions and temporal targets are absent from the training
cache. This filtered cohort is not the full official benchmark.

The frozen encoder is [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32),
revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, weight SHA256
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
The [CLIP release](https://github.com/openai/CLIP) uses the MIT license.
All pretrained loading keys and weight identity are checked. The encoder stays frozen;
residual text adapters were actually trained on the human training pairs. Their
validation results did not qualify for promotion, so frozen CLIP remains selected.

The small indexing demo uses licensed Blender Foundation animation excerpts:
[Big Buck Bunny](https://peach.blender.org/about/) and
[Sintel](https://durian.blender.org/sharing/) under CC BY 3.0, and
[Elephants Dream](https://orange.blender.org/) under CC BY 2.5.
Copyright and attribution belong to Blender Foundation and the credited project teams;
Elephants Dream also credits the Netherlands Media Art Institute.
Exact source URLs, excerpt hashes and attribution are retained in
`models/corpus-provenance.json`. These excerpts support the demo, not the natural-video
benchmark. Separate soundtrack files are not collected.

Raw data, feature arrays and weights remain ignored. Published notebooks expose
aggregate results and algorithms; model pretraining overlap and near-duplicate videos
cannot be ruled out by exact media hashes.

The complete official source has 33,005 training, 4,180 validation and 4,021 test queries across 8,511/1,094/1,037 videos. The current license scope admits 2,165/321/201 videos respectively. All are selected; final download and decoder-valid counts are pending. See [coverage audit](full-coverage-audit.json).

The expanded noncommercial study includes BY-NC 2.0 and BY-NC-SA 2.0 sources: 6,005/825/762 candidate videos. Per-source attribution and terms are preserved; media and weights remain private. NoDerivatives categories are excluded. [Prospective amendment](full-research-coverage-amendment.json) binds this wider scope before additional downloads or fitting.
