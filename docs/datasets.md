# Data and model provenance

The study uses real videos and human temporal descriptions from [DiDeMo](https://github.com/LisaAnne/LocalizingMoments), pinned at `b6a555c8134581305d0ed4716fbc192860e0b88c`.

The final retained cohort has **7,454 unique videos**: 5,916 training, 797 validation, and 741 test videos. It contains 23,202 training descriptions, 3,132 validation queries, and 2,972 eligible test queries. Source aliases are resolved to canonical video hashes; the 761 retained test source IDs represent 741 unique videos.

Sources carry CC BY, CC BY-SA, CC BY-NC, or CC BY-NC-SA 2.0 licenses. NoDerivatives categories are excluded. Each retained video keeps its source URL, attribution, license, download receipt, and decoder geometry. Noncommercial terms remain applicable. The complete retained licensed and available cohort differs from the complete official benchmark.

Footage, captions, frame samples, feature arrays, and checkpoints are not redistributed in this repository. Reproduction requires obtaining source material under its license and passing the recorded availability and integrity checks. Published reports and notebooks contain aggregate results, algorithms, and limited analysis examples.

The encoder is [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32), revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, with weight SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. The [CLIP release](https://github.com/openai/CLIP) uses the MIT license. Weight identity, loading keys, and the numerical environment are checked before inference.

All 21 residual-adapter fits completed. No trained policy passed the fixed validation promotion gate, so frozen CLIP remains selected. Final held-out scores cover all 2,972 eligible queries, with no final-test tuning. Some test queries were inspected during earlier development; unknown pretraining exposure and near duplicates also limit independence claims.
