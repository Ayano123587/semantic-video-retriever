# Data and model provenance

Original film and license: [Blender Foundation Peach project](https://peach.blender.org/about/), [official download page](https://peach.blender.org/download/). Film license is Creative Commons Attribution 3.0. Attribution: (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org. This does not imply the isolated soundtrack has the same license; no separate soundtrack dataset is used.

The official download host returned HTTP 403 in this environment. The adapter therefore downloads a 480p, 30-second excerpt from [chintan9/Big-Buck-Bunny](https://github.com/chintan9/Big-Buck-Bunny), pinned commit `915c4b2aba75614b20dec3852375b394bb305f10`, exact SHA256 `84abe3ddf1938f0a6053cf6cc7874434177cdce927ba07d603cdd84c4c67a6ab`. The mirror reproduces the original attribution/license. Hash verification prevents unnoticed mirror changes. This excerpt represents only the opening sequence and is not a diverse retrieval dataset.

Model: [OpenAI CLIP source and license](https://github.com/openai/CLIP), [model card](https://huggingface.co/openai/clip-vit-base-patch32), [paper](https://arxiv.org/abs/2103.00020). OpenAI code/model release uses MIT license. Transformers model snapshot is pinned at `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. No task-specific fine tuning was done, so no new learned weights are claimed.

Data folders follow raw/external/interim/processed layout and are ignored. The index summary records model provenance and source checksums; binaries stay local. Additional media require their own license and attribution records. No commercial film, anime episode, or user account content was downloaded.

## Additional licensed media

[Elephants Dream license and attribution](https://orange.blender.org/) permits reuse with attribution: (c) copyright 2006, Blender Foundation / Netherlands Media Art Institute / www.elephantsdream.org. The 37-second excerpt uses the same pinned GitHub mirror commit; its SHA256 is `5e77a77fbcb73ed2f1f2ff6b9361f2d9867de1390f5256ccfcce6ed755d9e704`. The film uses CC-BY-2.5; separately published soundtrack files have different terms and are not collected.

[Sintel license and attribution](https://durian.blender.org/sharing/) is CC-BY-3.0, copyright Blender Foundation | www.sintel.org. A 52-second trailer is retrieved from the W3C media mirror at `https://media.w3.org/2010/05/sintel/trailer.mp4` and frozen by SHA256 `b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254`. This is a hash-pinned remote asset rather than an immutable versioned release; hash drift fails ingestion.

These three excerpts span about119 seconds and multiple visual settings, but are all animated open movies. They are not a substitute for a diverse licensed real-world video benchmark. Full source/license records are in `models/corpus-provenance.json`.

## Natural-video benchmark research

[FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) contains diverse natural CC-BY videos with source attribution and temporal descriptions. Checked revision `84c74091e1c6ee7a5dffabfafb5c9033e4718883`: repository is gated (`auto`), and a direct request for `data/train-00000-of-01357.parquet` returned HTTP401 without an authorized Hugging Face session. No FineVideo videos were downloaded or evaluated. The benchmark should be added after access/terms are established, following its current version/opt-out requirements.

[MSR-VTT](https://www.microsoft.com/en-us/research/?p=238188) is a relevant caption-retrieval benchmark, but source-video rights, usable official download availability, and a reproducible licensed subset were not established in this bounded pass. No scores are invented for either dataset. The current results remain an open-animation study and must not be advertised as natural-video performance.
