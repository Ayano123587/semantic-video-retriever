# Semantic Video Retriever

Find timestamped visual moments in video with a natural-language query. This pipeline samples frames, embeds them using **real pretrained OpenAI CLIP ViT-B/32 weights**, persists a normalized NumPy vector index, and returns cosine-ranked timestamps with temporal nonmax suppression. The model is pinned to an immutable Hugging Face revision; no randomly initialized encoder or metadata-only search is used.

## Run the real demo

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev]"
python -m src.data.make_dataset
python -m src.models.train_model data/raw/big-buck-bunny.mp4
python -m src.models.predict_model "An animated bird singing on a tree branch"
python -m src.models.evaluate
python -m pytest -q
```

The first model download is approximately 605 MB; CPU inference on the 30-second sample is sufficient. `train_model` follows the requested repository naming convention but **builds an index using pretrained weights; it does not train a new CLIP model**. Index vectors and downloaded video remain in ignored `data/`; small provenance/evaluation summaries are committed in `models/`.

Index your own licensed footage or multiple files:

```sh
video-index path/to/clip1.mp4 path/to/clip2.mp4 --output data/processed/my-index --interval 2 --device cpu
video-search "dramatic sword clash in the rain" --index data/processed/my-index --top-k 5 --suppress-seconds 4
```

A response contains the source filename and SHA256, sampled frame number, exact frame-derived timestamp in seconds, sampled scene ID, and cosine similarity. Similarity is not a calibrated probability. Results are frame matches; the system does not infer actions, dialogue, music or plot from audio.

## Retrieval design

Frames are selected at fixed time intervals using the video's reported FPS. A small RGB change heuristic assigns approximate shot IDs at sampled frames. Near-identical CLIP embeddings (cosine >= .985) are removed within the same sampled shot. Query results suppress nearby timestamps from the same video so one shot does not fill the entire result list. NumPy exact cosine search keeps the implementation inspectable; approximate indexing can be added when corpus size justifies it.

The index manifest records video hashes, encoder identity/revision, interval, dimension, frame counts and vector-file hash. Querying verifies vector integrity and encoder identity. Index files use `allow_pickle=False`. Source code never silently substitutes random weights if model download fails.

## Data and attribution

The reproducible demo uses a 30-second Big Buck Bunny excerpt, approximately 5.6 MB, from a commit-pinned mirror. Original film: **(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org**, [CC-BY-3.0](https://peach.blender.org/about/). Source URL, exact hash and excerpt scope are in `models/sample-provenance.json`. [Dataset notes](docs/datasets.md) explain mirror provenance. Downloaded media are not redistributed in this repository.

The four-query first diagnostic is supplemented by a controlled movie-held-out study: eight validation queries across Big Buck Bunny and Elephants Dream select retrieval settings; eight frozen Sintel queries are evaluated once against all three films. Four sampling/prompt configurations and chronological/uniform-random baselines are reported. See [experiment plan](docs/experiment-plan.md), [evaluation protocol](docs/evaluation.md), and actual JSON results. This small open-movie study does not establish large-scale video benchmark performance.

## Limits

Coarse sampling can miss short scenes; sampled shot IDs are a heuristic rather than frame-perfect scene detection. Static CLIP features may miss temporal relations, complex actions, negation and small objects. Encoding truncates long text to CLIP's token limit. Variable-frame-rate videos may have approximate timestamps because this reader uses reported average FPS; constant-frame-rate media are recommended when exact seek alignment matters. The current index is rebuilt rather than incrementally updated, and search is linear in frame count. Use only footage you are authorized to process.

`notebooks/` contains sampling EDA and a complete runnable retrieval walkthrough. Production indexing and evaluation live in `src/`; CI runs isolated unit tests without downloading model weights or media.

## Controlled movie-held-out study

```sh
python -m src.data.make_dataset --all
python -m src.models.train_model data/raw/big-buck-bunny.mp4 data/raw/elephants-dream.mp4 --interval 1 --output data/processed/validation-1
python -m src.models.train_model data/raw/big-buck-bunny.mp4 data/raw/elephants-dream.mp4 --interval 2 --output data/processed/validation-2
python -m src.models.experiment select
python -m src.models.train_model data/raw/big-buck-bunny.mp4 data/raw/elephants-dream.mp4 data/raw/sintel-trailer.mp4 --interval 2 --output data/processed/heldout
video-search "Wide rolling sand dunes in a desert" --index data/processed/heldout
```

The checked-in selection chooses interval2 and raw query. The final test output is preserved in `models/heldout-evaluation.json`; the test runner refuses to overwrite it. A new independent test set is required for future tuning studies. Index building uses pretrained CLIP rather than learning new model parameters. Use separate virtual environments for these repositories because the requested layout exposes a package named `src`.

## Runnable notebook

Open [the complete workflow](notebooks/02_retrieval_workflow.ipynb) for real-data preparation, production-module execution, saved artifacts, and results plots. Expensive execution is explicit; recorded results are labeled and held-out evaluation is not automatically repeated.
