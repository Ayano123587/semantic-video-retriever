"""Import public inference/data modules without fitting or downloading assets."""
import importlib
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if os.environ.get(name) != "1":
        raise RuntimeError("CI imports require explicit offline mode")

for name in (
    "src.models.research_search",
    "src.models.release_evidence",
    "src.models.moment_evaluation",
    "src.models.resumable_training",
    "src.models.portable_training",
    "src.models.advanced_training",
    "src.data.prepare_research",
):
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Imported another project's module: " + name)
print("Public source imports passed; no model fitting or asset retrieval requested.")
