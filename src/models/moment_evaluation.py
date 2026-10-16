"""Explicit one-shot DiDeMo test release after a globally frozen policy.

The marker belongs to the prepared source directory, not the requested report.
Copying the entire source to evade this guard violates the evaluation protocol;
filesystem markers provide consistency, not adversarial authenticity.
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch

from src.config import MODEL_ID, MODEL_REVISION
from src.data.didemo import HASHES, REVISION
from src.models.encoder import CLIPEncoder, WEIGHTS_SHA256
from src.models.moment_training import Adapter, digest, scores_for
from src.models.moments import evaluate, moments, interval

POLICY = "frozen-test-policy.json"
MARKER = "test-release-marker.json"
ADAPTER_WINNERS = {"lr001", "lr0003", "decay01"} | {
    f"lambda{weight}-alpha{alpha}" for weight in (0, 0.5, 1, 2)
    for alpha in (0.25, 0.5, 0.75, 1.0)}


def adapted_weight(selected):
    """Historical phase-1 checkpoints imply full adapter weight."""
    value = selected.get("adapted_score_weight", 1.0)
    if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("adapted_score_weight must be a finite float in [0, 1]")
    return value


def policy_scores(selected, x, v):
    """Mix cosine scores, not parameters, after equal-seed adapter averaging."""
    alpha = adapted_weight(selected)
    baseline = scores_for([], x, v)
    if not selected["states"]:
        return baseline
    adapted = scores_for(selected["states"], x, v)
    return (1.0 - alpha) * baseline + alpha * adapted


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Unreadable JSON: " + str(path)) from exc


def pinned_encoder(receipt):
    expected = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "weights_sha256": WEIGHTS_SHA256}
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Encoder identity differs from the pinned production model")
    info = receipt.get("loading_info")
    if not isinstance(info, dict) or any(info.get(k) != [] for k in
            ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete pretrained loading evidence")


def verified_inputs(source, cache, checkpoint):
    source, cache, checkpoint = Path(source), Path(cache), Path(checkpoint)
    prepared, encoded = read_json(source / "manifest.json"), read_json(cache / "manifest.json")
    if not isinstance(prepared, dict) or not isinstance(encoded, dict):
        raise ValueError("Invalid source/cache manifest")
    if not {"media.json", "train.json", "val.json"}.issubset(prepared.get("files", {})) or not {
            "test-moments.npy", "test-candidates.json"}.issubset(encoded.get("files", {})):
        raise ValueError("Required source/test cache hashes are absent")
    for directory, manifest in ((source, prepared), (cache, encoded)):
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Missing source/cache hashes")
        for name, expected in files.items():
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Unsafe manifest filename")
            if digest(directory / name) != expected:
                raise ValueError("Changed source/cache file: " + name)
    if prepared.get("official_metadata_sha256", {}).get("test_data.json") != HASHES["test_data.json"]:
        raise ValueError("Unpinned official test source")
    if prepared.get("source_revision") != REVISION:
        raise ValueError("Unpinned prepared source revision")
    if (source / "test.json").exists() or any((cache / name).exists() for name in
            ("test-text.npy", "test-queries.json")) or encoded.get("test_text_or_targets_encoded") is not False:
        raise ValueError("Test text/targets must remain absent from the prepared cache")
    if encoded.get("smoke_only") is not False:
        raise ValueError("Smoke cache cannot support holdout release")
    if encoded.get("source_manifest_sha256") != digest(source / "manifest.json"):
        raise ValueError("Encoder cache uses a different prepared source")
    pinned_encoder(encoded.get("encoder"))
    try:
        selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError("Unreadable safe selected checkpoint") from exc
    if not isinstance(selected, dict) or type(selected.get("schema")) is not int or selected.get("schema") != 1 or selected.get("test_targets_used") is not False:
        raise ValueError("Invalid selected checkpoint schema")
    if selected.get("cache_manifest_sha256") != digest(cache / "manifest.json"):
        raise ValueError("Checkpoint/cache identity mismatch")
    if selected.get("encoder") != encoded["encoder"]:
        raise ValueError("Checkpoint encoder identity mismatch")
    states = selected.get("states")
    winner = selected.get("winner")
    alpha = adapted_weight(selected)
    if not isinstance(winner, str) or winner not in ADAPTER_WINNERS | {"frozen_clip"} or not isinstance(states, list) or (
            winner == "frozen_clip" and states) or (winner != "frozen_clip" and len(states) != 3):
        raise ValueError("Expected frozen baseline or exactly three selected seed states")
    if winner.startswith("lambda") and alpha != float(winner.split("-alpha")[1]):
        raise ValueError("Winner label and adapted score weight disagree")
    for state in states:
        if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or not torch.isfinite(v).all()
                                              for v in state.values()):
            raise ValueError("Malformed or nonfinite adapter state")
        try:
            Adapter().load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("Adapter state schema mismatch") from exc
    media = read_json(source / "media.json")
    if not isinstance(media, list) or any(not isinstance(r, dict) or not isinstance(r.get("video"), str)
                                         or r.get("split") not in ("train", "val", "test") for r in media):
        raise ValueError("Invalid media list")
    available = {r["video"]: r for r in media if r["split"] == "test"}
    if not available or len(available) != sum(r["split"] == "test" for r in media) or len(available) != prepared.get("counts", {}).get("test", {}).get("unique_media"):
        raise ValueError("Incomplete prepared test media")
    candidates = read_json(cache / "test-candidates.json")
    if not isinstance(candidates, list) or any(not isinstance(r, dict) or "video" not in r or "moment" not in r for r in candidates):
        raise ValueError("Invalid candidate list")
    expected = {(name, m) for name, row in available.items() for m in moments(row["num_segments"])}
    actual = [(r["video"], interval(r["moment"])) for r in candidates]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Test candidate geometry does not exactly cover available media")
    vectors = np.load(cache / "test-moments.npy", allow_pickle=False)
    if vectors.dtype != np.float32 or vectors.shape != (len(candidates), 512) or not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid normalized test media embeddings")
    identities = {"source_manifest_sha256": digest(source / "manifest.json"),
                  "cache_manifest_sha256": digest(cache / "manifest.json"),
                  "selected_checkpoint_sha256": digest(checkpoint),
                  "official_test_sha256": HASHES["test_data.json"], "winner": winner,
                  "adapted_score_weight": alpha}
    return identities, selected, available, candidates, vectors


def freeze(source, cache, checkpoint):
    """Serialize the final cross-study choice before any test labels are opened."""
    source = Path(source)
    if (source / MARKER).exists():
        raise ValueError("Test release already attempted")
    identities, _, _, _, _ = verified_inputs(source, cache, checkpoint)
    policy = {"schema": 1, **identities, "test_targets_used": False,
              "selection_scope": "Final global policy, not a per-report choice"}
    with (source / POLICY).open("x", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, allow_nan=False)
    return policy


def release(source, cache, checkpoint, official_test, output, *, encoder_factory=CLIPEncoder):
    """Deliberately open labels once; use cloud CPU for actual pretrained encoding."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Refuse to overwrite an existing report")
    if (source / MARKER).exists():
        raise ValueError("Test release already attempted, regardless of report path")
    identities, selected, available, candidates, vectors = verified_inputs(source, cache, checkpoint)
    policy = read_json(source / POLICY)
    if not isinstance(policy, dict) or policy.get("schema") != 1 or policy.get("test_targets_used") is not False or any(
            policy.get(k) != value for k, value in identities.items()):
        raise ValueError("Frozen global policy differs from supplied artifacts")
    # Hashing bytes does not parse targets. The exclusive marker precedes parsing.
    if digest(official_test) != identities["official_test_sha256"]:
        raise ValueError("Changed official test annotation file")
    with (source / MARKER).open("x", encoding="utf-8") as handle:
        json.dump({"status": "release attempted; preserve even on failure", **identities,
                   "policy_sha256": digest(source / POLICY), "report": str(output.resolve())}, handle, indent=2)
    rows = read_json(official_test)
    if not isinstance(rows, list):
        raise ValueError("Invalid official annotation root")
    queries = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed official annotation row")
        if row.get("video") not in available:
            continue
        geometry = available[row["video"]]["num_segments"]
        if row.get("num_segments") != geometry:
            raise ValueError("Official/prepared test geometry mismatch")
        text, references = row.get("description"), row.get("times")
        if not isinstance(text, str) or not text.strip() or not isinstance(references, list) or not references:
            raise ValueError("Malformed human query/references")
        if any(interval(t)[1] >= geometry for t in references):
            raise ValueError("Human target outside available geometry")
        queries.append({"video": row["video"], "times": references, "text": text})
        seen.add(row["video"])
    if seen != set(available):
        raise ValueError("Official test queries do not cover all available test media")
    encoder = encoder_factory("cpu")
    pinned_encoder(encoder.receipt)
    if encoder.receipt != selected["encoder"]:
        raise ValueError("Actual inference encoder differs from frozen checkpoint")
    text_vectors = np.concatenate([encoder.texts([r["text"] for r in queries[i:i+32]])
                                   for i in range(0, len(queries), 32)])
    if text_vectors.dtype != np.float32 or text_vectors.shape != (len(queries), 512) or not np.isfinite(text_vectors).all() or not np.allclose(
            np.linalg.norm(text_vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid encoded test queries")
    x, v = torch.from_numpy(text_vectors), torch.from_numpy(vectors)
    report = {"schema": 1, **identities, "policy_sha256": digest(source / POLICY),
              "baseline": evaluate(scores_for([], x, v), candidates, queries),
              "selected": evaluate(policy_scores(selected, x, v), candidates, queries),
              "encoder": encoder.receipt, "test_labels_used_only_after_global_freeze": True,
              "limits": ["Available licensed subset, not the full benchmark.",
                         "One-shot consistency guard is not adversarial authentication.",
                         "Unknown pretrained exposure; repeated validation selection."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "release"))
    for name in ("source", "cache", "checkpoint"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--official-test")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.source, args.cache, args.checkpoint)
    else:
        if not args.official_test or not args.output:
            parser.error("release requires --official-test and --output")
        result = release(args.source, args.cache, args.checkpoint, args.official_test, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
