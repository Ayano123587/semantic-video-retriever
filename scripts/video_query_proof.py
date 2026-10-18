"""Prospective CLIP asset inventory and full-validation production routing proof."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

QUERY_COUNT = 3132
EMBEDDING_TOLERANCE = 1e-5
METRIC_TOLERANCE = 1e-10
TOP_K = 5
ASSET_REQUIRED = {
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
}
ASSET_OPTIONAL = {"tokenizer.json", "special_tokens_map.json"}
PACKAGES = (
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "torch",
    "numpy",
    "pillow",
    "imageio",
    "imageio-ffmpeg",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def asset_inventory(paths, versions, model_id, revision):
    if (
        not ASSET_REQUIRED.issubset(paths)
        or set(paths) - ASSET_REQUIRED - ASSET_OPTIONAL
    ):
        raise ValueError("Complete supported pinned CLIP processor assets required")
    if set(versions) != set(PACKAGES) or any(
        not isinstance(value, str) or not value for value in versions.values()
    ):
        raise ValueError("Actual prospective dependency versions required")
    return {
        "schema": "prospective_video_query_assets_v1",
        "model_id": model_id,
        "revision": revision,
        "files": {name: sha(path) for name, path in sorted(paths.items())},
        "packages": dict(versions),
        "historical_encoding_assets_attested": False,
        "scope": "Prospective query proof inventory; historical encoding did not record these asset hashes",
    }


def verify_assets(
    paths, versions, inventory, model_id, revision, expected_packages=None
):
    if inventory.get("schema") == "prospective_video_asset_inventory_v2":
        if (
            inventory.get("model_id") != model_id
            or inventory.get("revision") != revision
            or inventory.get("resolved_commit") != revision
            or inventory.get("native_loading") is not False
            or inventory.get("native_runtime_measured") is not False
            or inventory.get("historical_encoding_assets_attested") is not False
            or inventory.get("future_weight_sha256_obligation")
            != "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
        ):
            raise ValueError(
                "Actual prospective asset identity and separate weight obligation required"
            )
        if (
            not isinstance(expected_packages, dict)
            or set(expected_packages) != set(PACKAGES)
            or versions != expected_packages
            or any(
                not isinstance(value, str) or not value
                for value in expected_packages.values()
            )
        ):
            raise ValueError(
                "Separately root-frozen production packages required; inventory observations are not pins"
            )
        if (
            not ASSET_REQUIRED.issubset(paths)
            or set(paths) - ASSET_REQUIRED - ASSET_OPTIONAL
            or set(paths) != set(inventory["files"])
        ):
            raise ValueError("Exact complete actual processor asset names required")
        for name, path in paths.items():
            if inventory["files"][name] != {
                "sha256": sha(path),
                "bytes": Path(path).stat().st_size,
            }:
                raise ValueError("Root-frozen prospective asset bytes differ")
        return inventory
    actual = asset_inventory(paths, versions, model_id, revision)
    if actual != inventory:
        raise ValueError(
            "Root-frozen actual query assets or dependency versions differ"
        )
    return actual


def validation_rows(prepared, cache, cache_manifest, prepared_manifest):
    prepared, cache = Path(prepared), Path(cache)
    for path, expected in (
        (prepared / "val.json", prepared_manifest["files"]["val.json"]),
        (cache / "val-queries.json", cache_manifest["files"]["val-queries.json"]),
        (cache / "val-text.npy", cache_manifest["files"]["val-text.npy"]),
    ):
        if sha(path) != expected:
            raise ValueError("Full frozen validation inputs differ before parsing")
    rows = json.loads((cache / "val-queries.json").read_bytes())
    original = json.loads((prepared / "val.json").read_bytes())
    if (
        rows != original
        or len(rows) != QUERY_COUNT
        or any(
            not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
            for row in rows
        )
        or len({row["id"] for row in rows}) != QUERY_COUNT
    ):
        raise ValueError("All ordered unique validation queries required")
    return rows


def original_batched_encoding(encoder, rows):
    import numpy as np

    return np.concatenate(
        [
            encoder.texts([row["text"] for row in rows[start : start + 32]])
            for start in range(0, len(rows), 32)
        ]
    )


def same_vector_reference(index, vector, top_k):
    """Per-split original score calls, then independent stable concatenation."""
    import numpy as np
    import torch

    from src.models.localization_followup import score_policy

    scores = []
    locations = []
    for split, part in index.indexes.items():
        values = score_policy(
            part.bundle, torch.from_numpy(vector[None]), part.features
        )[0]
        scores.append(values)
        locations.extend((split, row) for row in part.candidates)
    concatenated = np.concatenate(scores)
    order = np.argsort(-concatenated, kind="stable")[:top_k]
    return [
        (locations[position][0], locations[position][1], float(concatenated[position]))
        for position in order
    ]


def check_matches(matches, reference, media, aliases, durations):
    if len(matches) != len(reference):
        raise ValueError("Production/reference top-k count differs")
    for match, (split, candidate, score) in zip(matches, reference):
        key = (split, candidate["video"])
        source = media[key]
        expected = {
            "split": split,
            "video": candidate["video"],
            "start_seconds": candidate["moment"][0] * 5,
            "end_seconds": min((candidate["moment"][1] + 1) * 5, durations[key]),
            "score": score,
            "source_sha256": source["sha256"],
            "source_video_ids": aliases[key],
            "license": source["license"],
            "original_url": source["original_url"],
        }
        if match != expected:
            raise ValueError(
                "Stable query ranking, source citation or temporal geometry differs"
            )


def replay(
    index, encoder, rows, cached_vectors, recorded_truncation, global_report, metadata
):
    """Cloud only: every query is replayed; no test descriptions or targets."""
    import numpy as np
    import torch

    from src.models.localization_followup import score_policy
    from src.models.moments import evaluate

    if tuple(index.indexes) != ("train", "val", "test"):
        raise ValueError(
            "Complete fixed train-val-test production search order required"
        )
    if (
        len(rows) != QUERY_COUNT
        or cached_vectors.shape != (QUERY_COUNT, 512)
        or cached_vectors.dtype != np.float32
        or not np.isfinite(cached_vectors).all()
    ):
        raise ValueError("Full finite float32 validation cache required")
    batch = original_batched_encoding(encoder, rows)
    if (
        batch.shape != cached_vectors.shape
        or batch.dtype != np.float32
        or not np.isfinite(batch).all()
    ):
        raise ValueError("Full real pretrained encoding shape differs")
    batch_error = float(np.max(np.abs(batch - cached_vectors)))
    if batch_error > EMBEDDING_TOLERANCE:
        raise ValueError("Full cached query embeddings do not replay")
    texts = [row["text"] for row in rows]
    lengths = [
        len(ids)
        for ids in encoder.processor.tokenizer(texts, truncation=False)["input_ids"]
    ]
    truncation = {
        "queries": len(rows),
        "over_token_limit": sum(length > 77 for length in lengths),
        "maximum_tokens": max(lengths),
    }
    if truncation != recorded_truncation:
        raise ValueError("Full validation tokenization/truncation inventory differs")
    val_index = index.indexes["val"]
    metrics = evaluate(
        score_policy(
            val_index.bundle, torch.from_numpy(cached_vectors), val_index.features
        ),
        val_index.candidates,
        rows,
    )
    expected_metrics = global_report["selected_validation"]
    if set(metrics) != set(expected_metrics):
        raise ValueError("Recorded full-validation metric fields differ")
    for name, expected in expected_metrics.items():
        actual = metrics[name]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if (
                not isinstance(actual, (int, float))
                or not np.isfinite(actual)
                or abs(actual - expected) > METRIC_TOLERANCE
            ):
                raise ValueError(
                    "Original full3132 selected-validation metric does not replay"
                )
        elif actual != expected:
            raise ValueError("Original validation metric identity differs")

    class Recorder:
        receipt = encoder.receipt
        vector = None

        def text(self, query):
            self.vector = encoder.text(query)
            return self.vector

    recording = Recorder()
    scalar_batch_error = 0.0
    scalar_cached_error = 0.0
    for position, row in enumerate(rows):
        matches = index.search(row["text"], TOP_K, encoder=recording)
        vector = recording.vector
        scalar_batch_error = max(
            scalar_batch_error, float(np.max(np.abs(vector - batch[position])))
        )
        scalar_cached_error = max(
            scalar_cached_error,
            float(np.max(np.abs(vector - cached_vectors[position]))),
        )
        if max(scalar_batch_error, scalar_cached_error) > EMBEDDING_TOLERANCE:
            raise ValueError(
                "Scalar/batch query encoding agreement exceeds declared tolerance"
            )
        reference = same_vector_reference(index, vector, TOP_K)
        check_matches(matches, reference, *metadata)
    return {
        "schema": "full_validation_video_query_replay_v1",
        "queries": len(rows),
        "query_ids_sha256": canonical([row["id"] for row in rows]),
        "reencoding_batch": 32,
        "embedding_tolerance": EMBEDDING_TOLERANCE,
        "batch_cached_max_abs_error": batch_error,
        "scalar_batch_max_abs_error": scalar_batch_error,
        "scalar_cached_max_abs_error": scalar_cached_error,
        "same_vector_stable_all_split_topk_matches": len(rows),
        "citation_geometry_checks": len(rows)
        * min(TOP_K, sum(len(part.candidates) for part in index.indexes.values())),
        "top_k": TOP_K,
        "truncation": truncation,
        "selected_validation_metrics_replayed": True,
        "test_descriptions_or_targets_used": False,
        "all_split_benchmark_score_claimed": False,
        "different_vector_rank_bitwise_claimed": False,
    }
