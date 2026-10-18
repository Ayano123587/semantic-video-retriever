"""Cloud-only complete phase1 validation rank export, without fitting."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay_metrics(candidates, rows):
    """Plain exported-rank replay; no native arrays, model, or scoring calls."""
    keys = [(row["video"], tuple(row["moment"])) for row in candidates]
    require(
        len(keys) == 15927 and len(set(keys)) == 15927,
        "All unique ordered validation candidates required",
    )
    require(
        len(rows) == 3132 and len({row["id"] for row in rows}) == 3132,
        "All unique validation queries required",
    )
    totals = {
        name: 0.0
        for name in (
            "corpus_consensus_r1_iou50",
            "corpus_consensus_r5_iou50",
            "corpus_video_r1",
            "corpus_video_r5",
            "within_video_official_rank1",
            "within_video_official_rank5",
            "within_video_official_miou",
            "consensus_attainable_fraction",
        )
    }

    def consensus(moment, refs):
        a, b = moment
        values = [
            max(0, min(b, d) - max(a, c) + 1) / (max(b, d) - min(a, c) + 1)
            for c, d in refs
        ]
        return sum(sorted(values)[-3:]) / min(3, len(values))

    for row in rows:
        top, local, refs = row["top5"], row["within_video_order"], row["references"]
        require(
            isinstance(row["id"], str) and row["id"],
            "Actual nonempty query identity required",
        )
        require(
            all(type(index) is int and 0 <= index < len(keys) for index in local),
            "Plain integer complete local indices required",
        )
        require(
            len(top) == 5 and len({entry["index"] for entry in top}) == 5,
            "Five unique corpus ranks required",
        )
        require(
            all(
                type(entry["index"]) is int
                and 0 <= entry["index"] < len(keys)
                and math.isfinite(entry["score"])
                for entry in top
            ),
            "Finite valid ranked predictions required",
        )
        require(
            all(top[i]["score"] >= top[i + 1]["score"] for i in range(4)),
            "Descending corpus scores required",
        )
        require(
            all(
                top[i]["score"] != top[i + 1]["score"]
                or top[i]["index"] < top[i + 1]["index"]
                for i in range(4)
            ),
            "Original-order stable tied top ranks required",
        )
        expected_local = {i for i, key in enumerate(keys) if key[0] == row["video"]}
        require(
            len(local) == len(expected_local)
            and set(local) == expected_local
            and local,
            "Complete unique correct-video order required",
        )
        require(
            refs
            and all(
                type(a) is int and type(b) is int and 0 <= a <= b < 6 for a, b in refs
            ),
            "Actual inclusive temporal references required",
        )
        moments = [keys[index][1] for index in local]
        require(
            all(tuple(ref) in moments for ref in refs),
            "All actual human references must exist in geometry",
        )
        positive = lambda index, video=row["video"], references=refs: (
            keys[index][0] == video and consensus(keys[index][1], references) >= 0.5
        )
        totals["corpus_consensus_r1_iou50"] += positive(top[0]["index"])
        totals["corpus_consensus_r5_iou50"] += any(
            positive(item["index"]) for item in top
        )
        totals["corpus_video_r1"] += keys[top[0]["index"]][0] == row["video"]
        totals["corpus_video_r5"] += any(
            keys[item["index"]][0] == row["video"] for item in top
        )
        ranks = sorted(moments.index(tuple(ref)) + 1 for ref in refs)[:3]
        average = sum(ranks) / len(ranks)
        totals["within_video_official_rank1"] += average <= 1
        totals["within_video_official_rank5"] += average <= 5
        totals["within_video_official_miou"] += consensus(moments[0], refs)
        totals["consensus_attainable_fraction"] += any(
            positive(index) for index in local
        )
    return {
        "queries": len(rows),
        "candidate_moments": len(candidates),
        **{name: value / len(rows) for name, value in totals.items()},
        "metric_scope": "licensed available subset; corpus metrics custom; within-video formula official",
        "tie_rule": "stable original candidate order; no gold-dependent tie breaking",
    }


def metrics_equal(actual, expected):
    require(set(actual) == set(expected), "Exact original metric fields required")
    for name, value in expected.items():
        require(
            abs(actual[name] - value) <= 1e-12
            if isinstance(value, (int, float))
            else actual[name] == value,
            "Full validation metrics do not replay",
        )


def export(
    cache,
    prepared,
    selected,
    results,
    output,
    parent_freeze_path,
    parent_freeze_sha256,
    guard,
    runtime,
    *,
    started,
    max_seconds=1800,
):
    """Emit sufficient actual ranks for independent metric replay, not training."""
    cache, prepared, selected, output = map(Path, (cache, prepared, selected, output))
    require(
        sys.platform == "linux"
        and cache.resolve().is_relative_to(Path("/kaggle/input"))
        and selected.resolve().is_relative_to(Path("/kaggle/input")),
        "Actual cloud mounted inputs required",
    )
    require(
        digest(parent_freeze_path) == parent_freeze_sha256,
        "Actual external parent-freeze SHA required",
    )
    contract = json.loads(Path(parent_freeze_path).read_bytes())
    require(
        type(max_seconds) is int
        and max_seconds == 1800
        and isinstance(started, (int, float))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and 0 <= time.monotonic() - started < max_seconds,
        "Actual original finite service budget required",
    )
    require(
        contract["schema"] == "video_completed_phase1_validation_parent_freeze_v1"
        and contract["artifact_revision"] == 2
        and contract["parent_status"] == "COMPLETE"
        and contract["action"] == "export_existing_full_validation_rankings_only"
        and contract["training"] is False
        and contract["test_annotations_used"] is False
        and contract["validation_queries"] == 3132
        and contract["candidate_moments"] == 15927,
        "Independent full validation export contract required",
    )
    output.mkdir(exist_ok=False)
    guard()
    require(
        digest(selected) == contract["selected_checkpoint_sha256"]
        and digest(results) == contract["results_sha256"],
        "Actual completed selected policy and results required",
    )
    # The caller's stdlib guard authenticates every cache and source file before
    # these native imports. Only validation feature arrays/annotations are parsed.
    import numpy as np
    import torch

    from scripts.video_query_proof import validation_rows
    from src.models.localization_followup import score_policy
    from src.models.moments import evaluate, moments

    actual_runtime = runtime()
    original = json.loads(Path(results).read_bytes())
    require(
        original["completed_fits"] == original["required_fits"] == 9
        and original["test_targets_used"] is False,
        "Scientifically complete phase1 result required",
    )
    cache_manifest = json.loads((cache / "manifest.json").read_bytes())
    prepared_manifest = json.loads((prepared / "manifest.json").read_bytes())
    queries = validation_rows(prepared, cache, cache_manifest, prepared_manifest)
    candidates = json.loads((cache / "val-candidates.json").read_bytes())
    media = json.loads((prepared / "media.json").read_bytes())
    expected = [
        {"video": row["video"], "moment": list(moment)}
        for row in media
        if row["split"] == "val"
        for moment in moments(row["num_segments"])
    ]
    require(
        candidates == expected and len(candidates) == 15927,
        "Complete original ordered validation geometry required",
    )
    x = np.load(cache / "val-text.npy", allow_pickle=False)
    v = np.load(cache / "val-moments.npy", allow_pickle=False)
    require(
        x.shape == (3132, 512)
        and v.shape == (15927, 512)
        and x.dtype == np.float32
        and v.dtype == np.float32,
        "Exact complete float32 validation feature geometry required",
    )
    require(
        np.isfinite(x).all()
        and np.isfinite(v).all()
        and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4)
        and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4),
        "Original finite unit-feature semantics required",
    )
    guard()
    bundle = torch.load(selected, map_location="cpu", weights_only=True)
    require(
        bundle["validation"] == original["selected_validation"]
        and bundle["winner"] == original["selected"]
        and bundle["test_targets_used"] is False,
        "Actual complete selected policy identity required",
    )
    scores = score_policy(bundle, torch.from_numpy(x), torch.from_numpy(v))
    require(
        scores.shape == (3132, 15927)
        and scores.dtype == np.float32
        and np.isfinite(scores).all(),
        "Full original finite score matrix required",
    )
    measured = evaluate(scores, candidates, queries)
    metrics_equal(measured, original["selected_validation"])
    video_indices = {}
    for index, candidate in enumerate(candidates):
        video_indices.setdefault(candidate["video"], []).append(index)
    rows = []
    for query, row in zip(queries, scores, strict=True):
        order = np.argsort(-row, kind="stable")
        local = [
            int(index)
            for index in order
            if candidates[index]["video"] == query["video"]
        ]
        require(
            set(local) == set(video_indices[query["video"]]),
            "Every correct-video moment must be ranked",
        )
        rows.append(
            {
                "id": query["id"],
                "video": query["video"],
                "references": query["times"],
                "top5": [
                    {"index": int(index), "score": float(row[index])}
                    for index in order[:5]
                ],
                "within_video_order": local,
            }
        )
    metrics_equal(replay_metrics(candidates, rows), measured)
    require(
        [row["id"] for row in rows] == [query["id"] for query in queries],
        "Exact original full query order required",
    )
    guard()
    require(
        runtime() == actual_runtime and 0 <= time.monotonic() - started < max_seconds,
        "Original whole-service budget and postflight runtime required",
    )
    (output / "candidate-keys.json").write_bytes(
        json.dumps(candidates, allow_nan=False).encode() + b"\n"
    )
    with (output / "validation-ranks.jsonl").open("xb") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False).encode() + b"\n")
    (output / "metrics.json").write_bytes(
        json.dumps(measured, indent=2, allow_nan=False).encode() + b"\n"
    )
    manifest = {
        "schema": "phase1_full_validation_rank_export_v1",
        "complete": True,
        "queries": 3132,
        "candidate_moments": 15927,
        "training": False,
        "test_annotations_used": False,
        "actual_runtime": actual_runtime,
        "parent_freeze_sha256": parent_freeze_sha256,
        "selected_checkpoint_sha256": digest(selected),
        "results_sha256": digest(results),
        "score_matrix_sha256": hashlib.sha256(scores.tobytes(order="C")).hexdigest(),
        "score_matrix_hash_is_not_independent_host_verification": True,
        "query_order_sha256": hashlib.sha256(
            canonical([row["id"] for row in rows])
        ).hexdigest(),
        "files": {
            name: {
                "sha256": digest(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in (
                "candidate-keys.json",
                "validation-ranks.jsonl",
                "metrics.json",
            )
        },
    }
    guard()
    require(
        runtime() == actual_runtime and time.monotonic() - started < max_seconds,
        "Final postflight failed",
    )
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    )
    (output / "COMPLETE").write_text(digest(output / "manifest.json") + "\n")
    return manifest
