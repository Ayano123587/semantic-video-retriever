"""Additive rank evidence for the unchanged, guarded one-shot evaluator.

Importing this module uses only the standard library. Native model operations
occur only in the explicitly invoked cloud release function.
"""

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Inclusive start/end bins required")
    start, end = value
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < 6:
        raise ValueError("Valid retained temporal bins required")
    return start, end


def _iou(prediction, reference):
    a, b = _interval(prediction)
    c, d = _interval(reference)
    return max(0, min(b, d) + 1 - max(a, c)) / (max(b, d) + 1 - min(a, c))


def _consensus(prediction, references):
    values = sorted(_iou(prediction, reference) for reference in references)
    if not values:
        raise ValueError("Actual human references required")
    selected = values[-3:]
    total = 0.0
    for value in selected:
        total += value
    return total / len(selected)


def replay_rank_evidence(candidates, rows, expected_queries, expected_ids=None):
    """Independently replay all eight metrics from saved finite ranks, no arrays."""
    if type(expected_queries) is not int or expected_queries <= 0 or not candidates:
        raise ValueError("Full nonempty cohort counts required")
    keys = [(row["video"], _interval(row["moment"])) for row in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Unique canonical candidates required")
    by_video = {}
    for index, (video, _) in enumerate(keys):
        if not isinstance(video, str) or not video:
            raise ValueError("Actual canonical video required")
        by_video.setdefault(video, []).append(index)
    for indices in by_video.values():
        moments = [keys[index][1] for index in indices]
        segments = max(end for _, end in moments) + 1
        if moments != [
            (start, end) for start in range(segments) for end in range(start, segments)
        ]:
            raise ValueError("Complete original triangular candidate geometry required")
    totals = {
        name: []
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
    ids = []
    for row in rows:
        query_id, video = row["id"], row["video"]
        if not isinstance(query_id, str) or not query_id or video not in by_video:
            raise ValueError("Full actual query identity required")
        ids.append(query_id)
        references = [_interval(value) for value in row["references"]]
        if not references:
            raise ValueError("Actual human references required")
        top = row["top5"]
        local = row["within_video_order"]
        if (
            len(top) != 5
            or len(local) != len(by_video[video])
            or len(local) != len(set(local))
            or set(local) != set(by_video[video])
        ):
            raise ValueError("Full unique within-video order and top five required")
        top_indices = [item["index"] for item in top]
        if len(top_indices) != len(set(top_indices)) or any(
            type(index) is not int or not 0 <= index < len(keys)
            for index in top_indices + local
        ):
            raise ValueError("Exact integer candidate indices required")
        values = [item["score"] for item in top]
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Finite actual score values required")
        if any(
            values[index] < values[index + 1]
            or (
                values[index] == values[index + 1]
                and top_indices[index] > top_indices[index + 1]
            )
            for index in range(4)
        ):
            raise ValueError("Stable original-candidate top-five ordering required")
        top_local = [index for index in top_indices if keys[index][0] == video]
        if top_local != local[: len(top_local)]:
            raise ValueError("Global and within-video prefix differs")
        moments = [keys[index][1] for index in local]
        if any(reference not in moments for reference in references):
            raise ValueError("Every human reference must exist in the geometry")

        def positive(index, video=video, references=references):
            return (
                keys[index][0] == video
                and _consensus(keys[index][1], references) >= 0.5
            )

        ranks = sorted(moments.index(reference) + 1 for reference in references)
        average_rank = math.fsum(ranks[:3]) / len(ranks[:3])
        totals["corpus_consensus_r1_iou50"].append(positive(top_indices[0]))
        totals["corpus_consensus_r5_iou50"].append(
            any(positive(index) for index in top_indices)
        )
        totals["corpus_video_r1"].append(keys[top_indices[0]][0] == video)
        totals["corpus_video_r5"].append(
            any(keys[index][0] == video for index in top_indices)
        )
        totals["within_video_official_rank1"].append(average_rank <= 1)
        totals["within_video_official_rank5"].append(average_rank <= 5)
        totals["within_video_official_miou"].append(_consensus(moments[0], references))
        totals["consensus_attainable_fraction"].append(
            any(_consensus(moment, references) >= 0.5 for moment in moments)
        )
    if (
        len(ids) != expected_queries
        or len(ids) != len(set(ids))
        or (expected_ids is not None and ids != expected_ids)
    ):
        raise ValueError("All ordered, unique, externally bound queries required")
    return {
        "queries": len(ids),
        "candidate_moments": len(candidates),
        **{name: math.fsum(values) / len(values) for name, values in totals.items()},
    }


def _export_evidence(
    output, scores, candidates, queries, identities, available, metrics, np
):
    directory = Path(output).with_name(Path(output).stem + "-rank-evidence")
    directory.mkdir(parents=True, exist_ok=False)
    matrix = np.asarray(scores)
    if (
        matrix.shape != (2972, 15039)
        or not np.isfinite(matrix).all()
        or len(queries) != 2972
        or len(available) != 741
    ):
        raise ValueError("Exact complete frozen held-out cohort required")
    before = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    geometry = [
        {"video": row["video"], "moment": list(_interval(row["moment"]))}
        for row in candidates
    ]
    geometry_path = directory / "candidate-keys.json"
    geometry_path.write_text(
        json.dumps(geometry, allow_nan=False) + "\n", encoding="utf-8"
    )
    ranks_path = directory / "test-ranks.jsonl"
    with ranks_path.open("x", encoding="utf-8") as stream:
        for vector, query, identity in zip(matrix, queries, identities, strict=True):
            order = np.argsort(-vector, kind="stable")
            local = [
                int(index)
                for index in order
                if candidates[index]["video"] == query["video"]
            ]
            row = {
                "id": identity["id"],
                "video": query["video"],
                "references": [list(_interval(value)) for value in query["times"]],
                "top5": [
                    {"index": int(index), "score": float(vector[index])}
                    for index in order[:5]
                ],
                "within_video_order": local,
                "text_characters": len(query["text"]),
                "whitespace_words": len(query["text"].split()),
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if hashlib.sha256(matrix.tobytes(order="C")).hexdigest() != before:
        raise ValueError("Selected score matrix changed during rank export")
    written = {
        path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in (geometry_path, ranks_path)
    }
    with ranks_path.open(encoding="utf-8") as stream:
        replayed = replay_rank_evidence(
            json.loads(geometry_path.read_bytes()),
            (json.loads(line) for line in stream),
            2972,
            [row["id"] for row in identities],
        )
    for path in (geometry_path, ranks_path):
        if {"sha256": digest(path), "bytes": path.stat().st_size} != written[path.name]:
            raise ValueError("Saved evidence changed during independent replay")
    for name, value in replayed.items():
        if not math.isclose(value, metrics[name], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Independent saved-rank aggregate replay differs: " + name)
    media = [
        {
            "video": video,
            **{
                key: row[key]
                for key in ("num_segments", "sha256", "license", "original_url")
                if key in row
            },
        }
        for video, row in available.items()
    ]
    media_path = directory / "media-citations.json"
    media_path.write_text(json.dumps(media, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "schema": "one_shot_video_rank_evidence_v1",
        "queries": 2972,
        "videos": 741,
        "candidate_moments": 15039,
        "caption_text_exported": False,
        "score_matrix_sha256": before,
        "score_matrix_hash_is_not_independent_host_verification": True,
        "selected_metrics_replayed": replayed,
        "files": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in (geometry_path, ranks_path, media_path)
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (directory / "COMPLETE").write_text(digest(manifest_path) + "\n", encoding="ascii")
    return {
        "manifest_sha256": digest(manifest_path),
        "directory": directory.name,
        "queries": 2972,
        "videos": 741,
        "independent_saved_rank_replay": True,
    }


def release_with_rank_evidence(
    source, cache, checkpoint, official_test, output, *, encoder_factory=None
):
    """Original release guards/math plus same-execution additive evidence IO."""
    import numpy as np
    import torch

    from src.models import moment_evaluation as original

    MARKER, POLICY = original.MARKER, original.POLICY
    verified_inputs = original.verified_inputs
    read_json = original.read_json
    checked_test_cohort = original.checked_test_cohort
    interval = original.interval
    pinned_encoder = original.pinned_encoder
    scores_for = original.scores_for
    policy_scores = original.policy_scores
    evaluate = original.evaluate
    if encoder_factory is None:
        encoder_factory = original.CLIPEncoder
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Refuse to overwrite an existing report")
    if (source / MARKER).exists():
        raise ValueError("Test release already attempted, regardless of report path")
    identities, selected, available, candidates, vectors = verified_inputs(
        source, cache, checkpoint
    )
    policy = read_json(source / POLICY)
    if (
        not isinstance(policy, dict)
        or policy.get("schema") != 1
        or policy.get("test_targets_used") is not False
        or any(policy.get(k) != value for k, value in identities.items())
    ):
        raise ValueError("Frozen global policy differs from supplied artifacts")
    if digest(official_test) != identities["official_test_sha256"]:
        raise ValueError("Changed official test annotation file")
    with (source / MARKER).open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "release attempted; preserve even on failure",
                **identities,
                "policy_sha256": digest(source / POLICY),
                "report": str(output.resolve()),
            },
            handle,
            indent=2,
        )
    cohort = checked_test_cohort(source, read_json(source / "manifest.json"), available)
    rows = read_json(official_test)
    if not isinstance(rows, list):
        raise ValueError("Invalid official annotation root")  # noqa: TRY004 - preserve original guard
    queries = []
    seen = set()
    actual_identities = []
    aliases = cohort["aliases"]
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Malformed official annotation row")  # noqa: TRY004 - preserve original guard
        source_video = row.get("video")
        if source_video not in aliases:
            continue
        video = aliases[source_video]
        geometry = available[video]["num_segments"]
        actual_identities.append(
            dict(  # noqa: C408 - preserve original query identity AST
                id=f"test:{position}:{row.get('annotation_id')}",
                source_video=source_video,
                video=video,
                num_segments=row.get("num_segments"),
            )
        )
        if row.get("num_segments") != geometry:
            raise ValueError("Official/prepared test geometry mismatch")
        text, references = row.get("description"), row.get("times")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(references, list)
            or not references
        ):
            raise ValueError("Malformed human query/references")
        if any(interval(t)[1] >= geometry for t in references):
            raise ValueError("Human target outside available geometry")
        queries.append({"video": video, "times": references, "text": text})
        seen.add(video)
    if seen != set(available):
        raise ValueError("Official test queries do not cover all available test media")
    if actual_identities != cohort["queries"]:
        raise ValueError(
            "Full official source query identities differ from prospective freeze"
        )
    encoder = encoder_factory("cpu")
    pinned_encoder(encoder.receipt)
    if encoder.receipt != selected["encoder"]:
        raise ValueError("Actual inference encoder differs from frozen checkpoint")
    text_vectors = np.concatenate(
        [
            encoder.texts([r["text"] for r in queries[i : i + 32]])
            for i in range(0, len(queries), 32)
        ]
    )
    if (
        text_vectors.dtype != np.float32
        or text_vectors.shape != (len(queries), 512)
        or not np.isfinite(text_vectors).all()
        or not np.allclose(np.linalg.norm(text_vectors, axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Invalid encoded test queries")
    x, v = torch.from_numpy(text_vectors), torch.from_numpy(vectors)
    report_policy_sha256 = digest(source / POLICY)
    baseline_scores = scores_for([], x, v)
    baseline_metrics = evaluate(baseline_scores, candidates, queries)
    del baseline_scores
    selected_scores = policy_scores(selected, x, v)
    selected_metrics = evaluate(selected_scores, candidates, queries)
    report = {
        "schema": 1,
        **identities,
        "policy_sha256": report_policy_sha256,
        "baseline": baseline_metrics,
        "selected": selected_metrics,
        "encoder": encoder.receipt,
        "test_labels_used_only_after_global_freeze": True,
        "limits": [
            "Available licensed subset, not the full benchmark.",
            "One-shot consistency guard is not adversarial authentication.",
            "Unknown pretrained exposure; repeated validation selection.",
        ],
    }
    report["rank_evidence"] = _export_evidence(
        output,
        selected_scores,
        candidates,
        queries,
        actual_identities,
        available,
        selected_metrics,
        np,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report
