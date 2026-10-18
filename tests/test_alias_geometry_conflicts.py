import copy
import hashlib
import json

import pytest

from src.data import decoder_preflight, prepare_moments


def cohort(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    plan = {"splits": {}}
    receipts = []
    hashes = {}
    conflict_sha = hashlib.sha256(b"conflicting-source").hexdigest()
    for split, count in (("train", 100), ("val", 20), ("test", 40)):
        rows = []
        plan["splits"][split] = []

        def add(name, segments, digest, annotations=1):
            plan["splits"][split].append(
                {"video": name, "aws_url": "https://example.org/" + name,
                 "original_url": "https://example.org/" + name, "license": "fixture"}
            )
            receipts.append({"split": split, "video": name, "status": "complete",
                             "sha256": digest, "bytes": 10})
            rows.extend({"video": name, "num_segments": segments,
                         "annotation_id": f"{name}-{i}", "description": "fixture question",
                         "times": [[0, 0]]} for i in range(annotations))

        for i in range(count):
            name = f"{split}-{i}.mov"
            add(name, 6, hashlib.sha256(name.encode()).hexdigest())
        if split == "val":
            add("same-geometry-alias.mov", 6, hashlib.sha256(b"val-0.mov").hexdigest())
            add("conflict-five.mov", 5, conflict_sha, annotations=4)
            add("conflict-six.mov", 6, conflict_sha, annotations=2)
        raw = json.dumps(rows).encode()
        filename = split + "_data.json"
        (metadata / filename).write_bytes(raw)
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(decoder_preflight, "HASHES", hashes)
    monkeypatch.setattr(prepare_moments, "HASHES", hashes)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(plan))
    receipt_file = tmp_path / "receipts.jsonl"
    receipt_file.write_text("".join(json.dumps(r) + "\n" for r in receipts))

    def inspect(record, path):
        assert record["sha256"] != conflict_sha, "Ambiguous geometry must not be decoded"
        return {**record, "duration": 30.0, "fps": 25.0,
                "last_frame_decoded": True, "status": "valid"}

    monkeypatch.setattr(decoder_preflight, "inspect", inspect)
    report = decoder_preflight.preflight(receipt_file, metadata, tmp_path, tmp_path / "decoder")
    return metadata, selection, receipt_file, report, conflict_sha


def test_entire_conflicting_hash_is_excluded_before_alias_coalescing(tmp_path, monkeypatch):
    metadata, selection, receipts, report, conflict_sha = cohort(tmp_path, monkeypatch)
    assert report["total"] == 161 and report["valid"] == 160
    rows = [r for r in report["rows"] if r["sha256"] == conflict_sha]
    assert len(rows) == 1
    assert rows[0]["reason"] == "annotation_geometry_conflict"
    assert [(r["video"], r["num_segments"]) for r in rows[0]["aliases"]] == [
        ("conflict-five.mov", 5), ("conflict-six.mov", 6)]
    output = tmp_path / "prepared"
    manifest = prepare_moments.stage(output, metadata, receipts, selection,
                                    tmp_path / "decoder/preflight.json")
    assert manifest["excluded_decoder_media_sha256"] == [conflict_sha]
    assert manifest["within_split_alias_copies"] == 1
    assert manifest["counts"]["val"]["source_video_ids"] == 21
    assert manifest["counts"]["val"]["unique_media"] == 20
    queries = json.loads((output / "val.json").read_text())
    assert len(queries) == 21  # All six conflicting annotations, and only those, removed.
    assert not any(q["source_video"].startswith("conflict-") for q in queries)
    assert not any(r["sha256"] == conflict_sha for r in json.loads((output / "media.json").read_text()))


@pytest.mark.parametrize("tamper", ["missing_alias", "geometry", "status", "missing_reason", "invented_conflict"])
def test_conflict_provenance_tampering_fails_before_output(tmp_path, monkeypatch, tamper):
    metadata, selection, receipts, report, conflict_sha = cohort(tmp_path, monkeypatch)
    report = copy.deepcopy(report)
    conflict = next(r for r in report["rows"] if r["sha256"] == conflict_sha)
    if tamper == "missing_alias":
        conflict["aliases"].pop()
    elif tamper == "geometry":
        conflict["aliases"][0]["num_segments"] = 6
    elif tamper == "status":
        conflict["status"] = "valid"
    elif tamper == "missing_reason":
        conflict.pop("reason")
    else:
        report["rows"][0]["reason"] = "annotation_geometry_conflict"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="conflict provenance mismatch"):
        prepare_moments.stage(output, metadata, receipts, selection, path)
    assert not output.exists()
