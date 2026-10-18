"""Freeze source identities for full research test coverage without decoding gold."""

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = {"video", "annotation_id", "num_segments"}
CONTRACT = "test-cohort.json"
FULL_COUNTS = (761, 741)
OFFICIAL_SHA256 = "1891c04ec48b3d364c739594b2b6413806b74bd9027c092d896e7ebb930ff1cd"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metadata_only(raw, expected_sha256):
    """Skip target values structurally, trusting syntax only in SHA-pinned source.

    This is not a replacement JSON validator for ignored nested values: it never
    decodes their keys, strings, numbers or intervals into Python objects.
    """
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Pinned official bytes required before extraction")
    text = raw.decode("utf8")
    decoder = json.JSONDecoder()

    def whitespace(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i

    def skip(i):
        i = whitespace(i)
        if i >= len(text):
            raise ValueError("Truncated JSON")
        if text[i] == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    return i + 1
                else:
                    i += 1
            raise ValueError("Unterminated skipped string")
        if text[i] in "[{":
            end = "]" if text[i] == "[" else "}"
            i = whitespace(i + 1)
            if text[i] == end:
                return i + 1
            while True:
                i = skip(i)
                i = whitespace(i)
                if end == "}" and text[i] == ":":
                    i = whitespace(skip(i + 1))
                if text[i] == end:
                    return i + 1
                if text[i] != ",":
                    raise ValueError("Malformed skipped container")
                i = whitespace(i + 1)
        start = i
        while i < len(text) and text[i] not in ",]} \r\n\t":
            i += 1
        if i == start:
            raise ValueError("Malformed skipped scalar")
        return i

    i = whitespace(0)
    if text[i] != "[":
        raise ValueError("Official row array required")
    i = whitespace(i + 1)
    rows = []
    while i < len(text) and text[i] != "]":
        if text[i] != "{":
            raise ValueError("Official row objects required")
        i = whitespace(i + 1)
        row = {}
        seen = set()
        while text[i] != "}":
            key, i = decoder.raw_decode(text, i)
            if not isinstance(key, str) or key in seen:
                raise ValueError("Duplicate or invalid row key")
            seen.add(key)
            i = whitespace(i)
            if text[i] != ":":
                raise ValueError("Missing key separator")
            i = whitespace(i + 1)
            if key in FIELDS:
                value, i = decoder.raw_decode(text, i)
                row[key] = value
            else:
                i = skip(i)
            i = whitespace(i)
            if text[i] == "}":
                break
            if text[i] != ",":
                raise ValueError("Missing row separator")
            i = whitespace(i + 1)
        if (
            set(row) != FIELDS
            or not isinstance(row["video"], str)
            or not row["video"].strip()
            or type(row["num_segments"]) is not int
            or not 1 <= row["num_segments"] <= 6
            or type(row["annotation_id"]) not in (int, str)
            or (
                isinstance(row["annotation_id"], str)
                and not row["annotation_id"].strip()
            )
        ):
            raise ValueError("Complete source identity fields required")
        rows.append(dict(source_index=len(rows), **row))
        i = whitespace(i + 1)
        if text[i] == "]":
            break
        if text[i] != ",":
            raise ValueError("Missing array separator")
        i = whitespace(i + 1)
    if not rows or i >= len(text) or text[i] != "]" or whitespace(i + 1) != len(text):
        raise ValueError("Complete official row array required")
    return {
        "schema": "official_test_identity_index_v1",
        "official_test_sha256": expected_sha256,
        "rows": rows,
        "target_values_decoded": False,
    }


def metadata_only(raw, expected_sha256):
    try:
        return _metadata_only(raw, expected_sha256)
    except (IndexError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed or truncated pinned JSON structure") from exc


def build_contract(prepared, receipts, index):
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_bytes())
    if digest(receipts) != manifest.get("receipt_sha256"):
        raise ValueError("Prepared terminal receipts required")
    if (
        index.get("schema") != "official_test_identity_index_v1"
        or index.get("target_values_decoded") is not False
        or index.get("official_test_sha256")
        != manifest["official_metadata_sha256"]["test_data.json"]
    ):
        raise ValueError("Pinned metadata-only official index required")
    media = json.loads((prepared / "media.json").read_bytes())
    if digest(prepared / "media.json") != manifest["files"]["media.json"]:
        raise ValueError("Prepared media changed")
    raw = Path(receipts).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Terminal receipt journal required")
    records = [json.loads(line) for line in raw.splitlines() if line]
    keys = [(r["split"], r["video"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate receipt identity")
    groups = {}
    excluded = set(manifest["excluded_decoder_media_sha256"])
    for r in records:
        if r["status"] == "complete" and r["sha256"] not in excluded:
            groups.setdefault(r["sha256"], []).append(r)
    aliases = {}
    test_media = {r["video"]: r for r in media if r["split"] == "test"}
    if len(test_media) != sum(r["split"] == "test" for r in media):
        raise ValueError("Duplicate canonical test media")
    for sha, group in groups.items():
        if len({r["split"] for r in group}) != 1:
            raise ValueError("Retained media crosses splits")
        if group[0]["split"] != "test":
            continue
        name = min(r["video"] for r in group)
        if name not in test_media or test_media[name]["sha256"] != sha:
            raise ValueError("Unknown canonical test source")
        for r in group:
            aliases[r["video"]] = name
    counts = manifest["counts"]["test"]
    if (counts["source_video_ids"], counts["unique_media"]) != FULL_COUNTS:
        raise ValueError("Full761-source/741-canonical test cohort required")
    if (
        len(aliases) != counts["source_video_ids"]
        or set(aliases.values()) != set(test_media)
        or len(test_media) != counts["unique_media"]
    ):
        raise ValueError("Complete source-to-canonical test coverage required")
    identities = []
    seen = set()
    covered = set()
    annotations = set()
    for position, row in enumerate(index["rows"]):
        if (
            type(row.get("source_index")) is not int
            or row.get("source_index") != position
            or set(row) != FIELDS | {"source_index"}
        ):
            raise ValueError("Complete ordered metadata index required")
        name = row["video"]
        if (
            type(row["annotation_id"]) not in (int, str)
            or (
                isinstance(row["annotation_id"], str)
                and not row["annotation_id"].strip()
            )
            or type(row["num_segments"]) is not int
            or not 1 <= row["num_segments"] <= 6
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError("Invalid metadata identity types")
        if name not in aliases:
            continue
        canonical_video = aliases[name]
        if row["num_segments"] != test_media[canonical_video]["num_segments"]:
            raise ValueError("Aliased official geometry mismatch")
        identity = f"test:{position}:{row['annotation_id']}"
        annotation = (name, row["annotation_id"])
        if identity in seen or annotation in annotations:
            raise ValueError("Duplicate source query identity")
        annotations.add(annotation)
        seen.add(identity)
        covered.add(name)
        identities.append(
            dict(
                id=identity,
                source_video=name,
                video=canonical_video,
                num_segments=row["num_segments"],
            )
        )
    if covered != set(aliases):
        raise ValueError("Metadata index misses eligible source queries")
    return dict(
        schema="full_research_test_cohort_v1",
        prepared_manifest_sha256=digest(prepared / "manifest.json"),
        receipt_sha256=digest(receipts),
        official_test_sha256=index["official_test_sha256"],
        metadata_index_sha256=canonical(index),
        source_video_ids=len(aliases),
        unique_media=len(test_media),
        aliases=aliases,
        queries=identities,
        query_count=len(identities),
        query_ids_sha256=canonical(identities),
        targets_used=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official", required=True)
    p.add_argument("--official-sha256", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--receipts", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.official_sha256 != OFFICIAL_SHA256:
        raise ValueError("Pinned official test source SHA required")
    output = Path(a.output)
    if output.exists():
        raise ValueError("Fresh prospective test contract required")
    index = metadata_only(Path(a.official).read_bytes(), a.official_sha256)
    contract = build_contract(a.prepared, a.receipts, index)
    with output.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(contract, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
