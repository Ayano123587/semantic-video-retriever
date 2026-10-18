"""Identity-only fixtures; no real annotations or model inference."""

import hashlib
import json
import pytest
from src.data import test_cohort as c


def test_extractor_does_not_decode_targets(monkeypatch):
    raw = json.dumps(
        [
            dict(
                video="source",
                annotation_id=4,
                num_segments=1,
                description='SECRET gold \\" quote',
                times=[[0, 0]],
                extra={"nested": ["hidden"]},
            )
        ]
    ).encode()
    original = json.JSONDecoder.raw_decode
    decoded = []

    def track(self, text, idx=0):
        value, end = original(self, text, idx)
        decoded.append(value)
        assert value not in ('SECRET gold \\" quote', "hidden")
        assert not isinstance(value, (dict, list))
        return value, end

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", track)
    index = c.metadata_only(raw, hashlib.sha256(raw).hexdigest())
    assert index["rows"] == [
        dict(source_index=0, video="source", annotation_id=4, num_segments=1)
    ]
    assert index["target_values_decoded"] is False


def test_wrong_bytes_refused_before_decode(monkeypatch):
    monkeypatch.setattr(
        json.JSONDecoder, "raw_decode", lambda *_: pytest.fail("decoded")
    )
    with pytest.raises(ValueError, match="Pinned"):
        c.metadata_only(b"[]", "wrong")


@pytest.mark.parametrize(
    "raw",
    [b"[", b'[{"video":"x","times":[[', b'[{"video":"x","description":"unterminated'],
)
def test_truncated_target_structures_raise_valueerror(raw):
    with pytest.raises(ValueError):
        c.metadata_only(raw, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("identity", [True, None, [], {}, "", "   "])
def test_invalid_annotation_metadata_refused(identity):
    raw = json.dumps(
        [
            dict(
                video="fixture",
                annotation_id=identity,
                num_segments=1,
                description="unread",
                times=[[0, 0]],
            )
        ]
    ).encode()
    with pytest.raises(ValueError):
        c.metadata_only(raw, hashlib.sha256(raw).hexdigest())


@pytest.fixture
def alias_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "FULL_COUNTS", (21, 1))
    prepared = tmp_path / "prepared"
    prepared.mkdir()

    def write(path, value):
        path.write_bytes(json.dumps(value).encode())

    names = ["canonical"] + [f"z-alias-{i}" for i in range(20)]
    media = [dict(video="canonical", split="test", num_segments=1, sha256="a" * 64)]
    write(prepared / "media.json", media)
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_bytes(
        b"".join(
            (
                json.dumps(
                    dict(video=n, split="test", status="complete", sha256="a" * 64)
                )
                + "\n"
            ).encode()
            for n in names
        )
    )
    raw = json.dumps(
        [
            dict(
                video=n,
                annotation_id=i,
                num_segments=1,
                description="fixture",
                times=[[0, 0]],
            )
            for i, n in enumerate(names)
        ]
    ).encode()
    index = c.metadata_only(raw, hashlib.sha256(raw).hexdigest())
    manifest = dict(
        receipt_sha256=c.digest(receipts),
        files={"media.json": c.digest(prepared / "media.json")},
        official_metadata_sha256={"test_data.json": index["official_test_sha256"]},
        excluded_decoder_media_sha256=[],
        counts={"test": dict(source_video_ids=21, unique_media=1)},
    )
    write(prepared / "manifest.json", manifest)
    return prepared, receipts, index, manifest


def test_twenty_coalesced_aliases_preserve_every_query(alias_fixture):
    prepared, receipts, index, _ = alias_fixture
    contract = c.build_contract(prepared, receipts, index)
    assert contract["source_video_ids"] == 21 and contract["unique_media"] == 1
    assert contract["query_count"] == 21
    assert len(set(r["id"] for r in contract["queries"])) == 21
    assert {r["video"] for r in contract["queries"]} == {"canonical"}
    assert set(contract["aliases"]) == {r["source_video"] for r in contract["queries"]}


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate_index",
        "geometry",
        "cross_split",
        "unknown",
        "duplicate_receipt",
    ],
)
def test_alias_contract_rejects_incomplete_or_ambiguous_sources(alias_fixture, fault):
    prepared, receipts, index, manifest = alias_fixture
    if fault == "missing":
        index["rows"].pop()
    elif fault == "duplicate_index":
        index["rows"][-1]["source_index"] = 0
    elif fault == "geometry":
        index["rows"][0]["num_segments"] = 2
    else:
        rows = [json.loads(line) for line in receipts.read_bytes().splitlines()]
        if fault == "cross_split":
            rows.append(dict(rows[0], split="train"))
        elif fault == "unknown":
            rows[0]["video"] = "unknown"
        else:
            rows.append(rows[0])
        receipts.write_bytes(b"".join((json.dumps(r) + "\n").encode() for r in rows))
        manifest["receipt_sha256"] = c.digest(receipts)
        (prepared / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        c.build_contract(prepared, receipts, index)
