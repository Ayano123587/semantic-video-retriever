"""Tiny mechanics fixtures only; never open real DiDeMo test annotations."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import moment_evaluation as release
from src.data import test_cohort as identities


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    official = tmp_path / "fixture-only.json"
    write(
        official,
        [
            {
                "video": "fixture",
                "annotation_id": 1,
                "num_segments": 1,
                "description": "A fixture query",
                "times": [[0, 0]] * 4,
            }
        ],
    )
    expected = release.digest(official)
    monkeypatch.setattr(release, "HASHES", {"test_data.json": expected})
    for name, data in (
        (
            "media.json",
            [
                {
                    "video": "fixture",
                    "split": "test",
                    "num_segments": 1,
                    "sha256": "a" * 64,
                }
            ],
        ),
        ("train.json", []),
        ("val.json", []),
    ):
        write(source / name, data)
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        json.dumps(
            dict(video="fixture", split="test", status="complete", sha256="a" * 64)
        )
        + "\n",
        newline="\n",
    )
    monkeypatch.setattr(identities, "FULL_COUNTS", (1, 1))
    write(
        source / "manifest.json",
        {
            "source_revision": release.REVISION,
            "files": {p.name: release.digest(p) for p in source.iterdir()},
            "official_metadata_sha256": {"test_data.json": expected},
            "counts": {"test": {"unique_media": 1, "source_video_ids": 1}},
            "receipt_sha256": release.digest(receipts),
            "excluded_decoder_media_sha256": [],
        },
    )
    index = identities.metadata_only(official.read_bytes(), expected)
    write(
        source / identities.CONTRACT, identities.build_contract(source, receipts, index)
    )
    write(cache / "test-candidates.json", [{"video": "fixture", "moment": [0, 0]}])
    vector = np.zeros((1, 512), dtype=np.float32)
    vector[0, 0] = 1
    np.save(cache / "test-moments.npy", vector, allow_pickle=False)
    encoder = {
        "model_id": release.MODEL_ID,
        "revision": release.MODEL_REVISION,
        "weights_sha256": release.WEIGHTS_SHA256,
        "loading_info": {
            k: []
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        },
    }
    write(
        cache / "manifest.json",
        {
            "files": {p.name: release.digest(p) for p in cache.iterdir()},
            "encoder": encoder,
            "smoke_only": False,
            "test_text_or_targets_encoded": False,
            "source_manifest_sha256": release.digest(source / "manifest.json"),
        },
    )
    checkpoint = tmp_path / "selected.pt"
    selected = {
        "schema": 1,
        "winner": "frozen_clip",
        "states": [],
        "test_targets_used": False,
        "encoder": encoder,
        "cache_manifest_sha256": release.digest(cache / "manifest.json"),
    }
    torch.save(selected, checkpoint)

    class FixtureEncoder:
        def __init__(self, device):
            assert device == "cpu"
            self.receipt = encoder

        def texts(self, queries):
            return np.repeat(vector, len(queries), axis=0)

    return source, cache, checkpoint, official, FixtureEncoder, selected


def test_one_shot_marker_blocks_another_output(cohort, tmp_path):
    source, cache, checkpoint, official, encoder, _ = cohort
    release.freeze(source, cache, checkpoint)
    report = release.release(
        source,
        cache,
        checkpoint,
        official,
        tmp_path / "report.json",
        encoder_factory=encoder,
    )
    assert report["selected"]["corpus_consensus_r1_iou50"] == 1
    assert "A fixture query" not in json.dumps(report)
    with pytest.raises(ValueError, match="already attempted"):
        release.release(
            source,
            cache,
            checkpoint,
            official,
            tmp_path / "different.json",
            encoder_factory=encoder,
        )


def test_missing_prospective_contract_cannot_bypass_release(cohort):
    source, cache, checkpoint, *_ = cohort
    (source / identities.CONTRACT).unlink()
    with pytest.raises(ValueError, match="Unreadable"):
        release.freeze(source, cache, checkpoint)
    assert not (source / release.MARKER).exists()


@pytest.mark.parametrize("preferred", ["phase1", "phase2"])
def test_global_chooser_output_integrates_with_freeze_and_release(
    cohort, tmp_path, monkeypatch, preferred
):
    """Tiny mocked validation mechanics, followed by the actual release contract."""
    from src.models import research_followup as r

    source, cache, _, official, encoder, selected = cohort
    baseline = dict(
        corpus_consensus_r1_iou50=float(np.float32(0.1)), within_video_official_miou=0.5
    )
    p = dict(
        global_selection=dict(
            phases=["phase1", "phase2"],
            criterion="highest eligible validation corpus_consensus_r1_iou50",
            ties="frozen_clip then phase1 then phase2; fixed within-phase policy order",
            test_access=False,
        ),
        source_manifest_sha256=release.digest(cache / "manifest.json"),
        baseline=baseline,
    )
    protocol = tmp_path / "protocol.json"
    write(protocol, p)
    v = torch.zeros(1, 512)
    v[0, 0] = 1
    data = {"val": dict(x=v, v=v, candidates=[], queries=[])}
    manifest = {"encoder": selected["encoder"]}
    monkeypatch.setattr(r, "checked_protocol", lambda *args: (data, manifest, p))
    monkeypatch.setattr(
        r, "scores_for", lambda *args: np.asarray([[0.1]], dtype=np.float32)
    )
    monkeypatch.setattr(
        r,
        "evaluate",
        lambda scores, *args: dict(
            corpus_consensus_r1_iou50=float(scores[0, 0]),
            within_video_official_miou=0.5,
        ),
    )
    monkeypatch.setattr(
        r,
        "score_policy",
        lambda bundle, *args: np.asarray(
            [
                [
                    (
                        0.5
                        if preferred == "phase2"
                        and bundle["adapted_score_weight"] == 0.25
                        else 0.4
                    )
                ]
            ],
            dtype=np.float32,
        ),
    )
    directories = []
    hashes = []
    for phase, winner, alpha, primary in [
        ("phase1", "lr001", 1.0, 0.4),
        ("phase2", "lambda0-alpha0.25", 0.25, 0.5 if preferred == "phase2" else 0.4),
    ]:
        path = tmp_path / phase
        path.mkdir()
        directories.append(path)
        metrics = dict(
            corpus_consensus_r1_iou50=float(np.float32(primary)),
            within_video_official_miou=0.5,
        )
        bundle = dict(
            selected,
            winner=winner,
            adapted_score_weight=alpha,
            states=[release.Adapter().state_dict() for _ in range(3)],
            validation=metrics,
        )
        torch.save(bundle, path / "selected.pt")
        report = dict(
            selected=winner,
            selected_validation=metrics,
            baseline=baseline,
            cache_manifest_sha256=p["source_manifest_sha256"],
            selected_checkpoint_sha256=release.digest(path / "selected.pt"),
            protocol_sha256=release.digest(protocol),
            test_targets_used=False,
        )
        write(path / "results.json", report)
        hashes.append(release.digest(path / "results.json"))
    choice = tmp_path / "choice"
    result = r.choose(
        source, cache, *directories, choice, protocol, release.digest(protocol), *hashes
    )
    assert result["selected"].startswith(preferred + ":")
    release.freeze(source, cache, choice / "selected.pt")
    report = release.release(
        source,
        cache,
        choice / "selected.pt",
        official,
        tmp_path / "report.json",
        encoder_factory=encoder,
    )
    assert report["selected"]["queries"] == 1


def test_marker_exists_before_target_parse_and_survives_failure(
    cohort, tmp_path, monkeypatch
):
    source, cache, checkpoint, official, encoder, _ = cohort
    release.freeze(source, cache, checkpoint)
    original = release.read_json

    def fail_on_gold(path):
        if Path(path) == official:
            assert (source / release.MARKER).exists()
            raise ValueError("Deliberate parser failure")
        return original(path)

    monkeypatch.setattr(release, "read_json", fail_on_gold)
    with pytest.raises(ValueError, match="parser failure"):
        release.release(
            source,
            cache,
            checkpoint,
            official,
            tmp_path / "report.json",
            encoder_factory=encoder,
        )
    assert (source / release.MARKER).exists()


def test_unfrozen_and_changed_official_refuse_before_release(cohort, tmp_path):
    source, cache, checkpoint, official, encoder, _ = cohort
    with pytest.raises(ValueError, match="Unreadable JSON"):
        release.release(
            source,
            cache,
            checkpoint,
            official,
            tmp_path / "report.json",
            encoder_factory=encoder,
        )
    release.freeze(source, cache, checkpoint)
    official.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed official"):
        release.release(
            source,
            cache,
            checkpoint,
            official,
            tmp_path / "report.json",
            encoder_factory=encoder,
        )
    assert not (source / release.MARKER).exists()


@pytest.mark.parametrize("fault", ["cache", "checkpoint", "geometry", "states"])
def test_identity_and_geometry_fail_closed(cohort, fault):
    source, cache, checkpoint, official, encoder, selected = cohort
    if fault == "cache":
        (cache / "test-candidates.json").write_text("[]")
    elif fault == "checkpoint":
        selected["cache_manifest_sha256"] = "0" * 64
        torch.save(selected, checkpoint)
    elif fault == "states":
        selected.update(winner="lr001", states=[release.Adapter().state_dict()])
        torch.save(selected, checkpoint)
    else:
        write(cache / "test-candidates.json", [{"video": "other", "moment": [0, 0]}])
        manifest = release.read_json(cache / "manifest.json")
        manifest["files"]["test-candidates.json"] = release.digest(
            cache / "test-candidates.json"
        )
        write(cache / "manifest.json", manifest)
        selected["cache_manifest_sha256"] = release.digest(cache / "manifest.json")
        torch.save(selected, checkpoint)
    with pytest.raises(ValueError):
        release.freeze(source, cache, checkpoint)
    assert not (source / release.MARKER).exists()


def test_existing_report_does_not_release(cohort, tmp_path):
    source, cache, checkpoint, official, encoder, _ = cohort
    release.freeze(source, cache, checkpoint)
    output = tmp_path / "report.json"
    output.write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        release.release(
            source, cache, checkpoint, official, output, encoder_factory=encoder
        )
    assert output.read_text() == "preserve"
    assert not (source / release.MARKER).exists()


def test_missing_available_source_queries_burns_release(cohort, tmp_path, monkeypatch):
    source, cache, checkpoint, official, encoder, _ = cohort
    release.freeze(source, cache, checkpoint)
    original = release.read_json
    monkeypatch.setattr(
        release, "read_json", lambda p: [] if Path(p) == official else original(p)
    )
    with pytest.raises(ValueError, match="cover all available"):
        release.release(
            source,
            cache,
            checkpoint,
            official,
            tmp_path / "report.json",
            encoder_factory=encoder,
        )
    assert (source / release.MARKER).exists()


def test_nonfinite_adapter_weights_refused_before_labels(cohort):
    source, cache, checkpoint, _, _, selected = cohort
    states = [release.Adapter().state_dict() for _ in range(3)]
    states[1]["up.weight"][0, 0] = float("nan")
    selected.update(winner="lr001", states=states)
    torch.save(selected, checkpoint)
    with pytest.raises(ValueError, match="nonfinite adapter"):
        release.freeze(source, cache, checkpoint)
    assert not (source / release.MARKER).exists()


def test_test_query_cache_refused_before_labels(cohort):
    source, cache, checkpoint, _, _, _ = cohort
    (cache / "test-queries.json").write_text("[]")
    with pytest.raises(ValueError, match="must remain absent"):
        release.freeze(source, cache, checkpoint)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        1,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        -float("inf"),
        "0.5",
        None,
        10**400,
    ],
)
def test_mixture_weight_rejects_malformed_values_before_labels(cohort, value):
    source, cache, checkpoint, _, _, selected = cohort
    selected["adapted_score_weight"] = value
    torch.save(selected, checkpoint)
    with pytest.raises(ValueError, match="finite float"):
        release.freeze(source, cache, checkpoint)
    assert not (source / release.MARKER).exists()


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_saved_mixture_exact_score_replay_and_policy_identity(cohort, alpha):
    source, cache, checkpoint, _, _, selected = cohort
    torch.manual_seed(17)
    states = []
    for _ in range(3):
        model = release.Adapter()
        torch.nn.init.normal_(model.up.weight, std=0.1)
        states.append(model.state_dict())
    selected.update(winner="lr001", states=states, adapted_score_weight=alpha)
    torch.save(selected, checkpoint)
    policy = release.freeze(source, cache, checkpoint)
    assert policy["adapted_score_weight"] == alpha
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    x = torch.nn.functional.normalize(torch.randn(3, 512), dim=1)
    v = torch.nn.functional.normalize(torch.randn(7, 512), dim=1)
    expected = (1 - alpha) * release.scores_for([], x, v) + alpha * release.scores_for(
        states, x, v
    )
    assert np.array_equal(release.policy_scores(saved, x, v), expected)


def test_legacy_weight_default_and_baseline_are_preserved(cohort):
    _, _, _, _, _, selected = cohort
    x = torch.nn.functional.normalize(torch.randn(2, 512), dim=1)
    v = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
    assert release.adapted_weight(selected) == 1.0
    baseline = release.scores_for([], x, v)
    for alpha in (0.0, 0.25, 1.0):
        selected["adapted_score_weight"] = alpha
        assert np.array_equal(release.policy_scores(selected, x, v), baseline)


def test_localization_checkpoint_family_and_weight_must_agree(cohort):
    source, cache, checkpoint, _, _, selected = cohort
    selected.update(
        winner="lambda0.5-alpha0.25",
        adapted_score_weight=0.25,
        states=[release.Adapter().state_dict() for _ in range(3)],
    )
    torch.save(selected, checkpoint)
    identities, *_ = release.verified_inputs(source, cache, checkpoint)
    assert identities["adapted_score_weight"] == 0.25
    selected["adapted_score_weight"] = 0.5
    torch.save(selected, checkpoint)
    with pytest.raises(ValueError, match="label and adapted"):
        release.freeze(source, cache, checkpoint)


@pytest.mark.parametrize(
    "winner,alpha", [("phase1:lr001", 1.0), ("phase2:lambda0.5-alpha0.25", 0.25)]
)
def test_global_namespaced_policy_freezes_and_releases(cohort, tmp_path, winner, alpha):
    source, cache, checkpoint, official, encoder, selected = cohort
    selected.update(
        winner=winner,
        adapted_score_weight=alpha,
        global_protocol_sha256="a" * 64,
        states=[release.Adapter().state_dict() for _ in range(3)],
    )
    torch.save(selected, checkpoint)
    policy = release.freeze(source, cache, checkpoint)
    assert policy["winner"] == winner
    report = release.release(
        source,
        cache,
        checkpoint,
        official,
        tmp_path / "report.json",
        encoder_factory=encoder,
    )
    assert report["selected"]["queries"] == 1


@pytest.mark.parametrize(
    "winner,protocol,alpha",
    [
        ("phase3:lr001", "a" * 64, 1.0),
        ("phase1:lambda0.5-alpha0.25", "a" * 64, 0.25),
        ("phase2:lr001", "a" * 64, 1.0),
        ("phase1:lr001", "bad", 1.0),
        ("phase1:lr001", "a" * 64, 0.5),
    ],
)
def test_invalid_global_family_refused_before_gold(cohort, winner, protocol, alpha):
    source, cache, checkpoint, _, _, selected = cohort
    selected.update(
        winner=winner,
        global_protocol_sha256=protocol,
        adapted_score_weight=alpha,
        states=[release.Adapter().state_dict() for _ in range(3)],
    )
    torch.save(selected, checkpoint)
    with pytest.raises(ValueError):
        release.freeze(source, cache, checkpoint)
    assert not (source / release.MARKER).exists()


@pytest.mark.parametrize("fault", [None, "missing", "unknown", "duplicate"])
def test_full_alias_queries_follow_frozen_identity_contract(
    cohort, tmp_path, monkeypatch, fault
):
    from src.data import test_cohort as identities

    source, cache, checkpoint, official, encoder, selected = cohort
    names = ["fixture"] + [f"z-alias-{i}" for i in range(20)]
    rows = [
        dict(
            video=n,
            annotation_id=i,
            num_segments=1,
            description="Software query",
            times=[[0, 0]] * 4,
        )
        for i, n in enumerate(names)
    ]
    write(official, rows)
    monkeypatch.setattr(release, "HASHES", {"test_data.json": release.digest(official)})
    media = release.read_json(source / "media.json")
    media[0]["sha256"] = "a" * 64
    write(source / "media.json", media)
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        "".join(
            json.dumps(dict(video=n, split="test", status="complete", sha256="a" * 64))
            + "\n"
            for n in names
        ),
        newline="\n",
    )
    manifest = release.read_json(source / "manifest.json")
    manifest.update(
        receipt_sha256=release.digest(receipts), excluded_decoder_media_sha256=[]
    )
    manifest["official_metadata_sha256"]["test_data.json"] = release.digest(official)
    manifest["files"]["media.json"] = release.digest(source / "media.json")
    manifest["counts"]["test"]["source_video_ids"] = 21
    write(source / "manifest.json", manifest)
    monkeypatch.setattr(identities, "FULL_COUNTS", (21, 1))
    index = identities.metadata_only(official.read_bytes(), release.digest(official))
    write(
        source / identities.CONTRACT, identities.build_contract(source, receipts, index)
    )
    encoded = release.read_json(cache / "manifest.json")
    encoded["source_manifest_sha256"] = release.digest(source / "manifest.json")
    write(cache / "manifest.json", encoded)
    selected["cache_manifest_sha256"] = release.digest(cache / "manifest.json")
    torch.save(selected, checkpoint)
    policy = release.freeze(source, cache, checkpoint)
    assert policy["expected_test_queries"] == 21
    original = release.read_json
    if fault:
        altered = [dict(row) for row in rows]
        if fault == "missing":
            altered.pop()
        elif fault == "unknown":
            altered[-1]["video"] = "unrecognized"
        else:
            altered[-1] = dict(altered[0])
        monkeypatch.setattr(
            release,
            "read_json",
            lambda path: altered if Path(path) == official else original(path),
        )
        with pytest.raises(ValueError, match="identities differ"):
            release.release(
                source,
                cache,
                checkpoint,
                official,
                tmp_path / "report.json",
                encoder_factory=encoder,
            )
    else:
        report = release.release(
            source,
            cache,
            checkpoint,
            official,
            tmp_path / "report.json",
            encoder_factory=encoder,
        )
        assert report["selected"]["queries"] == 21
