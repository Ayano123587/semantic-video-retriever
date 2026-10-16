"""Tiny mechanics fixtures only; never open real DiDeMo test annotations."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import moment_evaluation as release


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir(); cache.mkdir()
    official = tmp_path / "fixture-only.json"
    write(official, [{"video": "fixture", "num_segments": 1,
                      "description": "A fixture query", "times": [[0, 0]] * 4}])
    expected = release.digest(official)
    monkeypatch.setattr(release, "HASHES", {"test_data.json": expected})
    for name, data in (("media.json", [{"video": "fixture", "split": "test", "num_segments": 1}]),
                       ("train.json", []), ("val.json", [])):
        write(source / name, data)
    write(source / "manifest.json", {"source_revision": release.REVISION,
          "files": {p.name: release.digest(p) for p in source.iterdir()},
          "official_metadata_sha256": {"test_data.json": expected},
          "counts": {"test": {"unique_media": 1}}})
    write(cache / "test-candidates.json", [{"video": "fixture", "moment": [0, 0]}])
    vector = np.zeros((1, 512), dtype=np.float32); vector[0, 0] = 1
    np.save(cache / "test-moments.npy", vector, allow_pickle=False)
    encoder = {"model_id": release.MODEL_ID, "revision": release.MODEL_REVISION,
               "weights_sha256": release.WEIGHTS_SHA256,
               "loading_info": {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}}
    write(cache / "manifest.json", {"files": {p.name: release.digest(p) for p in cache.iterdir()},
          "encoder": encoder, "smoke_only": False, "test_text_or_targets_encoded": False,
          "source_manifest_sha256": release.digest(source / "manifest.json")})
    checkpoint = tmp_path / "selected.pt"
    selected = {"schema": 1, "winner": "frozen_clip", "states": [], "test_targets_used": False,
                "encoder": encoder, "cache_manifest_sha256": release.digest(cache / "manifest.json")}
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
    report = release.release(source, cache, checkpoint, official, tmp_path / "report.json", encoder_factory=encoder)
    assert report["selected"]["corpus_consensus_r1_iou50"] == 1
    assert "A fixture query" not in json.dumps(report)
    with pytest.raises(ValueError, match="already attempted"):
        release.release(source, cache, checkpoint, official, tmp_path / "different.json", encoder_factory=encoder)


def test_marker_exists_before_target_parse_and_survives_failure(cohort, tmp_path, monkeypatch):
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
        release.release(source, cache, checkpoint, official, tmp_path / "report.json", encoder_factory=encoder)
    assert (source / release.MARKER).exists()


def test_unfrozen_and_changed_official_refuse_before_release(cohort, tmp_path):
    source, cache, checkpoint, official, encoder, _ = cohort
    with pytest.raises(ValueError, match="Unreadable JSON"):
        release.release(source, cache, checkpoint, official, tmp_path / "report.json", encoder_factory=encoder)
    release.freeze(source, cache, checkpoint)
    official.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed official"):
        release.release(source, cache, checkpoint, official, tmp_path / "report.json", encoder_factory=encoder)
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
        manifest["files"]["test-candidates.json"] = release.digest(cache / "test-candidates.json")
        write(cache / "manifest.json", manifest)
        selected["cache_manifest_sha256"] = release.digest(cache / "manifest.json")
        torch.save(selected, checkpoint)
    with pytest.raises(ValueError):
        release.freeze(source, cache, checkpoint)
    assert not (source / release.MARKER).exists()


def test_existing_report_does_not_release(cohort, tmp_path):
    source, cache, checkpoint, official, encoder, _ = cohort
    release.freeze(source, cache, checkpoint)
    output = tmp_path / "report.json"; output.write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        release.release(source, cache, checkpoint, official, output, encoder_factory=encoder)
    assert output.read_text() == "preserve"
    assert not (source / release.MARKER).exists()


def test_missing_available_source_queries_burns_release(cohort, tmp_path, monkeypatch):
    source, cache, checkpoint, official, encoder, _ = cohort
    release.freeze(source, cache, checkpoint)
    original = release.read_json
    monkeypatch.setattr(release, "read_json", lambda p: [] if Path(p) == official else original(p))
    with pytest.raises(ValueError, match="cover all available"):
        release.release(source, cache, checkpoint, official, tmp_path / "report.json", encoder_factory=encoder)
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


@pytest.mark.parametrize("value", [True, False, 0, 1, -0.1, 1.1, float("nan"),
                                  float("inf"), -float("inf"), "0.5", None, 10**400])
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
        torch.nn.init.normal_(model.up.weight, std=.1)
        states.append(model.state_dict())
    selected.update(winner="lr001", states=states, adapted_score_weight=alpha)
    torch.save(selected, checkpoint)
    policy = release.freeze(source, cache, checkpoint)
    assert policy["adapted_score_weight"] == alpha
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    x = torch.nn.functional.normalize(torch.randn(3, 512), dim=1)
    v = torch.nn.functional.normalize(torch.randn(7, 512), dim=1)
    expected = (1-alpha)*release.scores_for([], x, v) + alpha*release.scores_for(states, x, v)
    assert np.array_equal(release.policy_scores(saved, x, v), expected)


def test_legacy_weight_default_and_baseline_are_preserved(cohort):
    _, _, _, _, _, selected = cohort
    x = torch.nn.functional.normalize(torch.randn(2, 512), dim=1)
    v = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
    assert release.adapted_weight(selected) == 1.0
    baseline = release.scores_for([], x, v)
    for alpha in (0.0, .25, 1.0):
        selected["adapted_score_weight"] = alpha
        assert np.array_equal(release.policy_scores(selected, x, v), baseline)


def test_localization_checkpoint_family_and_weight_must_agree(cohort):
    source, cache, checkpoint, _, _, selected = cohort
    selected.update(winner="lambda0.5-alpha0.25", adapted_score_weight=.25,
                    states=[release.Adapter().state_dict() for _ in range(3)])
    torch.save(selected, checkpoint)
    identities, *_ = release.verified_inputs(source, cache, checkpoint)
    assert identities["adapted_score_weight"] == .25
    selected["adapted_score_weight"] = .5
    torch.save(selected, checkpoint)
    with pytest.raises(ValueError, match="label and adapted"):
        release.freeze(source, cache, checkpoint)
