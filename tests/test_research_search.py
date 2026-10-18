"""Tiny software arrays/states only; no benchmark, encoder download or training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import MODEL_ID, MODEL_REVISION
from src.models import research_search as r
from src.models.encoder import WEIGHTS_SHA256
from src.models.moment_training import Adapter


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def refresh(value):
    cache, prepared = value["cache"], value["prepared"]
    source = {
        "receipt_sha256": r.sha(value["receipts"]),
        "files": {"media.json": r.sha(prepared / "media.json")},
    }
    write(prepared / "manifest.json", source)
    environment = {key: "software fixture" for key in r.ENCODER_ENV_KEYS}
    environment["python"] = "software fixture"
    training_runtime = {
        "cpu": "software fixture",
        "threads": 2,
        "interop_threads": 2,
        "deterministic": True,
    }
    encoding_identity = {key: environment[key] for key in r.ENCODER_ENV_KEYS}
    encoding_identity["encoder"] = value["bundle"]["encoder"]
    encoded = {
        "status": "complete",
        "smoke_only": False,
        "test_text_or_targets_encoded": False,
        "source_manifest_sha256": r.sha(prepared / "manifest.json"),
        "encoder": value["bundle"]["encoder"],
        "encoding_identity": encoding_identity,
        "files": {
            path.name: r.sha(path)
            for path in cache.iterdir()
            if path.name != "manifest.json"
        },
    }
    write(cache / "manifest.json", encoded)
    protocol_path = value["root"] / "protocol.json"
    write(
        protocol_path,
        {
            "source_manifest_sha256": r.sha(cache / "manifest.json"),
            "encoding_identity": encoding_identity,
        },
    )
    protocol_sha256 = r.sha(protocol_path)
    value["bundle"]["global_protocol_sha256"] = protocol_sha256
    value["bundle"]["cache_manifest_sha256"] = r.sha(cache / "manifest.json")
    torch.save(value["bundle"], value["checkpoint"])
    inventory_rows = {}
    phase_evidence = []
    phase_reports = {}
    for phase, fits in (("phase1", 9), ("phase2", 12)):
        phase_report_path = value["root"] / (phase + "-report.json")
        write(
            phase_report_path,
            {
                "completed_fits": fits,
                "required_fits": fits,
                "continuation_identity": {"runtime": training_runtime},
            },
        )
        report_hash = r.sha(phase_report_path)
        phase_reports[phase] = {"path": str(phase_report_path), "sha256": report_hash}
        path = value["root"] / (phase + "-inventory.json")
        write(
            path,
            {
                "phase": phase,
                "completed_fits": fits,
                "source_manifest_sha256": "b" * 64,
                "learned_files_deserialized": False,
                "environment": environment,
                "validation_queries": 3132,
                "test_targets_used": False,
                "cache_manifest_sha256": r.sha(cache / "manifest.json"),
                "protocol_sha256": protocol_sha256,
                "files": {"results.json": report_hash},
            },
        )
        inventory_rows[phase] = {"path": str(path), "sha256": r.sha(path)}
        phase_evidence.append({"phase": phase, "report_sha256": report_hash})
    report = {
        "selected": value["bundle"]["winner"],
        "selected_checkpoint_sha256": r.sha(value["checkpoint"]),
        "test_targets_used": False,
        "baseline": {"queries": 3132},
        "selected_validation": {"queries": 3132},
        "phases": phase_evidence,
        "cache_manifest_sha256": r.sha(cache / "manifest.json"),
        "protocol_sha256": protocol_sha256,
    }
    report_path = value["root"] / "global-results.json"
    write(report_path, report)
    dispatch_path = value["root"] / "global-dispatch.json"
    write(
        dispatch_path,
        {
            "schema": "both_phase_global_selection_dispatch_v1",
            "result": report,
            "validation_queries": 3132,
            "test_targets_used": False,
            "inventory_sha256": {
                phase: row["sha256"] for phase, row in inventory_rows.items()
            },
        },
    )
    repo = Path(r.__file__).resolve().parents[2]
    approval = {
        "schema": "globally_selected_video_query_policy_v1",
        "test_targets_used_for_selection": False,
        "source_manifest_sha256": "b" * 64,
        "source_files": {name: r.sha(repo / name) for name in r.SOURCE_FILES},
        "report_path": str(report_path),
        "report_sha256": r.sha(report_path),
        "dispatch_path": str(dispatch_path),
        "dispatch_sha256": r.sha(dispatch_path),
        "inventories": inventory_rows,
        "phase_reports": phase_reports,
        "protocol_path": str(protocol_path),
        "environment": environment,
        "training_runtime": training_runtime,
        "checkpoint_sha256": r.sha(value["checkpoint"]),
        "cache_manifest_sha256": r.sha(cache / "manifest.json"),
        "prepared_manifest_sha256": r.sha(prepared / "manifest.json"),
        "protocol_sha256": protocol_sha256,
    }
    write(value["approval"], approval)
    value["approval_sha256"] = r.sha(value["approval"])
    value["observed_runtime"] = {
        "environment": environment,
        "training_runtime": training_runtime,
    }
    return value


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        r,
        "FULL_COUNTS",
        {
            "train": {"videos": 1, "source_ids": 2, "moments": 3},
            "val": {"videos": 1, "source_ids": 1, "moments": 1},
            "test": {"videos": 1, "source_ids": 1, "moments": 1},
        },
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    media = []
    frames = []
    receipts = []
    for split, dimension, segments, duration in (
        ("train", 0, 2, 7.25),
        ("val", 0, 1, 4.0),
        ("test", 1, 1, 5.0),
    ):
        video = split + "-video"
        source_sha = {"train": "3", "val": "4", "test": "5"}[split] * 64
        media.append(
            {
                "split": split,
                "video": video,
                "num_segments": segments,
                "sha256": source_sha,
                "license": "CC BY 2.0",
                "original_url": "https://example.invalid/" + video,
            }
        )
        frames.append(
            {
                "split": split,
                "video": video,
                "source_sha256": source_sha,
                "duration": duration,
            }
        )
        receipts.append(
            {"split": split, "video": video, "sha256": source_sha, "status": "complete"}
        )
        if split == "train":
            receipts.append(
                {
                    "split": split,
                    "video": "train-alias",
                    "sha256": source_sha,
                    "status": "complete",
                }
            )
        candidates = [
            {"video": video, "moment": [start, end]}
            for start in range(segments)
            for end in range(start, segments)
        ]
        write(cache / (split + "-candidates.json"), candidates)
        vectors = np.zeros((len(candidates), 512), dtype=np.float32)
        for i in range(len(candidates)):
            vectors[i, (dimension + i) % 512] = 1
        np.save(cache / (split + "-moments.npy"), vectors, allow_pickle=False)
    write(prepared / "media.json", media)
    write(cache / "frames.json", frames)
    receipt_path = tmp_path / "receipts.jsonl"
    receipt_path.write_text("".join(json.dumps(row) + "\n" for row in receipts))
    encoder = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "loading_info": {},
    }
    bundle = {
        "schema": 1,
        "states": [],
        "adapted_score_weight": 0.0,
        "winner": "frozen_clip",
        "global_protocol_sha256": "a" * 64,
        "encoder": encoder,
        "test_targets_used": False,
    }
    value = {
        "root": tmp_path,
        "cache": cache,
        "prepared": prepared,
        "receipts": receipt_path,
        "checkpoint": tmp_path / "selected.pt",
        "approval": tmp_path / "approval.json",
        "bundle": bundle,
    }
    refresh(value)
    monkeypatch.setattr(r, "runtime_observation", lambda: value["observed_runtime"])
    return value


def index(value, split="all"):
    return r.ResearchSearch(
        value["cache"],
        value["prepared"],
        value["receipts"],
        value["checkpoint"],
        value["approval"],
        value["approval_sha256"],
        split,
    )


def test_all_split_stable_ties_source_aliases_and_real_duration(artifact):
    selected = index(artifact)
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1
    matches = selected.search_vector(query, 5)
    assert [row["split"] for row in matches[:2]] == ["train", "val"]
    assert matches[0]["source_video_ids"] == ["train-alias", "train-video"]
    assert matches[0]["source_sha256"] == "3" * 64
    assert matches[0]["license"] == "CC BY 2.0"
    tail = next(
        row for row in matches if row["split"] == "train" and row["start_seconds"] == 5
    )
    assert tail["end_seconds"] == 7.25
    assert {row["split"] for row in matches} == {"train", "val", "test"}


@pytest.mark.parametrize("phase,alpha", [("phase1", 1.0), ("phase2", 0.25)])
def test_actual_global_wrapper_routes_equal_seed_ensemble_and_mixture(
    artifact, phase, alpha
):
    states = []
    for bias in (2.0, 3.0, 4.0):
        model = Adapter()
        with torch.no_grad():
            model.up.bias[1] = bias
        states.append(model.state_dict())
    artifact["bundle"].update(
        states=states,
        adapted_score_weight=alpha,
        winner="phase1:lr001" if phase == "phase1" else "phase2:lambda0.5-alpha0.25",
    )
    refresh(artifact)
    selected = index(artifact)
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1
    from src.models.localization_followup import score_policy

    expected = []
    for split, part in selected.indexes.items():
        scores = score_policy(
            part.bundle, torch.from_numpy(query[None]), part.features
        )[0]
        expected.extend((float(score), split, i) for i, score in enumerate(scores))
    expected.sort(key=lambda row: -row[0])
    actual = selected.search_vector(query, 5)
    assert np.allclose(
        [row["score"] for row in actual], [row[0] for row in expected], atol=0, rtol=0
    )
    assert [row["split"] for row in actual] == [row[1] for row in expected]


@pytest.mark.parametrize(
    "fault", ["checkpoint", "cache", "report", "inventory", "source"]
)
def test_tamper_blocks_before_checkpoint_or_array_loader(artifact, monkeypatch, fault):
    if fault == "checkpoint":
        artifact["checkpoint"].write_bytes(b"changed")
    elif fault == "cache":
        (artifact["cache"] / "test-moments.npy").write_bytes(b"changed")
    elif fault == "report":
        (artifact["root"] / "global-results.json").write_bytes(b"changed")
    elif fault == "inventory":
        (artifact["root"] / "phase2-inventory.json").write_bytes(b"changed")
    else:
        approval = json.loads(artifact["approval"].read_bytes())
        approval["source_files"][r.SOURCE_FILES[0]] = "0" * 64
        write(artifact["approval"], approval)
        artifact["approval_sha256"] = r.sha(artifact["approval"])
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_a, **_k: pytest.fail("Checkpoint deserialized before authentication"),
    )
    monkeypatch.setattr(
        np, "load", lambda *_a, **_k: pytest.fail("Array parsed before authentication")
    )
    with pytest.raises(ValueError):
        index(artifact)


@pytest.mark.parametrize("fault", ["geometry", "frames", "aliases", "attribution"])
def test_authenticated_malformed_full_corpus_rejected(artifact, fault):
    if fault == "geometry":
        path = artifact["cache"] / "train-candidates.json"
        rows = json.loads(path.read_bytes())
        rows[-1]["moment"] = [0, 1]
        write(path, rows)
    elif fault == "frames":
        path = artifact["cache"] / "frames.json"
        rows = json.loads(path.read_bytes())
        rows.append(rows[0])
        write(path, rows)
    elif fault == "aliases":
        path = artifact["receipts"]
        rows = path.read_text().splitlines()
        path.write_text("\n".join(rows[1:]) + "\n")
    else:
        path = artifact["prepared"] / "media.json"
        rows = json.loads(path.read_bytes())
        rows[0]["license"] = ""
        write(path, rows)
    refresh(artifact)
    with pytest.raises(ValueError):
        index(artifact)


def test_invalid_query_is_rejected_before_encoder_and_invalid_vector_before_scoring(
    artifact,
):
    selected = index(artifact)

    class NeverEncoder:
        @property
        def receipt(self):
            pytest.fail("Invalid query accessed encoder")

    with pytest.raises(ValueError, match="Nonempty"):
        selected.search("   ", encoder=NeverEncoder())
    with pytest.raises(ValueError):
        selected.search_vector(np.zeros(512, dtype=np.float32))
    with pytest.raises(ValueError):
        selected.search_vector(np.ones(512, dtype=np.float32), top_k=0)


@pytest.mark.parametrize(
    "field", ["torch", "numpy", "cpu_identity", "transformers", "pillow"]
)
def test_numerical_encoder_mismatch_blocks_before_model_or_array_load(
    artifact, monkeypatch, field
):
    import copy

    observed = copy.deepcopy(artifact["observed_runtime"])
    observed["environment"][field] = "mismatched runtime fixture"
    monkeypatch.setattr(r, "runtime_observation", lambda: observed)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_a, **_k: pytest.fail("Model loaded before runtime rejection"),
    )
    monkeypatch.setattr(
        np,
        "load",
        lambda *_a, **_k: pytest.fail("Native array parsed before runtime rejection"),
    )
    with pytest.raises(ValueError, match="runtime required before native"):
        index(artifact)


def test_training_runtime_mismatch_blocks_before_model_load(artifact, monkeypatch):
    import copy

    observed = copy.deepcopy(artifact["observed_runtime"])
    observed["training_runtime"]["threads"] = 99
    monkeypatch.setattr(r, "runtime_observation", lambda: observed)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_a, **_k: pytest.fail("Model loaded before training-runtime rejection"),
    )
    with pytest.raises(ValueError, match="runtime required before native"):
        index(artifact)


def test_query_source_closure_covers_project_runtime_dependencies():
    import ast

    root = Path(r.__file__).resolve().parents[2]
    assert {
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
    }.issubset(r.SOURCE_FILES)
    assert r.query_source_files({}) == r.SOURCE_FILES
    assert r.query_source_files({"schema": r.RECOVERED_POLICY_SCHEMA}) == r.RECOVERED_SOURCE_FILES
    for name in r.RECOVERED_SOURCE_FILES:
        for node in ast.walk(ast.parse((root / name).read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "src."
            ):
                relative = node.module.replace(".", "/")
                target = relative + ".py"
                if not (root / target).exists():
                    target = relative + "/__init__.py"
                assert target in r.RECOVERED_SOURCE_FILES, (name, target)


def test_frozen_cli_controls_initialize_before_runtime_and_native_loading(
    artifact, monkeypatch
):
    controls = {}
    monkeypatch.setattr(
        torch, "set_num_threads", lambda count: controls.update(threads=count)
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled: controls.update(deterministic=enabled),
    )

    def initialized_runtime():
        assert controls == {"threads": 2, "deterministic": True}
        return artifact["observed_runtime"]

    monkeypatch.setattr(r, "runtime_observation", initialized_runtime)
    selected = index(artifact)
    assert set(selected.indexes) == {"train", "val", "test"}


@pytest.mark.parametrize(
    "field,value", [("threads", 4), ("threads", True), ("deterministic", False)]
)
def test_unsupported_runtime_controls_are_not_compatibility_overrides(
    artifact, monkeypatch, field, value
):
    policy = json.loads(artifact["approval"].read_bytes())
    policy["training_runtime"][field] = value
    monkeypatch.setattr(
        torch,
        "set_num_threads",
        lambda *_: pytest.fail("Unsupported controls changed Torch state"),
    )
    with pytest.raises(ValueError, match="Only frozen"):
        r.initialize_runtime(policy)
