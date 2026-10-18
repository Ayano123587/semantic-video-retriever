"""Actual completed-parent metadata and val-only source safeguards."""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import kaggle_phase1_validation_ranks as transport
from scripts import phase1_validation_ranks as worker
from scripts import stage_phase1_validation_ranks as stager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(os.environ.get("VIDEO_PRIVATE", "D:/CodexData/portfolio/quant-research-lab-private"))


def actual(name):
    path = PRIVATE / name
    if not path.is_file():
        pytest.skip("Actual completed private evidence unavailable")
    return path.read_bytes()


def tree(name):
    return ast.parse((ROOT / "scripts" / name).read_text())


def test_actual_complete_parent_and_all3132_metrics():
    raw = actual("video-phase1-full-validation-parent-freeze-v2.json")
    assert hashlib.sha256(raw).hexdigest() == stager.PARENT_FREEZE_SHA
    freeze = json.loads(raw)
    stager.validate_parent(freeze)
    result_raw = actual("video-phase1-portable-training-v1/publication-v1/terminal-reconciliation-v1/results.json")
    assert hashlib.sha256(result_raw).hexdigest() == freeze["results_sha256"]
    result = json.loads(result_raw)
    assert result["completed_fits"] == result["required_fits"] == 9
    assert result["selected"] == freeze["selected"] == "frozen_clip"
    assert result["selected_validation"]["queries"] == 3132
    assert result["selected_validation"]["candidate_moments"] == 15927
    worker.metrics_equal(result["selected_validation"], result["baseline"])


def test_actual_final_header_is_done_not_a_partial_window():
    freeze = json.loads(actual("video-phase1-full-validation-parent-freeze-v2.json"))
    raw = actual("video-phase1-portable-training-v1/publication-v1/terminal-reconciliation-v1/checkpoint-000005.json")
    assert hashlib.sha256(raw).hexdigest() == freeze["final_header_sha256"]
    header = json.loads(raw)
    assert header["stage"] == "done" and header["completed_seeds"] == 9
    assert header["snapshot_sha256"] == freeze["final_snapshot_sha256"]


def test_only_validation_native_arrays_are_loaded():
    calls = [node for node in ast.walk(tree("phase1_validation_ranks.py")) if isinstance(node, ast.Call)]
    loads = [node for node in calls if isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "np" and node.func.attr == "load"]
    assert len(loads) == 2
    assert {node.args[0].right.value for node in loads} == {"val-text.npy", "val-moments.npy"}
    assert all(any(key.arg == "allow_pickle" and key.value.value is False for key in node.keywords) for node in loads)


def test_original_scoring_and_evaluation_without_training_calls():
    calls = [node for node in ast.walk(tree("phase1_validation_ranks.py")) if isinstance(node, ast.Call)]
    assert sum(isinstance(node.func, ast.Name) and node.func.id == "score_policy" for node in calls) == 1
    assert sum(isinstance(node.func, ast.Name) and node.func.id == "evaluate" for node in calls) == 1
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in {"step", "backward", "forward", "train"} for node in calls)
    argsort = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "argsort")
    assert any(key.arg == "kind" and key.value.value == "stable" for key in argsort.keywords)


def test_full_raw_guard_precedes_native_import():
    function = next(node for node in tree("phase1_validation_ranks.py").body
                    if isinstance(node, ast.FunctionDef) and node.name == "export")
    native = min(node.lineno for node in function.body if isinstance(node, ast.Import))
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == "guard"
               and node.lineno < native for node in function.body)


def test_actual_template_lf_roundtrip():
    raw = (ROOT / "scripts/kaggle_phase1_validation_ranks.py").read_bytes().replace(b"\r\n", b"\n")
    token = b"EXPECTED_PHASE1_RANK_PLAN_SHA256" + b" = None"
    pin = hashlib.sha256(actual("video-phase1-full-validation-parent-freeze-v2.json")).hexdigest()
    binding = ("EXPECTED_PHASE1_RANK_PLAN_SHA256 = " + repr(pin)).encode()
    assert raw.count(token) == 1
    rendered = raw.replace(token, binding)
    assert raw.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n").replace(token, binding) == rendered
    ast.parse(rendered)


def test_rank_replay_is_plain_metadata_only():
    function = next(node for node in tree("phase1_validation_ranks.py").body
                    if isinstance(node, ast.FunctionDef) and node.name == "replay_metrics")
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function))
    assert not any(isinstance(node, ast.Name) and node.id in {"np", "torch"} for node in ast.walk(function))


def test_actual_source_aliases_resolve_only_external_path(tmp_path):
    proof = json.loads(actual("video-phase1-rank-source-path-proof-v1.json"))
    freeze = json.loads(actual("video-phase1-full-validation-parent-freeze-v2.json"))
    source_raw = Path(proof["source_path"]).read_bytes()
    cache_raw = Path(proof["cache_path"]).read_bytes()
    assert hashlib.sha256(source_raw).hexdigest() == freeze["source_manifest_sha256"]
    assert hashlib.sha256(cache_raw).hexdigest() == freeze["cache_manifest_sha256"]
    source, cache = json.loads(source_raw), json.loads(cache_raw)
    aliases = [row for row in source["files"] if row["sha256"] == cache["source_manifest_sha256"]
               and Path(row["path"]).name == "manifest.json"]
    assert len(aliases) == 2
    for row in aliases:
        raw = (Path(proof["source_path"]).parent / row["flat_name"]).read_bytes()
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    selected = tmp_path / freeze["prepared_manifest_relative_path"]
    assert transport.resolve_prepared(tmp_path, source, cache, freeze) == selected.parent
    assert len(source["files"]) == freeze["source_transport_total_files"] == 41
    assert not any(Path(row["path"]).name in {"test.json", "test_data.json"} for row in source["files"])
    missing = dict(freeze, prepared_manifest_relative_path="missing/manifest.json")
    with pytest.raises(ValueError, match="Exactly one externally bound"):
        transport.resolve_prepared(tmp_path, source, cache, missing)
    wrong_pin = dict(freeze, prepared_manifest_sha256=freeze["cache_manifest_sha256"])
    with pytest.raises(ValueError, match="Exactly one externally bound"):
        transport.resolve_prepared(tmp_path, source, cache, wrong_pin)
