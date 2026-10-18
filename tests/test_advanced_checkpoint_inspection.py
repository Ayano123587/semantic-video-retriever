"""Actual archived metadata and source-order safety; no native host parsing."""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import stage_advanced_checkpoint_inspection as stager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(os.environ.get("VIDEO_PRIVATE", "D:/CodexData/portfolio/quant-research-lab-private"))


def actual(relative):
    path = PRIVATE / relative
    if not path.is_file():
        pytest.skip("Actual private artifact unavailable")
    return path.read_bytes()


def test_real_ten_prefix_and_original_six_hashes():
    freeze = json.loads(actual("video-advanced-phase2-parent-v1/checkpoint-dataset-v1/root-data-freeze-v1.json"))
    report = json.loads(actual("video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json"))
    contract = json.loads(actual("video-advanced-phase2-parent-v1/advanced-contract.json"))
    assert freeze["completed_seed_models"] == len(contract["completed_names"]) == 10
    for seed in report["phases"]["phase2"]["completed"]:
        assert freeze["files"][seed["name"]]["sha256"] == seed["sha256"]
    assert freeze["snapshot_sha256"] == contract["header"]["snapshot_sha256"]
    assert freeze["header_sha256"] == contract["header_sha256"]


def test_real_thirteen_remote_hashes():
    base = PRIVATE / "video-advanced-phase2-parent-v1/checkpoint-dataset-v1"
    freeze = json.loads(actual("video-advanced-phase2-parent-v1/checkpoint-dataset-v1/root-data-freeze-v1.json"))
    for name, entry in freeze["files"].items():
        raw = (base / "remote-files-v1" / name).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]


def test_worker_native_import_follows_guard_and_has_no_updates():
    tree = ast.parse((ROOT / "scripts/advanced_checkpoint_inspection.py").read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inspect_restore")
    native = next(node.lineno for node in worker.body if isinstance(node, ast.ImportFrom) and node.module == "src.models")
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == "guard"
               and node.lineno < native for node in worker.body)
    forbidden = {"forward", "backward", "step", "score_policy", "engine", "train"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in forbidden for node in ast.walk(worker))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "checked_resume" for node in ast.walk(worker))


def test_runner_bootstrap_timeout_and_final_guard():
    tree = ast.parse((ROOT / "scripts/kaggle_advanced_checkpoint_inspection.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    pip = next(node for node in calls if isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess")
    assert any(key.arg == "timeout" and isinstance(key.value, ast.Call)
               and key.value.func.id == "remaining" for key in pip.keywords)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    body = next(node for node in run.body if isinstance(node, ast.Try)).body
    position = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "inspect_restore")
    assert isinstance(body[position + 1], ast.Expr) and body[position + 1].value.func.id == "guard"


def test_real_origin_identity_is_not_rewritten():
    raw = actual("video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json")
    report = json.loads(raw)
    origin = report["phases"]["phase2"]["origin_identity"]
    identity = hashlib.sha256(json.dumps(origin, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    header = json.loads(actual("video-advanced-phase2-parent-v1/parent-header.json"))
    assert identity == header["identity_sha256"]


def test_actual_runner_template_lf_roundtrip():
    path = ROOT / "scripts/kaggle_advanced_checkpoint_inspection.py"
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    token = b"EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256" + b" = None"
    assert canonical.count(token) == 1
    # Actual source template and its Windows checkout representation must give
    # the same rendered program; all transport manifests still hash raw bytes.
    pin = hashlib.sha256(actual("video-advanced-phase2-parent-v1/checkpoint-dataset-v1/root-data-freeze-v1.json")).hexdigest()
    binding = ("EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256 = " + repr(pin)).encode()
    assert canonical.replace(token, binding) == canonical.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n").replace(token, binding)
    ast.parse(canonical.replace(token, binding))


def test_actual_parent_progress_diagnostic_precedes_state_assertion():
    tree = ast.parse((ROOT / "scripts/advanced_checkpoint_inspection.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    diagnostic = next(node for node in calls if isinstance(node.func, ast.Name)
                      and node.func.id == "diagnostic")
    assertion = next(node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "require" and any(isinstance(arg, ast.Constant)
                     and arg.value == "Exact actual newer parent progress required" for arg in node.args))
    assert diagnostic.lineno < assertion.lineno
    progress = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "actual_progress" for target in node.targets))
    keys = {key.value for key in progress.keys}
    assert {"stage", "batches", "completed_count", "active_present", "header_progress_matches_state"} <= keys
    assert not {"model", "optimizer", "rng", "permutation"} & keys


def test_actual_completed_boundary_contract_and_diagnostic():
    raw = actual("video-advanced-phase2-parent-v1/root-native-contract-v2.json")
    assert hashlib.sha256(raw).hexdigest() == "358bf6f334bec6dab320637bddc1164ba69e3962ef65a955a43422d5622268e5"
    contract = json.loads(raw)
    diagnostic_raw = actual("video-advanced-phase2-parent-v1/native-inspection-publication-v2/attempt-2/observation-20261017T183511/worker-status.json")
    assert hashlib.sha256(diagnostic_raw).hexdigest() == contract["actual_progress_diagnostic_sha256"]
    observed = json.loads(diagnostic_raw)["parent_progress"]
    assert observed == contract["actual_progress"]
    assert observed["active_present"] is False and observed["active_epoch"] is observed["active_cursor"] is None
    assert observed["completed_count"] == 10 and observed["batches"] == 34122
    assert observed["grid"] == 3 and observed["seed_index"] == 1 and observed["header_progress_matches_state"]
    old = json.loads(actual("video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json"))
    assert old["phases"]["phase2"]["origin_spec"]["seeds"][observed["seed_index"]] == contract["next_seed"] == 1337
    freeze = json.loads(actual("video-advanced-phase2-parent-v1/checkpoint-dataset-v1/root-data-freeze-v1.json"))
    stager.validate_contract(contract, freeze)


def test_boundary_report_has_no_invented_active_state():
    tree = ast.parse((ROOT / "scripts/advanced_checkpoint_inspection.py").read_text())
    restoration = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "restoration" for target in node.targets))
    fields = {key.value: value for key, value in zip(restoration.keys, restoration.values, strict=True) if key is not None}
    for name in ("active", "active_optimizer", "optimizer_steps", "generator", "permutation", "epoch", "cursor"):
        assert isinstance(fields[name], ast.Constant) and fields[name].value is None
    source = (ROOT / "scripts/advanced_checkpoint_inspection.py").read_text()
    assert 'sum(row["optimizer_steps"] for row in restored)' in source
    assert 'expected_rng == completed[-1]["rng"]' in source
