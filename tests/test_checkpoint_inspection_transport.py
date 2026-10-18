"""Actual root metadata and source-boundary checks; no host native loading."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import interrupted_checkpoint_recovery as recovery
from scripts import kaggle_checkpoint_inspection as inspector
from scripts import stage_checkpoint_inspection as stager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parents[1] / "portfolio/quant-research-lab-private"
CONTRACT = PRIVATE / "video-portability-inspection-v1/root-inspection-contract-v1.json"
CONTRACT_SHA = "24eaee9fe28b2858b9e3e9451971750cb199a1b57e6c8f55e3809fae434dc6e6"


def actual_contract():
    if not CONTRACT.exists():
        pytest.skip("Actual root-authored inspection contract required")
    return recovery._json(CONTRACT, CONTRACT_SHA)


def test_actual_root_inspection_contract_and_inventory_pins():
    contract = actual_contract()
    inventory_pins = {}
    for phase in ("phase1", "phase2"):
        path = (
            PRIVATE
            / f"video-{phase}-hashcheck-publication-v1/terminal-output-v1/interrupted-inventory.json"
        )
        raw = path.read_bytes()
        inventory_pins[phase] = hashlib.sha256(raw).hexdigest()
        metadata = json.loads(raw)
        assert (
            metadata["reference"]["training_identity_sha256"]
            == contract["origin_identity_sha256"][phase]
        )
    inspector.validate_contract(
        contract,
        recovery,
        contract["checkpoint_freeze_sha256"],
        contract["audit_sha256"],
        inventory_pins,
    )


@pytest.mark.parametrize(
    "field",
    ["training", "scoring", "test_annotations_used", "max_seconds", "cpu_policy"],
)
def test_actual_contract_rejects_changed_software_parameters(field):
    contract = actual_contract()
    changed = copy.deepcopy(contract)
    changed[field] = not contract[field] if isinstance(contract[field], bool) else None
    with pytest.raises(ValueError, match="inspection-only"):
        inspector.validate_contract(
            changed,
            recovery,
            contract["checkpoint_freeze_sha256"],
            contract["audit_sha256"],
            contract["inventories"],
        )


def test_runtime_diff_uses_actual_observed_cpu_diagnostic():
    path = (
        PRIVATE
        / "video-gateb-phase1-publication-v1/environment-terminal-v1/environment-diagnostic.json"
    )
    if not path.exists():
        pytest.skip("Actual original/current CPU diagnostic required")
    diagnostic = json.loads(path.read_bytes())
    differences = inspector.diff_paths(diagnostic["expected"], diagnostic["actual"])
    assert differences and all(path.startswith("cpu_identity.") for path in differences)
    assert "cpu_identity.cpuinfo.vendor_id" in differences
    assert "cpu_identity.cpuinfo.flags" in differences
    assert inspector.diff_paths(diagnostic["actual"], diagnostic["actual"]) == []


def test_actual_full_runner_render_and_native_source_boundaries():
    path = ROOT / "scripts/kaggle_checkpoint_inspection.py"
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    token = b"EXPECTED_INSPECTION_PLAN_SHA256" + b" = None"
    assert raw.count(token) == 1
    compile(
        raw.replace(
            token, ("EXPECTED_INSPECTION_PLAN_SHA256 = " + repr(CONTRACT_SHA)).encode()
        ),
        str(path),
        "exec",
    )
    tree = ast.parse(raw)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert not any(
                alias.name.split(".")[0] in {"torch", "numpy", "src"}
                for alias in node.names
            )
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    forbidden = {
        "train",
        "engine",
        "primary",
        "evaluate",
        "step",
        "forward",
        "checked_protocol",
    }
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
        for node in calls
    )
    first_torch = min(
        node.lineno
        for node in ast.walk(run)
        if isinstance(node, ast.Import)
        and any(alias.name == "torch" for alias in node.names)
    )
    assert any(
        isinstance(node.func, ast.Name)
        and node.func.id == "checkpoint_bytes"
        and node.lineno < first_torch
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Name)
        and node.func.id == "source_bytes"
        and node.lineno < first_torch
        for node in calls
    )
    loads = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    assert len(loads) == 2
    for node in loads:
        keywords = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
        assert keywords == {"map_location": "cpu", "weights_only": True}
    source = raw.decode()
    assert source.index("parent = recovery._cloud_parent") < source.index(
        "import torch"
    )
    assert "Original full checkpoint identity mismatch" in source


def test_inspection_stager_own_source_import_closure_and_dirty_refusal(tmp_path, monkeypatch):
    tree = ast.parse((ROOT / "scripts/stage_checkpoint_inspection.py").read_text())
    imported = {
        "scripts/" + alias.name + ".py"
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "scripts"
        for alias in node.names
    }
    assert imported <= set(stager.OWN)
    def dirty(command, **kwargs):
        assert command == ["git", "status", "--porcelain"]
        return " M scripts/stage_checkpoint_inspection.py\n"
    monkeypatch.setattr(stager.subprocess, "check_output", dirty)
    with pytest.raises(ValueError, match="Clean committed"):
        stager.stage(
            tmp_path / "not-created",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    assert not (tmp_path / "not-created").exists()
