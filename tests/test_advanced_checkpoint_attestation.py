"""Actual advanced-parent metadata and hash-only source-boundary checks."""

import ast
import json
from pathlib import Path

import pytest

from scripts import kaggle_advanced_checkpoint_attestation as runner

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path("D:/CodexData/portfolio/quant-research-lab-private")


@pytest.fixture
def actual():
    base = PRIVATE / "video-advanced-phase2-parent-v1"
    if not (base / "advanced-contract.json").is_file():
        pytest.skip("Actual root-authorized advanced-parent metadata required")
    contract = json.loads((base / "advanced-contract.json").read_bytes())
    header = json.loads((base / "parent-header.json").read_bytes())
    proof = json.loads((base / "parent-source-proof.json").read_bytes())
    return contract, header, proof


def test_actual_new_header_and_ten_seed_scope(actual):
    contract, header, proof = actual
    assert runner.validate_contract(contract, header, proof) == contract
    assert header["completed_seeds"] == 10 and header["batches"] == 34122
    assert proof["id"] == 139238005 and proof["version"] == 1
    report = (
        PRIVATE
        / "video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json"
    )
    assert runner.sha(report) == contract["origin_inspection_report_sha256"]
    old = json.loads(report.read_bytes())["phases"]["phase2"]["completed"]
    assert len(old) == 6
    assert all(
        contract["files"][row["name"]]["known_sha256"] == row["sha256"] for row in old
    )
    assert {
        name for name, row in contract["files"].items() if row["known_sha256"] is None
    } == {
        "config2-seed42.pt",
        "config2-seed1337.pt",
        "config2-seed2026.pt",
        "config3-seed42.pt",
    }


def test_actual_original_six_seed_header_is_not_advanced_parent(actual):
    contract, _, proof = actual
    path = (
        PRIVATE
        / "old-video-opaque-checkpoints-v1/phase2/research-phase2/checkpoint-000053.json"
    )
    if not path.is_file():
        pytest.skip("Actual original parent JSON header unavailable")
    with pytest.raises(ValueError, match="advanced parent header"):
        runner.validate_contract(contract, json.loads(path.read_bytes()), proof)


def test_no_native_or_project_helper_import_and_two_hash_passes():
    tree = ast.parse(
        (ROOT / "scripts/kaggle_advanced_checkpoint_attestation.py").read_bytes()
    )
    imports = {
        node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)
    }
    assert not imports & {"torch", "numpy", "pickle", "src", "scripts"}
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith(("src", "scripts"))
        for node in ast.walk(tree)
    )
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    hashes = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "hash_inventory"
    ]
    assert len(hashes) == 2
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func).split(".")[-1]
        in {"load", "loads_state_dict", "forward", "backward", "step"}
        for node in ast.walk(run)
    )


def test_actual_runner_literal_roundtrip_compiles():
    raw = (
        (ROOT / "scripts/kaggle_advanced_checkpoint_attestation.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    token = b"EXPECTED_ADVANCED_PLAN_SHA256" + b" = None"
    assert raw.count(token) == 1
    pin = runner.sha(ROOT / "scripts/kaggle_advanced_checkpoint_attestation.py")
    rendered = raw.replace(
        token, ("EXPECTED_ADVANCED_PLAN_SHA256 = " + repr(pin)).encode()
    )
    compile(rendered, "actual-source-render.py", "exec")


def test_host_directory_is_not_native_mount(actual):
    import sys

    if sys.platform == "win32":
        with pytest.raises(ValueError, match="Linux mounted"):
            runner.hash_inventory(ROOT, actual[0])
