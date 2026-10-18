"""Source/actual recovered metadata only; never model or price fixtures."""

import ast
import json
import os
from pathlib import Path

import pytest
from test_recovered_selection_bridge import PRIVATE, actual_inputs

from scripts import kaggle_recovered_phase_inventory as worker
from scripts import stage_recovered_phase_inventory as stager
from src.models import cpu_portability as portability

ROOT = Path(__file__).resolve().parents[1]


def actual_metadata_contract():
    inputs, scientific, _policy, _evidence, _context = actual_inputs()
    base = PRIVATE / "video-advanced-phase2-parent-v1"
    obs = base / "native-inspection-publication-v3/observation-20261017T192355"
    paths = {
        "policy.json": PRIVATE / "video-phase1-portable-training-v1/root-policy.json",
        "original-freeze.json": PRIVATE
        / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json",
        "original-inspection.json": PRIVATE
        / "video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json",
        "amd-diagnostic.json": PRIVATE
        / "video-gateb-phase1-publication-v1/environment-terminal-v1/environment-diagnostic.json",
        "advanced-freeze.json": base / "root-actual-native-restore-freeze-v1.json",
        "advanced-inspection.json": obs / "inspection.json",
        "advanced-restore.json": obs / "restore-result.json",
        "advanced-status.json": obs / "worker-status.json",
        "advanced-contract.json": base / "root-native-contract-v2.json",
        "advanced-data-freeze.json": base
        / "checkpoint-dataset-v1/root-data-freeze-v1.json",
        "original-source-manifest.json": Path(
            os.environ.get("VIDEO_ORIGINAL_SOURCE_MANIFEST", "D:/CodexData/portfolio/semantic-video-retriever/data/interim/research-cloud-encoding-v5/source/video-source-manifest.json")
        ),
        "cache-manifest.json": ROOT
        / "data/interim/research-cloud-encoding-v5/terminal-output-v1/research-features/manifest.json",
        "protocol.json": ROOT
        / "data/interim/research-cloud-encoding-v5/terminal-output-v1/research-training-protocol.json",
    }
    for phase, row in inputs.items():
        paths.update(
            {
                phase + "-results.json": row["directory"] / "results.json",
                phase + "-header.json": row["directory"]
                / row["contract"]["final_header"],
                phase + "-wrapper.json": row["wrapper_path"],
                phase + "-proof.json": row["proof_path"],
                phase + "-service.json": row["service_path"],
            }
        )
    contract = {
        "schema": "root_recovered_inventory_contract_v1",
        "action": "hash_only",
        "max_seconds": 1800,
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "metadata_sha256": {
            name: portability.sha(path) for name, path in paths.items()
        },
        "scientific": scientific,
        "phases": {phase: row["contract"] for phase, row in inputs.items()},
    }
    return paths, contract, {phase: row["directory"] for phase, row in inputs.items()}


def test_actual_metadata_gate_all_roles_and_native_free_scope():
    paths, contract, directories = actual_metadata_contract()
    inputs, scientific, _policy, _evidence, _context = worker.metadata_gate(
        paths, contract, directories
    )
    assert len(paths) == 23 and set(inputs) == {"phase1", "phase2"}
    assert inputs["phase1"]["contract"]["kernel_id"] == 139270426
    assert inputs["phase2"]["contract"]["kernel_id"] == 139291990
    assert scientific["baseline"]["queries"] == 3132
    wrong = dict(paths)
    wrong["phase2-proof.json"] = paths["phase1-proof.json"]
    with pytest.raises(ValueError, match="root-bound metadata differs"):
        worker.metadata_gate(wrong, contract, directories)


def test_actual_source_closure_covers_original_and_transitive_helpers():
    paths, _contract, _directories = actual_metadata_contract()
    original = json.loads(paths["original-source-manifest.json"].read_bytes())
    names = stager.source_names(ROOT, original)
    originals = {
        row["path"]
        for row in original["files"]
        if row["path"].startswith("src/") and row["path"].endswith(".py")
    }
    assert len(originals) == 19 and originals <= names
    assert set(stager.OWN) <= names
    for name in names:
        for node in ast.walk(ast.parse((ROOT / name).read_bytes())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(("src.", "scripts."))
            ):
                for module in [
                    node.module,
                    *[node.module + "." + alias.name for alias in node.names],
                ]:
                    candidate = ROOT.joinpath(*module.split(".")).with_suffix(".py")
                    if candidate.is_file():
                        assert candidate.relative_to(ROOT).as_posix() in names


def test_actual_runner_binding_and_byte_only_double_attestation():
    raw = (ROOT / stager.OWN[0]).read_bytes().replace(b"\r\n", b"\n")
    token = b"EXPECTED_INVENTORY_PLAN_SHA256" + b" = None"
    assert raw.count(token) == 1
    ast.parse(
        raw.replace(
            token,
            (
                "EXPECTED_INVENTORY_PLAN_SHA256 = "
                + repr(portability.sha(ROOT / stager.OWN[0]))
            ).encode(),
        )
    )
    tree = ast.parse(raw)
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name.split(".")[0] in {"torch", "numpy"} for alias in node.names)
        for node in ast.walk(tree)
    )
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [
        (node.lineno, ast.unparse(node.func))
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
    ]
    attest = [line for line, name in calls if name == "bridge.authenticate_both"]
    assert len(attest) == 2
    assert max(attest) < max(
        line for line, name in calls if name.endswith(".write_text")
    )
    assert min(line for line, name in calls if name == "raw_guard") < min(attest)
