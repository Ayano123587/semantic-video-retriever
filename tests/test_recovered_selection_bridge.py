"""Actual recovered terminal metadata; no synthetic learned artifacts."""

import ast
import copy
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import recovered_selection_bridge as bridge
from src.models import advanced_parent as advanced
from src.models import cpu_portability as portability

PRIVATE = Path(
    os.environ.get("VIDEO_PRIVATE", "D:/CodexData/portfolio/quant-research-lab-private")
)


def actual_inputs():
    base = PRIVATE / "video-advanced-phase2-parent-v1"
    freeze = (
        PRIVATE
        / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json"
    )
    if not freeze.is_file():
        pytest.skip("Actual original private evidence unavailable")
    evidence = portability.load_evidence(
        freeze,
        PRIVATE
        / "video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json",
        PRIVATE
        / "video-gateb-phase1-publication-v1/environment-terminal-v1/environment-diagnostic.json",
    )
    obs = base / "native-inspection-publication-v3/observation-20261017T192355"
    context = advanced.load_context(
        {
            "freeze": base / "root-actual-native-restore-freeze-v1.json",
            "inspection": obs / "inspection.json",
            "restore": obs / "restore-result.json",
            "status": obs / "worker-status.json",
            "contract": base / "root-native-contract-v2.json",
            "data": base / "checkpoint-dataset-v1/root-data-freeze-v1.json",
        },
        evidence,
    )
    policy_path = PRIVATE / "video-phase1-portable-training-v1/root-policy.json"
    policy_pin = portability.sha(policy_path)
    policy = portability.authenticate_policy(policy_path, policy_pin, evidence)
    inputs = {}
    for phase in ("phase1", "phase2"):
        publication = (
            PRIVATE / "video-phase1-portable-training-v1/publication-v1"
            if phase == "phase1"
            else base / "continuation-v1/publication-v1"
        )
        directory = publication / "terminal-reconciliation-v1"
        service = (
            PRIVATE / "video-phase1-portable-training-v1/phase1-training-service.json"
            if phase == "phase1"
            else base / "continuation-v1/prospective-training-service.json"
        )
        header = (
            "checkpoint-000005.json" if phase == "phase1" else "checkpoint-000022.json"
        )
        wrapper = directory / (
            "portable-status.json" if phase == "phase1" else "status.json"
        )
        report = json.loads((directory / "results.json").read_bytes())
        proof_path = publication / "server-source-proof.json"
        proof = json.loads(proof_path.read_bytes())
        contract = {
            "phase": phase,
            "kernel": proof["canonical"],
            "kernel_id": proof["id"],
            "version": proof["version"],
            "transport_sha256": proof["source_sha256"],
            "execution_proof_sha256": portability.sha(proof_path),
            "datasets": proof["datasets"],
            "encoding_parent": proof["encoding_parent"],
            "report_sha256": portability.sha(directory / "results.json"),
            "wrapper_sha256": portability.sha(wrapper),
            "final_header": header,
            "final_header_sha256": portability.sha(directory / header),
            "service_sha256": portability.sha(service),
            "policy_sha256": policy_pin,
            "execution_runtime": report["actual_execution_runtime"],
        }
        inputs[phase] = {
            "directory": directory,
            "wrapper_path": wrapper,
            "proof_path": proof_path,
            "service_path": service,
            "contract": contract,
        }
    origin = evidence["report"]["phases"]["phase1"]["origin_identity"]
    scientific = {
        key: origin[key]
        for key in (
            "cache_manifest_sha256",
            "protocol_sha256",
            "implementation_sha256",
            "coverage",
            "encoding_identity",
        )
    }
    scientific.update(
        source_manifest_sha256="265b18be07ec7ec87d60231175963eeafe63e3fc6547a019e206517e2bcc0a76",
        baseline=json.loads(
            (inputs["phase1"]["directory"] / "results.json").read_bytes()
        )["baseline"],
        training_runtime=origin["runtime"],
        environment=origin["encoding_identity"],
    )
    return inputs, scientific, policy, evidence, context


def _isolated_test_actual_both_recovered_schemas_validate_without_native_imports():
    inputs, scientific, policy, evidence, context = actual_inputs()
    for phase, row in inputs.items():
        report, header = bridge.phase_metadata(
            **row,
            scientific=scientific,
            policy=policy,
            evidence=evidence,
            advanced_context=context,
        )
        assert header["schema"] == (2 if phase == "phase1" else 3)
        assert report["completed_fits"] == (9 if phase == "phase1" else 12)
        assert report["selected"] == "frozen_clip"
    assert "torch" not in sys.modules and "numpy" not in sys.modules

def test_actual_both_recovered_schemas_validate_without_native_imports():
    import subprocess
    if not (PRIVATE / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json").is_file():
        pytest.skip("Actual original private evidence unavailable")
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'tests')]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from test_recovered_selection_bridge import _isolated_test_actual_both_recovered_schemas_validate_without_native_imports
_isolated_test_actual_both_recovered_schemas_validate_without_native_imports()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(root)], check=True, capture_output=True, text=True, timeout=120)



def test_actual_metadata_role_mismatch_fails_before_native():
    inputs, scientific, policy, evidence, context = actual_inputs()
    row = copy.deepcopy(inputs["phase2"])
    row["contract"]["execution_runtime"] = evidence["amd_observation"]["actual"]
    with pytest.raises(ValueError, match="execution runtime differs"):
        bridge.phase_metadata(
            **row,
            scientific=scientific,
            policy=policy,
            evidence=evidence,
            advanced_context=context,
        )


def test_no_inventory_can_call_selection_loader():
    called = []
    with pytest.raises(ValueError, match="root-frozen inventories"):
        bridge.guarded_choose({}, {}, {}, {}, None, None, lambda: called.append(True))
    assert not called


def test_selection_source_keeps_pre_and_postflight_and_no_native_imports():
    path = Path(bridge.__file__)
    tree = ast.parse(path.read_bytes())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "guarded_choose"
    )
    calls = [
        (n.lineno, ast.unparse(n.func))
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
    ]
    callback = next(line for line, name in calls if name == "choose_callback")
    assert any(line < callback for line, name in calls if name == "authenticate_both")
    assert any(line > callback for line, name in calls if name == "authenticate_both")
    assert not any(
        isinstance(n, ast.Import)
        and any(a.name.split(".")[0] in {"torch", "numpy"} for a in n.names)
        for n in ast.walk(tree)
    )
