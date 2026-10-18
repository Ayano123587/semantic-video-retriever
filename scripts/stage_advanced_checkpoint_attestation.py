"""Freeze a hash-only advanced-parent job; never upload or dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_advanced_checkpoint_attestation as attestation

OWN = (
    "scripts/kaggle_advanced_checkpoint_attestation.py",
    "scripts/stage_advanced_checkpoint_attestation.py",
)


def stage(
    output,
    *,
    contract_path,
    contract_sha256,
    header_path,
    source_proof_path,
    dataset_id,
    kernel_id,
):
    root = Path(__file__).resolve().parents[1]
    require = attestation.require
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
        "Clean committed source checkout required",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    for name in OWN:
        require(
            subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root)
            == (root / name).read_bytes(),
            "Committed raw attestation source required",
        )
        require(
            subprocess.check_output(
                ["git", "check-attr", "text", "--", name], cwd=root, text=True
            )
            .strip()
            .endswith(": unset"),
            "Raw source attributes required",
        )
    require(
        attestation.sha(contract_path) == contract_sha256,
        "Actual external contract SHA required",
    )
    contract = json.loads(Path(contract_path).read_bytes())
    require(
        attestation.sha(header_path) == contract["header_sha256"]
        and attestation.sha(source_proof_path)
        == contract["parent_source_proof_sha256"],
        "Actual parent metadata bytes required",
    )
    attestation.validate_contract(
        contract,
        json.loads(Path(header_path).read_bytes()),
        json.loads(Path(source_proof_path).read_bytes()),
    )
    require(
        dataset_id != kernel_id,
        "Distinct source dataset and kernel identities required",
    )
    payload = {
        "advanced-contract.json": Path(contract_path).read_bytes(),
        "parent-header.json": Path(header_path).read_bytes(),
        "parent-source-proof.json": Path(source_proof_path).read_bytes(),
        "advanced-runner.py": (root / OWN[0]).read_bytes().replace(b"\r\n", b"\n"),
        "stage-source.py": (root / OWN[1]).read_bytes(),
    }
    plan = {
        "schema": "advanced_checkpoint_attestation_transport_v1",
        "head": head,
        "contract_sha256": contract_sha256,
        "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in payload.items()
        },
    }
    output = Path(output)
    output.mkdir(exist_ok=False)
    package = output / "dataset"
    package.mkdir()
    for name, raw in payload.items():
        (package / name).write_bytes(raw)
    (package / "advanced-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    pin = attestation.sha(package / "advanced-plan.json")
    attestation.authenticate(package, pin)
    token = b"EXPECTED_ADVANCED_PLAN_SHA256" + b" = None"
    require(
        payload["advanced-runner.py"].count(token) == 1, "One plan binding required"
    )
    rendered = payload["advanced-runner.py"].replace(
        token, ("EXPECTED_ADVANCED_PLAN_SHA256 = " + repr(pin)).encode()
    )
    compile(rendered, "advanced-render.py", "exec")
    (output / "run.py").write_bytes(rendered)
    metadata = {
        "id": kernel_id,
        "title": kernel_id.split("/")[-1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [dataset_id],
        "kernel_sources": [contract["parent_execution"]["canonical"] + "/1"],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    upload = output / "upload-metadata"
    upload.mkdir()
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Advanced video parent source v1",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": pin,
        "runner_sha256": attestation.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }
