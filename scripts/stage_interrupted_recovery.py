"""Stage an externally reviewed interrupted-recovery service; never dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import interrupted_checkpoint_recovery as recovery
from scripts import kaggle_interrupted_attestation as gatea
from scripts import kaggle_research_training as legacy

OWN = (
    "scripts/interrupted_checkpoint_recovery.py",
    "scripts/kaggle_interrupted_attestation.py",
    "scripts/kaggle_research_training.py",
    "scripts/kaggle_interrupted_recovery.py",
    "scripts/stage_interrupted_recovery.py",
)


def stage(
    output,
    attestation_stage,
    attestation_plan_sha256,
    recovery_contract_path,
    recovery_contract_sha256,
    inventory_path,
    source_review_path,
    source_review_sha256,
    dataset_id,
    kernel_id,
):
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise ValueError("Clean committed reviewed checkout required")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    sources = {name: legacy.sha(root / name) for name in OWN}
    review = recovery._json(source_review_path, source_review_sha256)
    expected = {
        "schema": "interrupted_recovery_source_review_v1",
        "head": head,
        "source_sha256": sources,
        "attestation_plan_sha256": attestation_plan_sha256,
        "recovery_contract_sha256": recovery_contract_sha256,
    }
    if review != expected:
        raise ValueError("Actual root recovery source review required")
    for name in OWN:
        if (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
            != (root / name).read_bytes()
        ):
            raise ValueError("Committed raw recovery source differs")
        if (
            not subprocess.check_output(
                ["git", "check-attr", "text", "--", name], cwd=root, text=True
            )
            .strip()
            .endswith(": unset")
        ):
            raise ValueError("Explicit raw source attributes required")
    attestation_stage = Path(attestation_stage)
    prior = gatea.authenticate(attestation_stage / "dataset", attestation_plan_sha256)
    payload = {
        name: (attestation_stage / "dataset" / entry["upload_name"]).read_bytes()
        for name, entry in prior["files"].items()
    }
    contract = recovery._json(recovery_contract_path, recovery_contract_sha256)
    inventory = recovery._json(inventory_path, contract["inventory_sha256"])
    reference = prior["reference"]
    if contract != {
        "schema": "interrupted_checkpoint_recovery_v1",
        "action": "resume_exact_interrupted_checkpoint",
        "reference": reference,
        "inventory_sha256": legacy.sha(inventory_path),
        "max_seconds": 36000,
        "test_annotations_used": False,
        "restart_from_scratch": False,
    }:
        raise ValueError("Actual root interrupted recovery contract required")
    freeze = json.loads(payload["checkpoint-dataset-freeze.json"])
    provider = json.loads(payload["provider-provenance-reviewed.json"])
    mapping = gatea.checkpoint_mapping(freeze, provider, prior["phase"], reference)
    if inventory != {
        "schema": "interrupted_checkpoint_hash_inventory_v1",
        "reference": reference,
        "files": {name: freeze["remote_files"][flat] for name, flat in mapping.items()},
        "native_deserialized": False,
        "scientific_complete": False,
        "resume_authorized": False,
    }:
        raise ValueError("Actual root-frozen Gate A output required")
    payload.update(
        {
            "root-recovery-contract.json": Path(recovery_contract_path).read_bytes(),
            "root-inventory.json": Path(inventory_path).read_bytes(),
            "root-recovery-source-review.json": Path(source_review_path).read_bytes(),
            "interrupted_checkpoint_recovery.py": (root / OWN[0]).read_bytes(),
            "kaggle_interrupted_attestation.py": (root / OWN[1]).read_bytes(),
            "kaggle_research_training.py": (root / OWN[2]).read_bytes(),
            "recovery-runner.py": (root / OWN[3]).read_bytes().replace(b"\r\n", b"\n"),
            "stage_interrupted_recovery.py": (root / OWN[4]).read_bytes(),
        }
    )
    plan = {
        "schema": "interrupted_recovery_transport_v1",
        "phase": prior["phase"],
        "max_seconds": 36000,
        "source_head": head,
        "source_review_sha256": source_review_sha256,
        "audit_sha256": prior["audit_sha256"],
        "checkpoint_freeze_sha256": prior["checkpoint_freeze_sha256"],
        "recovery_contract_sha256": recovery_contract_sha256,
        "inventory_sha256": contract["inventory_sha256"],
        "source_manifest_sha256": legacy.SOURCE_SHA,
        "cache_manifest_sha256": legacy.CACHE_SHA,
        "protocol_sha256": legacy.PROTOCOL_SHA,
        "reference": reference,
        "test_annotations_used": False,
        "files": {
            name: {
                "upload_name": f"payload-{index:03d}" + Path(name).suffix,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for index, (name, raw) in enumerate(payload.items())
        },
    }
    output = Path(output)
    output.mkdir(exist_ok=False)
    package = output / "dataset"
    package.mkdir()
    for name, raw in payload.items():
        (package / plan["files"][name]["upload_name"]).write_bytes(raw)
    (package / "recovery-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    digest = legacy.sha(package / "recovery-plan.json")
    token = b"EXPECTED_RECOVERY_PLAN_SHA256" + b" = None"
    if payload["recovery-runner.py"].count(token) != 1:
        raise ValueError("One recovery literal binding required")
    (output / "run.py").write_bytes(
        payload["recovery-runner.py"].replace(
            token, ("EXPECTED_RECOVERY_PLAN_SHA256 = " + repr(digest)).encode()
        )
    )
    metadata = {
        "id": kernel_id,
        "title": kernel_id.split("/")[-1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_id, freeze["dataset"], legacy.SOURCE_DATASET],
        "kernel_sources": [legacy.ENCODING_KERNEL],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (package / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": dataset_id.split("/")[-1].replace("-", " "),
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": digest,
        "runner_sha256": legacy.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }
