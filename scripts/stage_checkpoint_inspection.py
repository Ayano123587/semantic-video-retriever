"""Prepare an externally reviewed metadata-only inspector; never dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import interrupted_checkpoint_recovery as recovery
from scripts import kaggle_checkpoint_inspection as runner
from scripts import kaggle_interrupted_attestation as gatea
from scripts import kaggle_research_training as legacy

OWN = (
    "scripts/interrupted_checkpoint_recovery.py",
    "scripts/kaggle_interrupted_attestation.py",
    "scripts/kaggle_research_training.py",
    "scripts/kaggle_checkpoint_inspection.py",
    "scripts/stage_checkpoint_inspection.py",
)


def stage(
    output,
    audit_path,
    audit_sha256,
    checkpoint_freeze_path,
    checkpoint_freeze_sha256,
    provider_provenance_path,
    inventory_paths,
    cache_manifest,
    protocol,
    inspection_contract_path,
    inspection_contract_sha256,
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
    if review != {
        "schema": "video_checkpoint_inspection_source_review_v1",
        "head": head,
        "source_sha256": sources,
        "inspection_contract_sha256": inspection_contract_sha256,
    }:
        raise ValueError("Actual root inspection source review required")
    for name in OWN:
        if (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
            != (root / name).read_bytes()
        ):
            raise ValueError("Exact committed raw inspection source required")
        if (
            not subprocess.check_output(
                ["git", "check-attr", "text", "--", name], cwd=root, text=True
            )
            .strip()
            .endswith(": unset")
        ):
            raise ValueError("Explicit -text source attributes required")
    contract = recovery._json(inspection_contract_path, inspection_contract_sha256)
    inventories = {phase: legacy.sha(path) for phase, path in inventory_paths.items()}
    runner.validate_contract(
        contract, recovery, checkpoint_freeze_sha256, audit_sha256, inventories
    )
    if set(inventory_paths) != {"phase1", "phase2"}:
        raise ValueError("Both actual inventories required")
    freeze = recovery._json(checkpoint_freeze_path, checkpoint_freeze_sha256)
    provider = recovery._json(
        provider_provenance_path, freeze["provider_provenance_sha256"]
    )
    for phase in ("phase1", "phase2"):
        reference = recovery.audit_reference(audit_path, audit_sha256, phase)
        mapping = gatea.checkpoint_mapping(freeze, provider, phase, reference)
        actual = recovery._json(inventory_paths[phase], contract["inventories"][phase])
        if actual != {
            "schema": "interrupted_checkpoint_hash_inventory_v1",
            "reference": reference,
            "files": {
                name: freeze["remote_files"][flat] for name, flat in mapping.items()
            },
            "native_deserialized": False,
            "scientific_complete": False,
            "resume_authorized": False,
        }:
            raise ValueError("Actual completed cloud inventories required")
    if (
        legacy.sha(cache_manifest) != legacy.CACHE_SHA
        or legacy.sha(protocol) != legacy.PROTOCOL_SHA
    ):
        raise ValueError("Actual original cache/protocol metadata required")
    dataset_title = dataset_id.split("/")[-1].replace("-", " ")
    kernel_title = kernel_id.split("/")[-1].replace("-", " ")
    if dataset_id == kernel_id or dataset_title == kernel_title:
        raise ValueError("Distinct source dataset and kernel identity required")
    payload = {
        "root-inspection-contract.json": Path(inspection_contract_path).read_bytes(),
        "root-inspection-source-review.json": Path(source_review_path).read_bytes(),
        "checkpoint-freeze.json": Path(checkpoint_freeze_path).read_bytes(),
        "provider-provenance-reviewed.json": Path(
            provider_provenance_path
        ).read_bytes(),
        "phase1-inventory.json": Path(inventory_paths["phase1"]).read_bytes(),
        "phase2-inventory.json": Path(inventory_paths["phase2"]).read_bytes(),
        "terminal-audit-report.json": Path(audit_path).read_bytes(),
        "cache-manifest.json": Path(cache_manifest).read_bytes(),
        "protocol.json": Path(protocol).read_bytes(),
        "interrupted_checkpoint_recovery.py": (root / OWN[0]).read_bytes(),
        "kaggle_interrupted_attestation.py": (root / OWN[1]).read_bytes(),
        "kaggle_research_training.py": (root / OWN[2]).read_bytes(),
        "inspection-runner.py": (root / OWN[3]).read_bytes().replace(b"\r\n", b"\n"),
        "stage_checkpoint_inspection.py": (root / OWN[4]).read_bytes(),
    }
    for entry in json.loads(Path(audit_path).read_bytes())["evidence_files"]:
        name = entry["path"]
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Safe audit metadata path required")
        raw = (Path(audit_path).parent / name).read_bytes()
        if (
            len(raw) != entry["bytes"]
            or hashlib.sha256(raw).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Actual audit evidence changed")
        payload[name] = raw
    plan = {
        "schema": "video_checkpoint_inspection_transport_v1",
        "source_head": head,
        "source_review_sha256": source_review_sha256,
        "inspection_contract_sha256": inspection_contract_sha256,
        "checkpoint_freeze_sha256": checkpoint_freeze_sha256,
        "audit_sha256": audit_sha256,
        "inventories": inventories,
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "max_seconds": 1800,
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
    (package / "inspection-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    digest = legacy.sha(package / "inspection-plan.json")
    token = b"EXPECTED_INSPECTION_PLAN_SHA256" + b" = None"
    if payload["inspection-runner.py"].count(token) != 1:
        raise ValueError("One literal inspection binding required")
    (output / "run.py").write_bytes(
        payload["inspection-runner.py"].replace(
            token, ("EXPECTED_INSPECTION_PLAN_SHA256 = " + repr(digest)).encode()
        )
    )
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": kernel_title,
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [
                    dataset_id,
                    freeze["dataset"],
                    legacy.SOURCE_DATASET,
                ],
                "kernel_sources": [],
                "competition_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    (package / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": dataset_title,
                "isPrivate": True,
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
