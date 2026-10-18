"""Prepare reviewed private hash-only metadata; never upload or dispatch."""

import json
import subprocess
from pathlib import Path

from scripts import interrupted_checkpoint_recovery as recovery
from scripts import kaggle_interrupted_attestation as runner

OWN = (
    "scripts/interrupted_checkpoint_recovery.py",
    "scripts/kaggle_interrupted_attestation.py",
    "scripts/stage_interrupted_attestation.py",
)


def stage(
    output,
    phase,
    audit_path,
    audit_sha256,
    approval_path,
    approval_sha256,
    cache_manifest,
    protocol,
    dataset_id,
    kernel_id,
    checkpoint_freeze_path=None,
    checkpoint_freeze_sha256=None,
    provider_provenance_path=None,
):
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise ValueError("Clean committed reviewed checkout required")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    approval = recovery._json(approval_path, approval_sha256)
    sources = {name: runner.sha(root / name) for name in OWN}
    if approval != {
        "schema": "interrupted_attestation_review_v1",
        "head": head,
        "source_sha256": sources,
        "audit_sha256": audit_sha256,
        "action": "hash_only_no_resume",
        "checkpoint_freeze_sha256": checkpoint_freeze_sha256,
    }:
        raise ValueError("Actual external root source review required")
    for name in OWN:
        attribute = subprocess.check_output(
            ["git", "check-attr", "text", "--", name], cwd=root, text=True
        ).strip()
        if not attribute.endswith(": unset"):
            raise ValueError(
                "Reviewed raw sources require explicit -text Git attributes"
            )
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
        if committed != (root / name).read_bytes():
            raise ValueError("Committed raw source differs")
    reference = recovery.audit_reference(audit_path, audit_sha256, phase)
    checkpoint_freeze = recovery._json(checkpoint_freeze_path, checkpoint_freeze_sha256)
    provider = recovery._json(
        provider_provenance_path, checkpoint_freeze["provider_provenance_sha256"]
    )
    runner.checkpoint_mapping(checkpoint_freeze, provider, phase, reference)
    if (
        runner.sha(cache_manifest) != recovery.CACHE_SHA
        or runner.sha(protocol) != recovery.PROTOCOL_SHA
    ):
        raise ValueError("Actual immutable cache/protocol metadata required")
    report = json.loads(Path(audit_path).read_bytes())
    evidence = [
        entry
        for entry in report["evidence_files"]
        if entry["path"].startswith(phase + "/")
    ]
    payload = {
        "terminal-audit-report.json": Path(audit_path).read_bytes(),
        "cache-manifest.json": Path(cache_manifest).read_bytes(),
        "research-training-protocol.json": Path(protocol).read_bytes(),
        "root-review.json": Path(approval_path).read_bytes(),
        "checkpoint-dataset-freeze.json": Path(checkpoint_freeze_path).read_bytes(),
        "provider-provenance-reviewed.json": Path(
            provider_provenance_path
        ).read_bytes(),
        "stage_interrupted_attestation.py": (root / OWN[2]).read_bytes(),
        "interrupted_checkpoint_recovery.py": (root / OWN[0]).read_bytes(),
        "attestation-runner.py": (root / OWN[1]).read_bytes().replace(b"\r\n", b"\n"),
    }
    for entry in evidence:
        payload[entry["path"]] = (Path(audit_path).parent / entry["path"]).read_bytes()
    import hashlib

    plan = {
        "schema": "interrupted_attestation_transport_v1",
        "phase": phase,
        "accepted_parent_kernel": reference["kernel_source"],
        "audit_path": "terminal-audit-report.json",
        "audit_sha256": audit_sha256,
        "reference": reference,
        "review_sha256": approval_sha256,
        "source_head": head,
        "checkpoint_dataset": checkpoint_freeze["dataset"],
        "checkpoint_dataset_id": checkpoint_freeze["dataset_id"],
        "checkpoint_dataset_version": checkpoint_freeze["version"],
        "checkpoint_freeze_sha256": checkpoint_freeze_sha256,
        "files": {
            name: {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "upload_name": f"payload-{index:03d}" + Path(name).suffix,
            }
            for index, (name, raw) in enumerate(payload.items())
        },
    }
    output = Path(output)
    output.mkdir(exist_ok=False)
    package = output / "dataset"
    package.mkdir()
    for name, raw in payload.items():
        path = package / plan["files"][name]["upload_name"]
        path.write_bytes(raw)
    (package / "attestation-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    digest = runner.sha(package / "attestation-plan.json")
    token = b"EXPECTED_PLAN_SHA256" + b" = None"
    template = payload["attestation-runner.py"]
    if template.count(token) != 1:
        raise ValueError("One literal plan binding required")
    (output / "run.py").write_bytes(
        template.replace(token, ("EXPECTED_PLAN_SHA256 = " + repr(digest)).encode())
    )
    metadata = {
        "id": kernel_id,
        "title": kernel_id.split("/", 1)[1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [dataset_id, checkpoint_freeze["dataset"]],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (package / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": f"Private {phase} checkpoint attestation",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {"plan_sha256": digest, "uploaded": False, "dispatched": False}
