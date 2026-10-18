"""Stage reviewed advanced-parent inspection and restoration, never dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_advanced_checkpoint_inspection as runner

OWN = (
    "scripts/advanced_checkpoint_inspection.py",
    "scripts/kaggle_advanced_checkpoint_inspection.py",
    "scripts/stage_advanced_checkpoint_inspection.py",
    "scripts/kaggle_research_training.py",
    "src/models/cpu_portability.py",
)
METADATA = {
    "root-contract.json",
    "checkpoint-freeze.json",
    "root-policy.json",
    "inspection-freeze.json",
    "inspection-report.json",
    "amd-diagnostic.json",
    "cache-manifest.json",
    "protocol.json",
    "actual-progress-diagnostic.json",
}


def validate_contract(contract, freeze):
    runner.require(
        contract["schema"] == "advanced_phase2_native_inspection_v1"
        and contract["artifact_revision"] == 2
        and contract["action"] == "inspect_and_restore_only"
        and contract["max_seconds"] == 1800
        and contract["training"] is False
        and contract["scoring"] is False
        and contract["test_annotations_used"] is False,
        "Root inspection and restore authorization required",
    )
    runner.require(
        freeze["schema"] == "video_advanced_phase2_checkpoint_data_freeze_v1"
        and freeze["private"] is True
        and freeze["status"] == "READY"
        and freeze["dataset_id"] == 12661805
        and freeze["version"] == 1
        and freeze["completed_seed_models"] == 10
        and len(freeze["files"]) == 13
        and contract["files"] == freeze["files"]
        and contract["snapshot_name"] == "checkpoint-000035.pt"
        and contract["snapshot_sha256"] == freeze["snapshot_sha256"]
        and contract["batches"] == 34122,
        "Actual independently frozen newer learned parent required",
    )
    names = [
        f"config{grid}-seed{seed}.pt" for grid in range(4) for seed in (42, 1337, 2026)
    ][:10]
    runner.require(
        contract["completed_names"] == names
        and set(freeze["files"])
        == set(
            names
            + [
                "checkpoint-000035.pt",
                "checkpoint-000035.json",
                "provider-provenance.json",
            ]
        ),
        "Exact actual ten-seed prefix and snapshot/header required",
    )


def stage(
    output,
    metadata_paths,
    metadata_sha256,
    source_review_path,
    source_review_sha256,
    dataset_id,
    kernel_id,
):
    root = Path(__file__).resolve().parents[1]
    runner.require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
        "Clean committed reviewed checkout required",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    runner.require(
        set(metadata_paths) == set(metadata_sha256) == METADATA,
        "All external root metadata pins required",
    )
    payload = {}
    for name, path in metadata_paths.items():
        raw = Path(path).read_bytes()
        runner.require(
            hashlib.sha256(raw).hexdigest() == metadata_sha256[name],
            "Actual root metadata changed",
        )
        payload[name] = raw
    contract = json.loads(payload["root-contract.json"])
    freeze = json.loads(payload["checkpoint-freeze.json"])
    validate_contract(contract, freeze)
    runner.require(
        metadata_sha256["actual-progress-diagnostic.json"]
        == contract["actual_progress_diagnostic_sha256"]
        and json.loads(payload["actual-progress-diagnostic.json"])["parent_progress"]
        == contract["actual_progress"],
        "Actual independently reviewed native boundary diagnostic required",
    )
    for key, name in (
        ("checkpoint_data_freeze_sha256", "checkpoint-freeze.json"),
        ("policy_sha256", "root-policy.json"),
        ("cache_manifest_sha256", "cache-manifest.json"),
        ("protocol_sha256", "protocol.json"),
    ):
        runner.require(
            contract[key] == metadata_sha256[name],
            "Contract must bind actual input bytes",
        )
    sources = {}
    for name in OWN:
        raw = (root / name).read_bytes()
        runner.require(
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root) == raw,
            "Exact committed raw source required",
        )
        attrs = subprocess.check_output(
            ["git", "check-attr", "text", "--", name], cwd=root, text=True
        )
        runner.require(
            attrs.strip().endswith(": unset"), "Raw -text source attributes required"
        )
        sources[name] = hashlib.sha256(raw).hexdigest()
    review_raw = Path(source_review_path).read_bytes()
    runner.require(
        hashlib.sha256(review_raw).hexdigest() == source_review_sha256,
        "Actual source review SHA required",
    )
    runner.require(
        json.loads(review_raw)
        == {
            "schema": "advanced_phase2_native_inspection_source_review_v1",
            "head": head,
            "source_sha256": sources,
            "contract_sha256": metadata_sha256["root-contract.json"],
        },
        "Actual independent root source review required",
    )
    payload["root-source-review.json"] = review_raw
    for name in OWN:
        mounted = "inspection-runner.py" if name == OWN[1] else Path(name).name
        payload[mounted] = (root / name).read_bytes()
    runner.require(
        dataset_id != kernel_id and dataset_id != freeze["dataset"],
        "Distinct source and kernel identities required",
    )
    plan = {
        "schema": "advanced_phase2_native_inspection_transport_v1",
        "source_head": head,
        "contract_sha256": metadata_sha256["root-contract.json"],
        "source_review_sha256": source_review_sha256,
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "max_seconds": 1800,
        "files": {
            name: {
                "upload_name": f"payload-{index:03d}" + Path(name).suffix,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
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
    plan_raw = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    (package / "advanced-inspection-plan.json").write_bytes(plan_raw)
    pin = hashlib.sha256(plan_raw).hexdigest()
    template = payload["inspection-runner.py"].replace(b"\r\n", b"\n")
    token = b"EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256" + b" = None"
    runner.require(template.count(token) == 1, "One literal plan pin required")
    rendered = template.replace(
        token, ("EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256 = " + repr(pin)).encode()
    )
    (output / "run.py").write_bytes(rendered)
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[-1].replace("-", " "),
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [
                    dataset_id,
                    freeze["dataset"],
                    "aikyatansinha/portfolio-didemo-full-research-source-v5",
                ],
                "kernel_sources": [],
                "competition_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    uploads = output / "upload-metadata"
    uploads.mkdir()
    (uploads / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Advanced video inspection source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": pin,
        "runner_sha256": hashlib.sha256(rendered).hexdigest(),
        "payload_files": len(payload),
        "uploaded": False,
        "dispatched": False,
    }
