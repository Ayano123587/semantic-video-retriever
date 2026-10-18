"""Metadata-only staging for the independently restored ten-seed boundary."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_advanced_training as transport
from scripts import kaggle_research_training as legacy
from src.models import advanced_parent as advanced
from src.models import advanced_training
from src.models import cpu_portability as portability

ORIGINAL_SOURCE_SHA = "265b18be07ec7ec87d60231175963eeafe63e3fc6547a019e206517e2bcc0a76"
OWN = (
    "scripts/kaggle_advanced_training.py",
    "scripts/stage_advanced_training.py",
    "scripts/kaggle_research_training.py",
    "src/models/cpu_portability.py",
    "src/models/portable_training.py",
    "src/models/advanced_parent.py",
    "src/models/advanced_training.py",
)


def source_names(root, manifest):
    names = set(manifest["files"]) | set(OWN)
    if (Path(root) / "scripts/__init__.py").is_file():
        names.add("scripts/__init__.py")
    return names


def stage(
    output,
    *,
    policy_path,
    policy_sha256,
    service_path,
    service_sha256,
    source_manifest_path,
    source_review_path,
    source_review_sha256,
    inventory_path,
    evidence_paths,
    advanced_paths,
    protocol_path,
    dataset_id,
    kernel_id,
):
    """No directory or provider action precedes the complete actual review."""
    root = Path(__file__).resolve().parents[1]
    require = portability.require
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
        "Clean committed reviewed source checkout required",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    original = portability.load_evidence(
        evidence_paths["freeze"], evidence_paths["report"], evidence_paths["amd"]
    )
    context = advanced.load_context(advanced_paths, original)
    policy = portability.authenticate_policy(policy_path, policy_sha256, original)
    service = advanced.authenticate_service(
        service_path, service_sha256, policy_sha256, original, context
    )
    require(
        service["parent_schema"] == 1,
        "This transport requires actual checkpoint35 only",
    )
    origin = context["report"]["origin_identity"]
    manifest = advanced_training.authenticate_source(
        source_manifest_path, service, origin
    )
    inventory = portability.read_bound_json(
        inventory_path, service["parent_inventory_sha256"]
    )
    expected = {
        name: row
        for name, row in context["data"]["files"].items()
        if name != "provider-provenance.json"
    }
    require(
        inventory
        == {"schema": "advanced_checkpoint_parent_inventory_v3", "files": expected}
        and service["parent_native_paths"] == {name: name for name in expected}
        and service["parent_snapshot_name"] == "checkpoint-000035.pt"
        and service["parent_checkpoint_sha256"] == context["data"]["snapshot_sha256"]
        and service["parent_header_sha256"] == context["data"]["header_sha256"],
        "Exact independently restored twelve-file boundary required",
    )
    portability.read_bound_json(protocol_path, policy["origin_protocol_sha256"])
    names = source_names(root, manifest)
    hashes = {name: portability.sha(root / name) for name in sorted(names)}
    for name in names:
        require(
            subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root)
            == (root / name).read_bytes(),
            "Committed reviewed raw source differs",
        )
    for name in OWN:
        require(
            subprocess.check_output(
                ["git", "check-attr", "text", "--", name], cwd=root, text=True
            )
            .strip()
            .endswith(": unset"),
            "Explicit raw source attributes required",
        )
    review = portability.read_bound_json(source_review_path, source_review_sha256)
    require(
        review
        == {
            "schema": "advanced_training_source_review_v3",
            "head": head,
            "source_sha256": hashes,
            "phase": "phase2",
            "action": "continue_training",
            "policy_sha256": policy_sha256,
            "service_sha256": service_sha256,
            "source_manifest_sha256": service["source_manifest_sha256"],
            "advanced_restore_freeze_sha256": advanced.RESTORE_FREEZE_SHA,
            "checkpoint_freeze_sha256": advanced.DATA_FREEZE_SHA,
            "inventory_sha256": service["parent_inventory_sha256"],
        },
        "Actual root source/service review required",
    )
    require(
        dataset_id != kernel_id
        and dataset_id != context["data"]["dataset"]
        and len(dataset_id.split("/")[-1]) <= 50
        and len(kernel_id.split("/")[-1]) <= 50,
        "Distinct bounded source/kernel identities required",
    )
    payload = {
        "root-policy.json": Path(policy_path).read_bytes(),
        "service.json": Path(service_path).read_bytes(),
        "source-manifest.json": Path(source_manifest_path).read_bytes(),
        "source-review.json": Path(source_review_path).read_bytes(),
        "parent-inventory.json": Path(inventory_path).read_bytes(),
        "protocol.json": Path(protocol_path).read_bytes(),
        "inspection-freeze.json": Path(evidence_paths["freeze"]).read_bytes(),
        "inspection-report.json": Path(evidence_paths["report"]).read_bytes(),
        "amd-diagnostic.json": Path(evidence_paths["amd"]).read_bytes(),
        "advanced-freeze.json": Path(advanced_paths["freeze"]).read_bytes(),
        "advanced-inspection.json": Path(advanced_paths["inspection"]).read_bytes(),
        "advanced-restore.json": Path(advanced_paths["restore"]).read_bytes(),
        "advanced-status.json": Path(advanced_paths["status"]).read_bytes(),
        "advanced-contract.json": Path(advanced_paths["contract"]).read_bytes(),
        "checkpoint-freeze.json": Path(advanced_paths["data"]).read_bytes(),
        "advanced-train-runner.py": (root / OWN[0]).read_bytes(),
        "kaggle_research_training.py": (root / OWN[2]).read_bytes(),
    }
    for name in (
        "cpu_portability.py",
        "portable_training.py",
        "advanced_parent.py",
        "advanced_training.py",
    ):
        payload[name] = (root / "src/models" / name).read_bytes()
    payload.update({"project/" + name: (root / name).read_bytes() for name in names})
    plan = {
        "schema": "advanced_training_transport_v3",
        "phase": "phase2",
        "action": "continue_training",
        "max_seconds": 36000,
        "source_head": head,
        "source_review_sha256": source_review_sha256,
        "policy_sha256": policy_sha256,
        "service_sha256": service_sha256,
        "checkpoint_freeze_sha256": advanced.DATA_FREEZE_SHA,
        "source_manifest_sha256": service["source_manifest_sha256"],
        "original_source_sha256": ORIGINAL_SOURCE_SHA,
        "cache_manifest_sha256": policy["cache_manifest_sha256"],
        "protocol_sha256": policy["origin_protocol_sha256"],
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
    (package / "advanced-training-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    digest = portability.sha(package / "advanced-training-plan.json")
    transport.authenticate(package, digest)
    template = payload["advanced-train-runner.py"].replace(b"\r\n", b"\n")
    token = b"EXPECTED_ADVANCED_TRAIN_PLAN_SHA256" + b" = None"
    require(template.count(token) == 1, "One literal plan binding required")
    rendered = template.replace(
        token, ("EXPECTED_ADVANCED_TRAIN_PLAN_SHA256 = " + repr(digest)).encode()
    )
    compile(rendered, "run.py", "exec")
    (output / "run.py").write_bytes(rendered)
    kernel = {
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
            context["data"]["dataset"],
            legacy.SOURCE_DATASET,
        ],
        "kernel_sources": [legacy.ENCODING_KERNEL],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    upload = output / "upload-metadata"
    upload.mkdir()
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Advanced video training source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": digest,
        "runner_sha256": portability.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }
