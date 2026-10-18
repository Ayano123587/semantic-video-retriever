"""Byte-only attestation of both genuinely completed recovered phases."""

import hashlib
import json
import platform
import stat
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXPECTED_INVENTORY_PLAN_SHA256 = None
PHASE_KERNELS = {
    "phase1": "aikyatansinha/portfolio-video-phase1-portable-training-v1/1",
    "phase2": "aikyatansinha/portfolio-video-phase2-advanced-training-v1/1",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(name):
    require(
        isinstance(name, str)
        and name
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and "\\" not in name
        and ":" not in name,
        "Safe package-relative path required",
    )
    return name


def authenticate(package, pin):
    require(
        sha(package / "recovered-inventory-plan.json") == pin,
        "Actual external plan pin required",
    )
    plan = json.loads((package / "recovered-inventory-plan.json").read_bytes())
    require(
        plan["schema"] == "recovered_inventory_transport_v1"
        and plan["max_seconds"] == 1800
        and plan["action"] == "hash_only"
        and plan["training"] is False
        and plan["scoring"] is False
        and plan["test_annotations_used"] is False,
        "Hash-only transport required",
    )
    uploads = []
    for name, entry in plan["files"].items():
        safe(name)
        upload = safe(entry["upload_name"])
        require(Path(upload).name == upload, "Flat source payload required")
        path = package / upload
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == entry["bytes"]
            and sha(path) == entry["sha256"],
            "All payload bytes must authenticate before helpers",
        )
        uploads.append(upload)
    require(len(uploads) == len(set(uploads)), "Unique source payload names required")
    return plan


def mounted(root, name, pin):
    matches = [p for p in Path(root).rglob(name) if p.is_file() and sha(p) == pin]
    require(len(matches) == 1, "One exact actual parent artifact required")
    path = matches[0]
    require(
        sys.platform == "linux"
        and path.resolve().is_relative_to(Path("/kaggle/input"))
        and all(not stat.S_ISLNK(p.lstat().st_mode) for p in (path, *path.parents)),
        "Actual regular Linux input path required",
    )
    return path


def metadata_gate(paths, approval, directories):
    """Actual protected JSON only; usable for source-only host validation."""
    from scripts import recovered_selection_bridge as bridge
    from src.models import advanced_parent as advanced
    from src.models import cpu_portability as portability

    require(
        approval["schema"] == "root_recovered_inventory_contract_v1"
        and approval["max_seconds"] == 1800
        and approval["action"] == "hash_only"
        and approval["training"] is False
        and approval["scoring"] is False
        and approval["test_annotations_used"] is False
        and set(approval["phases"]) == {"phase1", "phase2"},
        "Separate actual root hash-only contract required",
    )
    require(
        set(paths) == set(approval["metadata_sha256"]),
        "All actual metadata roles required",
    )
    for name, path in paths.items():
        require(
            sha(path) == approval["metadata_sha256"][name],
            "Actual root-bound metadata differs",
        )
    evidence = portability.load_evidence(
        paths["original-freeze.json"],
        paths["original-inspection.json"],
        paths["amd-diagnostic.json"],
    )
    context = advanced.load_context(
        {
            "freeze": paths["advanced-freeze.json"],
            "inspection": paths["advanced-inspection.json"],
            "restore": paths["advanced-restore.json"],
            "status": paths["advanced-status.json"],
            "contract": paths["advanced-contract.json"],
            "data": paths["advanced-data-freeze.json"],
        },
        evidence,
    )
    policy_pin = approval["metadata_sha256"]["policy.json"]
    policy = portability.authenticate_policy(paths["policy.json"], policy_pin, evidence)
    scientific = approval["scientific"]
    require(
        approval["metadata_sha256"]["original-source-manifest.json"]
        == scientific["source_manifest_sha256"]
        == "265b18be07ec7ec87d60231175963eeafe63e3fc6547a019e206517e2bcc0a76"
        and approval["metadata_sha256"]["cache-manifest.json"]
        == scientific["cache_manifest_sha256"]
        and approval["metadata_sha256"]["protocol.json"]
        == scientific["protocol_sha256"],
        "Original scientific transport/cache/protocol bytes required",
    )
    cache = json.loads(Path(paths["cache-manifest.json"]).read_bytes())
    protocol = json.loads(Path(paths["protocol.json"]).read_bytes())
    require(
        protocol["encoding_identity"]
        == cache["encoding_identity"]
        == scientific["encoding_identity"]
        and protocol["baseline"] == scientific["baseline"],
        "Historical encoder/full baseline provenance differs",
    )
    inputs = {}
    for phase in ("phase1", "phase2"):
        contract = approval["phases"][phase]
        require(
            contract["kernel"] + "/" + str(contract["version"]) == PHASE_KERNELS[phase]
            and contract["policy_sha256"] == policy_pin,
            "Exact completed phase/policy required",
        )
        row = {
            "directory": Path(directories[phase]),
            "wrapper_path": paths[phase + "-wrapper.json"],
            "proof_path": paths[phase + "-proof.json"],
            "service_path": paths[phase + "-service.json"],
            "contract": contract,
        }
        require(
            sha(row["directory"] / "results.json")
            == approval["metadata_sha256"][phase + "-results.json"]
            and sha(row["directory"] / contract["final_header"])
            == approval["metadata_sha256"][phase + "-header.json"],
            "Actual parent metadata must match frozen payload",
        )
        bridge.phase_metadata(
            **row,
            scientific=scientific,
            policy=policy,
            evidence=evidence,
            advanced_context=context,
        )
        inputs[phase] = row
    return inputs, scientific, policy, evidence, context


def run(root, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(exist_ok=False)
    status = {
        "action": "hash_only",
        "complete": False,
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "learned_files_deserialized": False,
        "stage": "start",
    }

    def record():
        (output / "status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    record()
    try:
        package = mounted(
            root, "recovered-inventory-plan.json", EXPECTED_INVENTORY_PLAN_SHA256
        ).parent
        plan = authenticate(package, EXPECTED_INVENTORY_PLAN_SHA256)
        workspace = Path(tempfile.mkdtemp(prefix="recovered-byte-attestation-"))
        for name, entry in plan["files"].items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / entry["upload_name"]).read_bytes())
        template = (
            (workspace / "runner-template.py").read_bytes().replace(b"\r\n", b"\n")
        )
        token = b"EXPECTED_INVENTORY_PLAN_SHA256" + b" = None"
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_INVENTORY_PLAN_SHA256 = "
                    + repr(EXPECTED_INVENTORY_PLAN_SHA256)
                ).encode(),
            ),
            "Exact rendered hash-only runner required",
        )

        def raw_guard():
            require(
                time.monotonic() - started < 1800,
                "Original hash-only service budget exhausted",
            )
            authenticate(package, EXPECTED_INVENTORY_PLAN_SHA256)
            for name, entry in plan["files"].items():
                require(
                    sha(workspace / name) == entry["sha256"],
                    "Materialized source/metadata changed",
                )

        raw_guard()
        sys.path.insert(0, str(workspace / "project"))
        from scripts import recovered_selection_bridge as bridge

        approval = json.loads((workspace / "approval.json").read_bytes())
        require(
            sha(workspace / "approval.json") == plan["approval_sha256"],
            "Actual root contract pin required",
        )
        paths = {
            name: workspace / "metadata" / name for name in approval["metadata_sha256"]
        }
        directories = {
            phase: mounted(root, "results.json", row["report_sha256"]).parent
            for phase, row in approval["phases"].items()
        }
        inputs, scientific, policy, evidence, context = metadata_gate(
            paths, approval, directories
        )
        status.update(
            stage="hashing_native_bytes",
            python=platform.python_version(),
            actual_metadata_verified=True,
        )
        record()
        inventories = bridge.authenticate_both(
            inputs, scientific, policy, evidence, context
        )
        raw_guard()
        require(
            inventories
            == bridge.authenticate_both(inputs, scientific, policy, evidence, context),
            "Native bytes changed during double attestation",
        )
        require(
            "torch" not in sys.modules and "numpy" not in sys.modules,
            "Hash-only execution must never import native model/array packages",
        )
        hashes = {}
        for phase, inventory in inventories.items():
            target = output / (phase + "-inventory.json")
            target.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
            hashes[phase] = sha(target)
        status.update(
            stage="complete",
            complete=True,
            inventories_sha256=hashes,
            resolved_parents={phase: str(path) for phase, path in directories.items()},
        )
        record()
        (output / "COMPLETE").write_text(sha(output / "status.json") + "\n")
    except Exception as error:
        status.update(
            stage="failed",
            complete=False,
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/recovered-phase-inventories")
