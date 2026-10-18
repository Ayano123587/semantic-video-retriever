"""Hash-only cancelled-parent attestation; no training or native deserialization."""

import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

EXPECTED_PLAN_SHA256 = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate(package, expected):
    package = Path(package)
    raw = (package / "attestation-plan.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Externally frozen plan differs")
    plan = json.loads(raw)
    if plan["schema"] != "interrupted_attestation_transport_v1":
        raise ValueError("Hash-only transport required")
    if (
        "reference" in plan
        and plan.get("accepted_parent_kernel") != plan["reference"]["kernel_source"]
    ):
        raise ValueError("Exact accepted kernel reference required")
    for name, entry in plan["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe package path")
        upload_name = entry["upload_name"]
        if Path(upload_name).name != upload_name:
            raise ValueError("Flat upload filename required")
        path = package / upload_name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError("Package file differs")
        if sha(path) != entry["sha256"]:
            raise ValueError("Package bytes differ")
    return plan


def discover_parent(input_root, reference):
    """Match immutable header bytes and content; never infer a mount dirname."""
    header_name = Path(reference["snapshot"]).with_suffix(".json").name
    matches = []
    for path in Path(input_root).rglob(header_name):
        if path.is_symlink() or sha(path) != reference["header_sha256"]:
            continue
        metadata = json.loads(path.read_bytes())
        expected = {
            "snapshot_sha256": reference["snapshot_sha256"],
            "identity_sha256": reference["training_identity_sha256"],
            "completed_seeds": reference["completed_seeds"],
            "batches": reference["batches"],
            "stage": "fit",
        }
        if metadata == expected:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError("One unique exact cancelled-parent header mount required")
    return matches[0]


def checkpoint_mapping(freeze, provider, phase, reference):
    """Validate the externally frozen actual dataset and exact historical roles."""
    if (
        freeze.get("schema") != "video_checkpoint_dataset_freeze_v1"
        or freeze.get("dataset")
        != "aikyatansinha/portfolio-video-recovery-checkpoints-v1"
        or not isinstance(freeze.get("dataset_id"), int)
        or freeze["dataset_id"] <= 0
        or freeze.get("version") != 1
        or freeze.get("private") is not True
        or freeze.get("status") != "ready"
        or freeze.get("root_audit_sha256") != reference["root_audit_sha256"]
        or freeze.get("transfer_manifest_sha256")
        != "e87063ecd0fe5c6b031a399f9bdf4d95016f786d9e929b921d3f446f98a27de6"
    ):
        raise ValueError("Actual root-frozen private READY checkpoint dataset required")
    if (
        provider.get("schema") != "video_checkpoint_provider_v1"
        or provider.get("dataset") != freeze["dataset"]
        or provider.get("transfer_manifest_sha256")
        != freeze["transfer_manifest_sha256"]
        or provider.get("root_audit_sha256") != freeze["root_audit_sha256"]
        or provider.get("native_deserialized") is not False
        or provider.get("scientific_complete") is not False
    ):
        raise ValueError("Actual opaque provider provenance required")
    native = {}
    selected = None
    for job in provider["jobs"]:
        current = job["phase"]
        version, accepted, checkpoint, completed = {
            "phase1": (2, 139178784, 47, 8),
            "phase2": (3, 139178893, 53, 6),
        }[current]
        parent = f"aikyatansinha/portfolio-didemo-full-research-{current}-v1/{version}"
        if job["parent"] != parent or job["accepted_id"] != accepted:
            raise ValueError("Historical parent differs")
        names = {f"checkpoint-{checkpoint:06d}.json", f"checkpoint-{checkpoint:06d}.pt"}
        seeds = (42, 1337, 2026)
        names.update(f"config{i // 3}-seed{seeds[i % 3]}.pt" for i in range(completed))
        mapping = {}
        for item in job["files"]:
            prefix = f"{current}/research-{current}/"
            if not item["logical_path"].startswith(prefix):
                raise ValueError("Exact phase logical path required")
            basename = item["logical_path"][len(prefix) :]
            upload = item["upload_name"]
            if (
                basename in mapping
                or basename not in names
                or Path(upload).name != upload
                or "/" in upload
                or "\\" in upload
                or upload in native
            ):
                raise ValueError("Unique safe exact phase roles required")
            mapping[basename] = upload
            native[upload] = {"sha256": item["sha256"], "bytes": item["bytes"]}
        if set(mapping) != names:
            raise ValueError("All exact phase checkpoint siblings required")
        if current == phase:
            if selected is not None:
                raise ValueError("Duplicate phase provider")
            selected = mapping
    if len(provider["jobs"]) != 2 or selected is None or len(native) != 18:
        raise ValueError("Both complete opaque phase inventories required")
    if (
        native[selected[reference["snapshot"]]]["sha256"]
        != reference["snapshot_sha256"]
        or native[selected[Path(reference["snapshot"]).with_suffix(".json").name]][
            "sha256"
        ]
        != reference["header_sha256"]
    ):
        raise ValueError("Original audited snapshot/header hashes required")
    expected = {
        **native,
        "provider-provenance.json": freeze["remote_files"].get(
            "provider-provenance.json"
        ),
    }
    if (
        freeze["remote_files"] != expected
        or expected["provider-provenance.json"]["sha256"]
        != freeze["provider_provenance_sha256"]
    ):
        raise ValueError("Full externally bound nineteen-file inventory required")
    return selected


def run(package, input_root, output, expected):
    output = Path(output)
    output.mkdir(exist_ok=False)
    receipt = {
        "status": "started",
        "native_deserialized": False,
        "scientific_complete": False,
    }
    try:
        plan = authenticate(package, expected)
        source = output / "authenticated-source"
        source.mkdir()
        for name, entry in plan["files"].items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (Path(package) / entry["upload_name"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Package changed before materialization")
            path.write_bytes(raw)
        template = (source / "attestation-runner.py").read_bytes()
        token = b"EXPECTED_PLAN_SHA256" + b" = None"
        rendered = template.replace(
            token, ("EXPECTED_PLAN_SHA256 = " + repr(expected)).encode()
        )
        if (
            token not in template
            or Path(__file__).read_bytes().replace(b"\r\n", b"\n") != rendered
        ):
            raise ValueError("Executed runner differs")
        freeze = json.loads((source / "checkpoint-dataset-freeze.json").read_bytes())
        providers = [
            p
            for p in Path(input_root).rglob("provider-provenance.json")
            if not p.is_symlink() and sha(p) == freeze["provider_provenance_sha256"]
        ]
        if len(providers) != 1:
            raise ValueError("One exact frozen checkpoint dataset mount required")
        provider = providers[0]
        mapping = checkpoint_mapping(
            freeze, json.loads(provider.read_bytes()), plan["phase"], plan["reference"]
        )
        parent = provider.parent
        for name, entry in freeze["remote_files"].items():
            path = parent / name
            if (
                path.is_symlink()
                or path.stat().st_size != entry["bytes"]
                or sha(path) != entry["sha256"]
            ):
                raise ValueError("Frozen checkpoint dataset byte differs")
        spec = importlib.util.spec_from_file_location(
            "interrupted_recovery", source / "interrupted_checkpoint_recovery.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        receipt["actual_parent_phase_directory"] = str(parent)
        inventory = module.cloud_hash_inventory(
            source / plan["audit_path"],
            plan["audit_sha256"],
            plan["phase"],
            parent,
            native_paths=mapping,
        )
        authenticate(package, expected)
        for name, entry in freeze["remote_files"].items():
            path = parent / name
            if (
                path.is_symlink()
                or path.stat().st_size != entry["bytes"]
                or sha(path) != entry["sha256"]
            ):
                raise ValueError("Frozen checkpoint dataset changed during attestation")
        for name, entry in plan["files"].items():
            if sha(source / name) != entry["sha256"]:
                raise ValueError("Authenticated materialized source changed")
        (output / "interrupted-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        receipt.update(
            status="hash_inventory_complete",
            inventory_sha256=sha(output / "interrupted-inventory.json"),
        )
    except Exception as error:
        receipt.update(
            status="failed", error_type=type(error).__name__, error=str(error)[:2000]
        )
        (output / "failure-trace.txt").write_text(traceback.format_exc()[-8000:])
        raise
    finally:
        (output / "attestation-status.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


def main():
    if EXPECTED_PLAN_SHA256 is None:
        raise ValueError("Reviewed rendered runner required")
    packages = [
        path.parent
        for path in Path("/kaggle/input").rglob("attestation-plan.json")
        if sha(path) == EXPECTED_PLAN_SHA256
    ]
    if len(packages) != 1:
        raise ValueError("One exact authenticated transport mount required")
    authenticate(packages[0], EXPECTED_PLAN_SHA256)
    run(
        packages[0],
        "/kaggle/input",
        "/kaggle/working/interrupted-attestation",
        EXPECTED_PLAN_SHA256,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        Path("/kaggle/working/attestation-bootstrap-failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "native_deserialized": False,
                    "scientific_complete": False,
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                },
                indent=2,
            )
            + "\n"
        )
        raise
