"""Distinct cloud recovery gates for the two audited cancelled training parents.

No graceful-pause receipt is invented. Gate A hashes native bytes without
deserializing; Gate B requires a later external root freeze of that inventory.
The unchanged trainer remains responsible for optimizer/RNG/cursor restoration.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
import time
from pathlib import Path

SOURCE_SHA = "265b18be07ec7ec87d60231175963eeafe63e3fc6547a019e206517e2bcc0a76"
CACHE_SHA = "ea73dced79e83cfc735652d6f8e702011d4e663c3fd9307ef010f66b9120a785"
PROTOCOL_SHA = "d81c9ef5aa881b04440806702312023135aea8eab9b6a4ba9c10fcab335a39a3"
PARENTS = {
    "phase1": {
        "kernel": "aikyatansinha/portfolio-didemo-full-research-phase1-v1/2",
        "accepted_id": 139178784,
        "checkpoint": 47,
        "completed": 8,
        "batches": 40929,
        "header_sha256": "40df1a7598b01a6c804d569b0fe260e907a93fcd7fb0598e21dedda076ceec2d",
        "snapshot_sha256": "d832fe6482f8aebf1e9aa44ade5b772612921d07b6ef3c88f3c9569a9d378e11",
        "identity_sha256": "8132f0004b232f7a6e305767dd84546c29cd956781cd58902291278868867ee2",
    },
    "phase2": {
        "kernel": "aikyatansinha/portfolio-didemo-full-research-phase2-v1/3",
        "accepted_id": 139178893,
        "checkpoint": 53,
        "completed": 6,
        "batches": 22028,
        "header_sha256": "3e87553f3146183077b28ef348b197cc4eee557655caca0b8f9b916ef7d9f8ee",
        "snapshot_sha256": "c27286fd32a281b834c0514ba1a1ce70b1eee6bb4b28b016377dcf160bc7ebca",
        "identity_sha256": "1f82bb6d2b0b199a7a622f2a690634feb707b1924d5f2978ddea2bdc0700b548",
    },
}


class RecoveryError(ValueError):
    """An unauthenticated or scientifically incompatible recovery input."""


def _require(condition, message):
    if not condition:
        raise RecoveryError(message)


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, expected):
    _require(_valid_sha(expected), "Externally recorded metadata SHA256 required")
    raw = Path(path).read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == expected,
        "Externally frozen metadata differs",
    )
    return json.loads(raw)


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_reference(audit_path, audit_sha256, phase):
    """Validate only actual small audit/header/source metadata; never touch .pt."""
    _require(phase in PARENTS, "Exact audited phase required")
    report = _json(audit_path, audit_sha256)
    expected = PARENTS[phase]
    jobs = [job for job in report["jobs"] if job["canonical"] == expected["kernel"]]
    _require(len(jobs) == 1, "One actual cancelled parent required")
    job = jobs[0]
    _require(
        job["accepted_id"] == expected["accepted_id"]
        and job["current_status"] == {"status": "CANCEL_ACKNOWLEDGED"}
        and job["accepted_source_matches_server"] is True
        and job["scientific_completion"] is False
        and job["graceful_pause_receipt_available"] is False
        and job["explicit_terminal_artifacts_present"] == [],
        "Interrupted parent evidence differs",
    )
    bindings = job["source_bindings"]
    _require(
        bindings["SOURCE_SHA"] == SOURCE_SHA
        and bindings["CACHE_SHA"] == CACHE_SHA
        and bindings["PROTOCOL_SHA"] == PROTOCOL_SHA,
        "Frozen scientific parent differs",
    )
    header = job["latest_checkpoint_header"]
    snapshot = f"checkpoint-{expected['checkpoint']:06d}.pt"
    _require(
        header["path"]
        == f"research-{phase}/" + Path(snapshot).with_suffix(".json").name
        and header["sha256"] == expected["header_sha256"],
        "Latest audited checkpoint differs",
    )
    metadata = {
        "snapshot_sha256": expected["snapshot_sha256"],
        "identity_sha256": expected["identity_sha256"],
        "completed_seeds": expected["completed"],
        "batches": expected["batches"],
        "stage": "fit",
    }
    _require(
        header["metadata"] == metadata, "Audited checkpoint header identity differs"
    )
    evidence = {item["path"]: item for item in report["evidence_files"]}
    names = [
        f"{phase}/server-source.py",
        f"{phase}/server-header.json",
        f"{phase}/status.json",
        f"{phase}/output-inventory.json",
        f"{phase}/small-metadata/" + header["path"],
    ]
    parent = Path(audit_path).parent
    for name in names:
        _require(name in evidence, "Explicit small evidence hash required")
        path = parent / name
        _require(
            path.stat().st_size == evidence[name]["bytes"]
            and _sha(path) == evidence[name]["sha256"],
            "Actual small audit evidence changed",
        )
    actual_header = json.loads((parent / names[-1]).read_bytes())
    _require(actual_header == metadata, "Downloaded audited header differs")
    seeds = (42, 1337, 2026)
    completed = [
        f"config{i // 3}-seed{seeds[i % 3]}.pt" for i in range(expected["completed"])
    ]
    _require(
        set(job["completed_seed_artifacts_listed"])
        == {f"research-{phase}/" + name for name in completed},
        "Completed seed prefix differs",
    )
    return {
        "phase": phase,
        "kernel_source": expected["kernel"],
        "accepted_id": expected["accepted_id"],
        "terminal_status": "CANCEL_ACKNOWLEDGED",
        "snapshot": snapshot,
        "header_sha256": expected["header_sha256"],
        "snapshot_sha256": expected["snapshot_sha256"],
        "training_identity_sha256": expected["identity_sha256"],
        "completed_seed_files": completed,
        "completed_seeds": expected["completed"],
        "batches": expected["batches"],
        "stage": "fit",
        "root_audit_sha256": audit_sha256,
        "source_manifest_sha256": SOURCE_SHA,
        "cache_manifest_sha256": CACHE_SHA,
        "protocol_sha256": PROTOCOL_SHA,
    }


def _cloud_parent(path):
    path = Path(path)
    _require(
        sys.platform == "linux"
        and path.is_absolute()
        and path.resolve().is_relative_to(Path("/kaggle/input").resolve()),
        "Native checkpoint authentication is cloud-only under /kaggle/input",
    )
    for item in (path, *path.parents):
        info = item.lstat()
        _require(not stat.S_ISLNK(info.st_mode), "Symlink parent forbidden")
    _require(path.is_dir(), "Actual mounted parent directory required")
    return path


def _required(reference):
    return [
        reference["snapshot"],
        Path(reference["snapshot"]).with_suffix(".json").name,
        *reference["completed_seed_files"],
    ]


def cloud_hash_inventory(
    audit_path, audit_sha256, phase, parent_phase_directory, native_paths=None
):
    """Gate A: return real native hashes; no tensor/array/model deserialization."""
    reference = audit_reference(audit_path, audit_sha256, phase)
    required = _required(reference)
    if native_paths is not None:
        _require(
            isinstance(native_paths, dict) and set(native_paths) == set(required),
            "Exact phase snapshot/header/completed-prefix mapping required",
        )
        _require(
            len(set(native_paths.values())) == len(required)
            and all(
                isinstance(name, str)
                and Path(name).name == name
                and name not in ("", ".", "..")
                and "/" not in name
                and "\\" not in name
                for name in native_paths.values()
            ),
            "Unique safe flat native filenames required",
        )
    parent = _cloud_parent(parent_phase_directory)
    files = {}
    for name in required:
        path = parent / (native_paths[name] if native_paths is not None else name)
        _require(
            path.is_file() and not path.is_symlink(),
            "Required immutable sibling missing",
        )
        files[name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    _require(
        files[reference["snapshot"]]["sha256"] == reference["snapshot_sha256"]
        and files[Path(reference["snapshot"]).with_suffix(".json").name]["sha256"]
        == reference["header_sha256"],
        "Actual native snapshot/header differs from cancelled parent",
    )
    _require(
        not (parent / "continuation.json").exists()
        and not (parent / "results.json").exists()
        and not (parent / "selected.pt").exists(),
        "Do not misclassify completed or paused artifacts",
    )
    return {
        "schema": "interrupted_checkpoint_hash_inventory_v1",
        "reference": reference,
        "files": files,
        "native_deserialized": False,
        "scientific_complete": False,
        "resume_authorized": False,
    }


def authenticate_frozen_inventory(
    audit_path,
    audit_sha256,
    phase,
    parent_phase_directory,
    inventory_path,
    inventory_sha256,
    native_paths=None,
):
    """Gate B byte barrier; inventory hash must come from the later root freeze."""
    inventory = _json(inventory_path, inventory_sha256)
    _require(
        inventory.get("schema") == "interrupted_checkpoint_hash_inventory_v1"
        and inventory.get("native_deserialized") is False
        and inventory.get("scientific_complete") is False
        and inventory.get("resume_authorized") is False,
        "Hash-only interrupted inventory required",
    )
    actual = cloud_hash_inventory(
        audit_path, audit_sha256, phase, parent_phase_directory, native_paths
    )
    _require(actual == inventory, "Root-frozen complete sibling byte inventory differs")
    return inventory


def guarded_recover(
    contract_path,
    contract_sha256,
    *,
    audit_path,
    audit_sha256,
    phase,
    parent_phase_directory,
    inventory_path,
    inventory_sha256,
    training_identity,
    resume_callback,
    service_started,
    native_paths=None,
):
    """Authorize the unchanged cloud trainer callback only after every byte guard.

    The callback must use checked_resume with the supplied original snapshot,
    preserve optimizer/permutation/RNG/batch cursor, and allocate fresh output.
    No continuation.json or graceful-pause status is synthesized by this API.
    """
    _require(
        isinstance(service_started, (int, float))
        and 0 <= service_started <= time.monotonic(),
        "Original service startup monotonic timestamp required",
    )
    contract = _json(contract_path, contract_sha256)
    reference = audit_reference(audit_path, audit_sha256, phase)
    expected = {
        "schema": "interrupted_checkpoint_recovery_v1",
        "action": "resume_exact_interrupted_checkpoint",
        "reference": reference,
        "inventory_sha256": inventory_sha256,
        "max_seconds": 36000,
        "test_annotations_used": False,
        "restart_from_scratch": False,
    }
    _require(
        contract == expected,
        "Exact externally frozen interrupted recovery contract required",
    )
    _require(
        isinstance(training_identity, dict)
        and _fingerprint(training_identity) == reference["training_identity_sha256"]
        and training_identity.get("phase") == phase
        and training_identity.get("protocol_sha256") == PROTOCOL_SHA
        and training_identity.get("cache_manifest_sha256") == CACHE_SHA
        and isinstance(training_identity.get("runtime"), dict)
        and bool(training_identity["runtime"]),
        "Original complete training identity/runtime required",
    )
    authenticate_frozen_inventory(
        audit_path,
        audit_sha256,
        phase,
        parent_phase_directory,
        inventory_path,
        inventory_sha256,
        native_paths,
    )
    parent = _cloud_parent(parent_phase_directory)
    remaining = contract["max_seconds"] - (time.monotonic() - service_started)
    _require(remaining > 0, "Startup-inclusive service budget exhausted")
    try:
        result = resume_callback(
            resume=parent
            / (
                native_paths[reference["snapshot"]]
                if native_paths is not None
                else reference["snapshot"]
            ),
            resume_sha256=reference["snapshot_sha256"],
            max_seconds=remaining,
        )
    finally:
        # A mutable mounted path cannot silently change while a consumer loads.
        authenticate_frozen_inventory(
            audit_path,
            audit_sha256,
            phase,
            parent_phase_directory,
            inventory_path,
            inventory_sha256,
            native_paths,
        )
        _require(
            _sha(contract_path) == contract_sha256, "Root recovery contract changed"
        )
    return result
