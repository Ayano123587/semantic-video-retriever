"""Prospective CPU portability metadata contracts; no tensor/model imports.

Origin identities remain historical facts. This policy permits only the two
observed CPU descriptor profiles and never claims bitwise continuation.
"""

import copy
import hashlib
import json
import re
import stat
import sys
from pathlib import Path

INSPECTION_FREEZE_SHA = (
    "a6ccc7dc3a2aac9ba1528fcd14e39489ce4da2c07bb024559d28ab6c75c0ce56"
)
INSPECTION_REPORT_SHA = (
    "2e2951b2025a48fe9a3035c576ba96c0ed96c4146bc1d73b5944279451290e34"
)
AMD_OBSERVATION_SHA = "e3a96cc3a976619b8c475d8cf172e51477af10dbbad41891ec398d0b72bd69c1"
CPU_PATHS = (
    "cpu.linux_cpu.vendor_id",
    "cpu.linux_cpu.model name",
    "cpu.linux_cpu.flags",
)
DESCRIPTORS = ("vendor_id", "model name", "flags")


class PortabilityError(ValueError):
    """An unauthenticated artifact or an unapproved execution difference."""


def require(condition, message):
    if not condition:
        raise PortabilityError(message)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def read_bound_json(path, expected):
    require(valid_sha(expected), "External metadata SHA256 required")
    raw = Path(path).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected,
        "Externally frozen metadata changed",
    )
    return json.loads(raw)


def load_evidence(freeze_path, report_path, amd_observation_path):
    freeze = read_bound_json(freeze_path, INSPECTION_FREEZE_SHA)
    report = read_bound_json(report_path, INSPECTION_REPORT_SHA)
    amd = read_bound_json(amd_observation_path, AMD_OBSERVATION_SHA)
    require(
        freeze["report_sha256"] == INSPECTION_REPORT_SHA
        and freeze["training_authorized"] is False,
        "Actual inspection evidence is not training authorization",
    )
    require(
        report["training"] is False and report["scoring"] is False,
        "Metadata-only inspection required",
    )
    require(
        set(report["phases"]) == {"phase1", "phase2"}, "Both actual origins required"
    )
    for phase, metadata in report["phases"].items():
        require(
            fingerprint(metadata["origin_identity"])
            == freeze["phases"][phase]["origin_identity_sha256"],
            "Actual origin identity differs",
        )
        require(
            fingerprint(metadata["origin_spec"])
            == metadata["origin_identity"]["spec_sha256"],
            "Actual origin spec differs",
        )
    require(
        report["phases"]["phase1"]["origin_identity"]["runtime"]
        == report["phases"]["phase2"]["origin_identity"]["runtime"],
        "Observed origin runtimes differ",
    )
    require(
        amd["differing_keys"] == ["cpu_identity"],
        "Actual CPU-only observation required",
    )
    return {"freeze": freeze, "report": report, "amd_observation": amd}


def policy_description(evidence):
    """Deterministic proposal content, never a training approval or saved freeze."""
    origins = evidence["report"]["phases"]
    intel = origins["phase1"]["origin_identity"]["runtime"]["cpu"]["linux_cpu"]
    amd = evidence["amd_observation"]["actual"]["cpu_identity"]["cpuinfo"]
    return {
        "schema": "cpu_portability_protocol_v2",
        "origin_inspection_freeze_sha256": INSPECTION_FREEZE_SHA,
        "inspection_report_sha256": INSPECTION_REPORT_SHA,
        "amd_observation_sha256": AMD_OBSERVATION_SHA,
        "origin_identity_sha256": {
            p: fingerprint(v["origin_identity"]) for p, v in origins.items()
        },
        "origin_spec_sha256": {
            p: fingerprint(v["origin_spec"]) for p, v in origins.items()
        },
        "origin_source_manifest_sha256": origins["phase1"]["reference"][
            "source_manifest_sha256"
        ],
        "origin_protocol_sha256": origins["phase1"]["origin_identity"][
            "protocol_sha256"
        ],
        "cache_manifest_sha256": origins["phase1"]["origin_identity"][
            "cache_manifest_sha256"
        ],
        "allowed_cpu_descriptor_paths": list(CPU_PATHS),
        "allowed_cpu_profiles": [
            {k: source[k] for k in DESCRIPTORS} for source in (intel, amd)
        ],
        "all_other_runtime_fields": "exact_origin",
        "bitwise_equivalence_claimed": False,
        "training_math": "unchanged_original_engine_math",
    }


def authenticate_policy(path, expected, evidence):
    policy = read_bound_json(path, expected)
    require(
        policy == policy_description(evidence),
        "Exact prospectively frozen CPU policy required",
    )
    return policy


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = prefix + "." + str(key) if prefix else str(key)
            result.extend(
                [name]
                if key not in left or key not in right
                else differences(left[key], right[key], name)
            )
        return result
    return [] if type(left) is type(right) and left == right else [prefix]


def runtime_guard(actual, phase, policy, evidence):
    require(
        policy == policy_description(evidence),
        "Unchanged frozen portability policy required",
    )
    origin = evidence["report"]["phases"][phase]["origin_identity"]["runtime"]
    changed = differences(origin, actual)
    require(
        set(changed) <= set(CPU_PATHS),
        "Unapproved non-CPU runtime difference: " + ",".join(changed),
    )
    profile = {key: actual["cpu"]["linux_cpu"][key] for key in DESCRIPTORS}
    require(
        profile in policy["allowed_cpu_profiles"],
        "CPU profile was not actually observed and approved",
    )
    return {
        "actual_runtime_sha256": fingerprint(actual),
        "changed_paths": changed,
        "cpu_profile_sha256": fingerprint(profile),
        "bitwise_equivalence_claimed": False,
    }


def authenticate_service(
    path, expected, policy_sha256, phase, evidence, *, action="continue_training"
):
    require(
        action in ("continue_training", "restore_only"),
        "Explicit supported service action required",
    )
    service = read_bound_json(path, expected)
    required = {
        "schema",
        "training_authorized",
        "restore_authorized",
        "action",
        "phase",
        "portable_policy_sha256",
        "origin_identity_sha256",
        "origin_inspection_freeze_sha256",
        "source_manifest_sha256",
        "parent_schema",
        "parent_checkpoint_sha256",
        "parent_header_sha256",
        "parent_inventory_sha256",
        "parent_lineage_sha256",
        "parent_snapshot_name",
        "parent_native_paths",
        "max_seconds",
        "test_annotations_used",
    }
    require(
        set(service) == required
        and service["schema"] == "portable_training_service_v2",
        "Exact portable service contract required",
    )
    require(
        service["action"] == action
        and service["training_authorized"] is (action == "continue_training")
        and service["restore_authorized"] is (action == "restore_only")
        and service["phase"] == phase
        and service["max_seconds"] == (36000 if action == "continue_training" else 1800)
        and service["test_annotations_used"] is False,
        "Actual separate training authorization required",
    )
    require(
        service["portable_policy_sha256"] == policy_sha256
        and service["origin_inspection_freeze_sha256"] == INSPECTION_FREEZE_SHA
        and service["origin_identity_sha256"]
        == fingerprint(evidence["report"]["phases"][phase]["origin_identity"]),
        "Service origin/policy differs",
    )
    for key in (
        "source_manifest_sha256",
        "parent_checkpoint_sha256",
        "parent_header_sha256",
        "parent_inventory_sha256",
    ):
        require(
            valid_sha(service[key]),
            "Actual externally frozen service artifact SHA required",
        )
    require(
        type(service["parent_schema"]) is int and service["parent_schema"] in (1, 2),
        "Explicit parent checkpoint schema required",
    )
    require(
        (service["parent_schema"] == 1 and service["parent_lineage_sha256"] is None)
        or (
            service["parent_schema"] == 2
            and valid_sha(service["parent_lineage_sha256"])
        ),
        "Exact parent lineage pin required",
    )
    names = service["parent_native_paths"]
    require(
        isinstance(names, dict)
        and service["parent_snapshot_name"] in names
        and len(set(names.values())) == len(names)
        and all(
            isinstance(value, str)
            and Path(value).name == value
            and "/" not in value
            and "\\" not in value
            for value in [*names, *names.values()]
        ),
        "Exact unique safe native role mapping required",
    )
    return service


def authenticate_restore_service(path, expected, policy_sha256, phase, evidence):
    """A restore-only approval cannot pass the default training service gate."""
    return authenticate_service(
        path, expected, policy_sha256, phase, evidence, action="restore_only"
    )


def cloud_directory(path):
    path = Path(path)
    require(
        sys.platform == "linux"
        and path.is_absolute()
        and path.resolve().is_relative_to(Path("/kaggle/input")),
        "Native checkpoint handling is cloud-only",
    )
    for item in (path, *path.parents):
        require(
            not stat.S_ISLNK(item.lstat().st_mode), "Symlink checkpoint path forbidden"
        )
    require(path.is_dir(), "Mounted checkpoint directory required")
    return path


def authenticate_parent(resume, inventory_path, service, evidence, phase):
    resume = Path(resume)
    parent = cloud_directory(resume.parent)
    inventory = read_bound_json(inventory_path, service["parent_inventory_sha256"])
    files = inventory["files"]
    mapping = service["parent_native_paths"]
    require(
        set(mapping) == set(files)
        and resume.name == mapping[service["parent_snapshot_name"]],
        "Exact frozen native role map differs",
    )
    for name, entry in files.items():
        require(
            isinstance(name, str)
            and Path(name).name == name
            and "/" not in name
            and "\\" not in name,
            "Safe native sibling name required",
        )
        path = parent / mapping[name]
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == entry["bytes"]
            and sha(path) == entry["sha256"],
            "Externally frozen native sibling differs",
        )
    snapshot_name = service["parent_snapshot_name"]
    header_name = Path(snapshot_name).with_suffix(".json").name
    require(
        snapshot_name in files
        and header_name in files
        and sha(resume) == service["parent_checkpoint_sha256"]
        and sha(parent / mapping[header_name]) == service["parent_header_sha256"],
        "Actual checkpoint/header pair required",
    )
    header = json.loads((parent / mapping[header_name]).read_bytes())
    original = evidence["report"]["phases"][phase]
    if service["parent_schema"] == 1:
        expected = {
            "snapshot_sha256": original["reference"]["snapshot_sha256"],
            "identity_sha256": service["origin_identity_sha256"],
            "completed_seeds": original["completed_seeds"],
            "batches": original["batches"],
            "stage": original["stage"],
        }
        require(
            header == expected
            and service["parent_checkpoint_sha256"]
            == original["reference"]["snapshot_sha256"],
            "Exact inspected original checkpoint required",
        )
        require(
            set(files)
            == {
                snapshot_name,
                header_name,
                *[r["name"] for r in original["completed"]],
            },
            "Exact original prefix inventory required",
        )
        for record in original["completed"]:
            require(
                files[record["name"]]["sha256"] == record["sha256"],
                "Inspected original seed differs",
            )
    else:
        required = {
            "schema",
            "snapshot_sha256",
            "origin_identity_sha256",
            "portable_policy_sha256",
            "lineage_sha256",
            "execution_runtime_sha256",
            "completed_seeds",
            "batches",
            "stage",
        }
        require(
            set(header) == required
            and header["schema"] == 2
            and header["snapshot_sha256"] == service["parent_checkpoint_sha256"]
            and header["origin_identity_sha256"] == service["origin_identity_sha256"]
            and header["portable_policy_sha256"] == service["portable_policy_sha256"]
            and header["lineage_sha256"] == service["parent_lineage_sha256"],
            "Exact portable parent header/lineage required",
        )
        count = header["completed_seeds"]
        spec = original["origin_spec"]
        require(
            type(count) is int
            and original["completed_seeds"]
            <= count
            <= len(spec["configs"]) * len(spec["seeds"]),
            "Valid full-prefix completion count required",
        )
        seeds = [
            f"config{i // len(spec['seeds'])}-seed{spec['seeds'][i % len(spec['seeds'])]}.pt"
            for i in range(count)
        ]
        require(
            set(files) == {snapshot_name, header_name, *seeds},
            "Exact portable sibling prefix required",
        )
    return inventory, header


def validate_lineage(events, phase, policy, evidence, policy_sha256):
    require(
        isinstance(events, list) and 0 < len(events) <= 1024,
        "Bounded complete execution lineage required",
    )
    previous = None
    previous_batches = -1
    fields = {
        "schema",
        "origin_identity_sha256",
        "portable_policy_sha256",
        "source_manifest_sha256",
        "parent_checkpoint_sha256",
        "parent_header_sha256",
        "previous_event_sha256",
        "actual_runtime",
        "actual_runtime_sha256",
        "start_progress",
    }
    for event in events:
        require(
            set(event) == fields
            and event["schema"] == "cpu_execution_lineage_event_v2",
            "Exact execution lineage event required",
        )
        require(
            event["previous_event_sha256"] == previous
            and event["portable_policy_sha256"] == policy_sha256
            and event["origin_identity_sha256"]
            == policy["origin_identity_sha256"][phase],
            "Execution lineage chain differs",
        )
        runtime_guard(event["actual_runtime"], phase, policy, evidence)
        require(
            event["actual_runtime_sha256"] == fingerprint(event["actual_runtime"]),
            "Actual execution runtime fingerprint differs",
        )
        for name in (
            "source_manifest_sha256",
            "parent_checkpoint_sha256",
            "parent_header_sha256",
        ):
            require(valid_sha(event[name]), "Lineage artifact SHA required")
        progress = event["start_progress"]
        require(
            set(progress)
            == {"batches", "grid", "seed_index", "stage", "epoch", "cursor"},
            "Exact starting cursor metadata required",
        )
        spec = evidence["report"]["phases"][phase]["origin_spec"]
        require(
            all(
                type(progress[key]) is int and progress[key] >= 0
                for key in ("batches", "grid", "seed_index")
            )
            and progress["batches"] >= previous_batches
            and progress["grid"] <= len(spec["configs"])
            and progress["seed_index"] <= len(spec["seeds"])
            and progress["stage"] in ("fit", "candidate", "final")
            and (
                (progress["epoch"] is None and progress["cursor"] is None)
                or (
                    type(progress["epoch"]) is int
                    and progress["epoch"] >= 1
                    and type(progress["cursor"]) is int
                    and 0 <= progress["cursor"] <= 23202
                )
            ),
            "Invalid or regressing execution cursor",
        )
        if previous is None:
            original = evidence["report"]["phases"][phase]
            require(
                event["parent_checkpoint_sha256"]
                == original["reference"]["snapshot_sha256"]
                and event["parent_header_sha256"]
                == original["reference"]["header_sha256"],
                "Lineage must begin at inspected original bytes",
            )
            require(
                progress
                == {
                    "batches": original["batches"],
                    "grid": original["grid"],
                    "seed_index": original["seed_index"],
                    "stage": original["stage"],
                    "epoch": original["active"]["epoch"]
                    if original["active"]
                    else None,
                    "cursor": original["active"]["cursor"]
                    if original["active"]
                    else None,
                },
                "First execution cursor differs from actual inspection",
            )
        previous_batches = progress["batches"]
        previous = fingerprint(event)
    return previous


def append_execution(
    events, service, actual_runtime, progress, phase, policy, evidence
):
    result = copy.deepcopy(events)
    previous = (
        validate_lineage(
            result, phase, policy, evidence, service["portable_policy_sha256"]
        )
        if result
        else None
    )
    event = {
        "schema": "cpu_execution_lineage_event_v2",
        "origin_identity_sha256": service["origin_identity_sha256"],
        "portable_policy_sha256": service["portable_policy_sha256"],
        "source_manifest_sha256": service["source_manifest_sha256"],
        "parent_checkpoint_sha256": service["parent_checkpoint_sha256"],
        "parent_header_sha256": service["parent_header_sha256"],
        "previous_event_sha256": previous,
        "actual_runtime": copy.deepcopy(actual_runtime),
        "actual_runtime_sha256": fingerprint(actual_runtime),
        "start_progress": copy.deepcopy(progress),
    }
    result.append(event)
    validate_lineage(result, phase, policy, evidence, service["portable_policy_sha256"])
    return result
